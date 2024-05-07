import math
import torch
import numpy as np
import torch.nn as nn
from typing import Optional
from einops import rearrange
import torch.nn.functional as F
from timm.models.layers import to_2tuple
from models.mamba_local.mamba.local_scan import local_scan, local_scan_bchw, local_reverse

__all__ = ["Super_Linear", "Super_Norm", "Super_Conv1d", "Super_PatchEmbed", "Super_MultiScan", "Super_BiAttn"]


def calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def rms_norm_ref(x, weight, bias, eps):
    rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
    out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    return out


class Super_Linear(nn.Module):
    def __init__(self, super_in_dim, super_out_dim, bias=True, scale=False):
        super().__init__()

        self.weight = nn.Parameter(torch.empty((super_out_dim, super_in_dim), ))
        if bias:
            self.bias = nn.Parameter(torch.empty(super_out_dim, ))
        else:
            self.bias = None
        self.reset_parameters()

        # super_in_dim and super_out_dim indicate the largest network!
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim

        # input_dim and output_dim indicate the current sampled size
        self.sample_in_dim = None
        self.sample_out_dim = None
        self.sample_scale = 1.

        self.samples = {}

        self.scale = scale

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def set_sample_config(self, sample_in_dim=None, sample_out_dim=None):
        if sample_in_dim is None:
            sample_in_dim = self.super_in_dim
        if sample_out_dim is None:
            sample_out_dim = self.super_out_dim
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim
        self.samples['weight'] = self.weight[:sample_out_dim, :sample_in_dim]
        if self.bias is not None:
            self.samples['bias'] = self.bias[:sample_out_dim]
        else:
            self.samples['bias'] = None
        if self.scale:
            self.sample_scale = self.sample_in_dim / self.super_in_dim

    def forward(self, x):
        return F.linear(x, self.samples['weight'], self.samples['bias']) * self.sample_scale

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        weight_numel = self.samples['weight'].numel()

        if self.samples['bias'] is not None:
            bias_numel = self.samples['bias'].numel()
        else:
            bias_numel = 0

        return weight_numel + bias_numel

    def get_complexity(self, sequence_length):
        return sequence_length * np.prod(self.samples['weight'].size())


class Super_Conv1d(nn.Module):
    def __init__(self, channels, kernel_size, bias=True, scale=False):
        super().__init__()
        self.super_channels = channels
        self.super_kernel_size = kernel_size

        self.weight = nn.Parameter(
            torch.empty((self.super_channels, 1, self.super_kernel_size)))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.super_channels, ))
        else:
            self.bias = None
        self.reset_parameters()

        self.sample_channels = None
        self.sample_kernel_size = None
        self.sample_padding = None
        self.sample_scale = None
        self.scale = scale
        self.samples = {}

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.conv1d(input=input, weight=self.samples['weight'], bias=self.samples['bias'],
                        padding=self.sample_padding, groups=self.sample_channels)

    def set_sample_config(self, sample_channels=None, sample_kernel_size=None):
        if sample_channels is None:
            sample_channels = self.super_channels
        if sample_kernel_size is None:
            sample_kernel_size = self.super_kernel_size

        self.sample_channels = sample_channels
        self.sample_kernel_size = sample_kernel_size
        self.sample_padding = self.sample_kernel_size - 1
        self.samples['weight'] = self.weight[:self.sample_channels, :, :self.sample_kernel_size]
        if self.bias is not None:
            self.samples['bias'] = self.bias[:self.sample_channels]
        else:
            self.samples['bias'] = None
        if self.scale:
            self.sample_scale = self.sample_channels / self.super_channels

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        weight_numel = self.samples['weight'].numel()

        if self.samples['bias'] is not None:
            bias_numel = self.samples['bias'].numel()
        else:
            bias_numel = 0

        return weight_numel + bias_numel

    def get_complexity(self, sequence_length):
        return sequence_length * np.prod(self.samples['weight'].size())


class Super_Norm(nn.Module):
    def __init__(self, super_embed_dim, bias=True, eps=1e-5, rms_norm=False, elementwise_affine=True):
        super().__init__()

        # the largest embed dim
        self.super_embed_dim = super_embed_dim
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.super_embed_dim, ))
            if bias:
                self.bias = nn.Parameter(torch.zeros(self.super_embed_dim, ))
            else:
                self.bias = None

            # the current sampled embed dim
            self.sample_embed_dim = None
            self.samples = {}

            self.rms_norm = rms_norm

        self.eps = eps

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.super_embed_dim, ))
            if self.bias is not None:
                self.bias = nn.Parameter(torch.zeros(self.super_embed_dim, ))

    def set_sample_config(self, sample_embed_dim=None):
        if sample_embed_dim is None:
            sample_embed_dim = self.super_embed_dim
        self.sample_embed_dim = sample_embed_dim
        if self.elementwise_affine:
            self.samples['weight'] = self.weight[:self.sample_embed_dim]
            if self.bias is not None:
                self.samples['bias'] = self.bias[:self.sample_embed_dim]
            else:
                self.samples['bias'] = None

    def forward(self, x, residual=None, prenorm=False):
        if residual is not None:
            x = x + residual
        if self.elementwise_affine:
            if not self.rms_norm:
                out = F.layer_norm(x, (self.sample_embed_dim,), weight=self.samples['weight'],
                                   bias=self.samples['bias'], eps=self.eps)
            else:
                out = rms_norm_ref(x, weight=self.samples['weight'], bias=self.samples['bias'], eps=self.eps)
        else:
            out = F.layer_norm(x, (self.sample_embed_dim,), weight=None, bias=None, eps=self.eps)
        return out if not prenorm else (out, x)

    def calc_sampled_param_num(self):
        if self.elementwise_affine:
            assert 'weight' in self.samples.keys()
            weight_numel = self.samples['weight'].numel()

            if self.samples['bias'] is not None:
                bias_numel = self.samples['bias'].numel()
            else:
                bias_numel = 0
            return weight_numel + bias_numel
        else:
            return 0

    def get_complexity(self, sequence_length):
        if self.elementwise_affine:
            return sequence_length * self.sample_embed_dim
        else:
            return 0


class Super_PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(
            self,
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm: bool = False,
            rms_norm: bool = False,
            flatten: bool = True,
            strict_img_size: bool = False,
            dynamic_img_pad: bool = True,
            scale: bool = False
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.img_size = to_2tuple(img_size)
        self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # flatten spatial dim and transpose to channels last, kept for bwd compat
        self.flatten = flatten
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

        self.norm = norm
        if self.norm:
            self.super_norm = Super_Norm(super_embed_dim=embed_dim, rms_norm=rms_norm)

        # sample
        self.scale = scale
        self.super_embed_dim = embed_dim
        self.sample_embed_dim = None
        self.sampled_weight = None
        self.sampled_bias = None
        self.sampled_scale = None

    def set_sample_config(self, sample_embed_dim=None):
        if sample_embed_dim is None:
            self.sample_embed_dim = self.super_embed_dim
        self.sampled_weight = self.proj.weight[:self.sample_embed_dim, ...]
        self.sampled_bias = self.proj.bias[:self.sample_embed_dim, ...]
        if self.scale:
            self.sampled_scale = self.super_embed_dim / self.sample_embed_dim
        if self.norm:
            self.super_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                assert H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]})."
                assert W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]})."
            elif not self.dynamic_img_pad:
                assert H % self.patch_size[0] == 0, \
                    f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
                assert W % self.patch_size[1] == 0, \
                    f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h))
        # x = self.proj(x)
        x = F.conv2d(input=x, weight=self.sampled_weight, stride=self.patch_size, bias=self.sampled_bias)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        if self.scale:
            return x * self.sampled_scale
        if self.norm:
            x = self.super_norm(x)
        return x

    def calc_sampled_param_num(self):
        total_param_num = self.sampled_weight.numel() + self.sampled_bias.numel()
        if self.norm:
            total_param_num += self.super_norm.calc_sampled_param_num()
        return total_param_num

    def get_complexity(self, sequence_length):
        total_flops = self.sampled_bias.size(0) + sequence_length * np.prod(self.sampled_weight.size())
        if self.norm:
            total_flops += self.super_norm.get_complexity(sequence_length)
        return total_flops


class Super_MultiScan(nn.Module):
    ALL_CHOICES = ('h', 'h_flip', 'v', 'v_flip')

    # ALL_CHOICES = ('h', 'h_flip', 'v', 'v_flip', 'w2', 'w2_flip', 'w7', 'w7_flip')

    def __init__(self, super_in_dim, choices=None, token_size=(14, 14)):
        super().__init__()
        self.token_size = token_size
        self.super_in_dim = super_in_dim
        if choices is None:
            self.choices = Super_MultiScan.ALL_CHOICES
            self.super_norms = nn.ModuleList(
                [Super_Norm(self.super_in_dim, elementwise_affine=False) for _ in self.choices]
            )
            self.weights = nn.Parameter(1e-3 * torch.randn(len(self.choices), 1, 1, 1))
            self.search = True
        else:
            self.choices = choices
            self.super_norms = None
            self.weights = None
            self.search = False

        self.sample_in_dim = None

    def forward(self, xs):
        """
        Input @xs: [[B, L, D], ...]
        """
        if self.search:
            weights = self.weights.softmax(0)
            xs = [norm(x) for norm, x in zip(self.super_norms, xs)]
            xs = torch.stack(xs) * weights
            x = xs.sum(0)
        else:
            x = torch.stack(xs).sum(0)
        return x

    def multi_scan(self, x):
        """
        Input @x: shape [B, L, D]
        """
        xs = []
        for direction in self.choices:
            xs.append(self.scan(x, direction))
        return xs

    def multi_reverse(self, xs):
        new_xs = []
        for x, direction in zip(xs, self.choices):
            new_xs.append(self.reverse(x, direction))
        return new_xs

    def scan(self, x, direction='h'):
        """
        Input @x: shape [B, L, D] or [B, C, H, W]
        Return torch.Tensor: shape [B, D, L]
        """
        H, W = self.token_size
        if len(x.shape) == 3:
            if direction == 'h':
                return x.transpose(-2, -1)
            elif direction == 'h_flip':
                return x.transpose(-2, -1).flip([-1])
            elif direction == 'v':
                return rearrange(x, 'b (h w) d -> b d (w h)', h=H, w=W)
            elif direction == 'v_flip':
                return rearrange(x, 'b (h w) d -> b d (w h)', h=H, w=W).flip([-1])
            elif direction.startswith('w'):
                K = int(direction[1:].split('_')[0])
                flip = direction.endswith('flip')
                return local_scan(x, K, H, W, flip=flip)
            else:
                raise RuntimeError(f'Direction {direction} not found.')
        elif len(x.shape) == 4:
            if direction == 'h':
                return x.flatten(2)
            elif direction == 'h_flip':
                return x.flatten(2).flip([-1])
            elif direction == 'v':
                return rearrange(x, 'b d h w -> b d (w h)', h=H, w=W)
            elif direction == 'v_flip':
                return rearrange(x, 'b d h w -> b d (w h)', h=H, w=W).flip([-1])
            elif direction.startswith('w'):
                K = int(direction[1:].split('_')[0])
                flip = direction.endswith('flip')
                return local_scan_bchw(x, K, H, W, flip=flip)
            else:
                raise RuntimeError(f'Direction {direction} not found.')

    def reverse(self, x, direction='h'):
        """
        Input @x: shape [B, D, L]
        Return torch.Tensor: shape [B, D, L]
        """
        H, W = self.token_size
        if direction == 'h':
            return x
        elif direction == 'h_flip':
            return x.flip([-1])
        elif direction == 'v':
            return rearrange(x, 'b d (h w) -> b d (w h)', h=H, w=W)
        elif direction == 'v_flip':
            return rearrange(x.flip([-1]), 'b d (h w) -> b d (w h)', h=H, w=W)
        elif direction.startswith('w'):
            K = int(direction[1:].split('_')[0])
            flip = direction.endswith('flip')
            return local_reverse(x, K, H, W, flip=flip)
        else:
            raise RuntimeError(f'Direction {direction} not found.')

    def __repr__(self):
        scans = ', '.join(self.choices)
        return super().__repr__().replace(self.__class__.__name__, f'{self.__class__.__name__}[{scans}]')

    def set_sample_config(self, sample_in_dim=None):
        if sample_in_dim is None:
            sample_in_dim = self.super_in_dim
        self.sample_in_dim = sample_in_dim
        if self.super_norms is not None:
            for super_norm in self.super_norms:
                super_norm.set_sample_config(sample_embed_dim=self.sample_in_dim)


class Super_BiAttn(nn.Module):
    def __init__(self, super_in_channels, act_ratio=0.125, act_fn=nn.GELU, gate_fn=nn.Sigmoid):
        super().__init__()
        self.reduce_channels = int(super_in_channels * act_ratio)
        self.super_in_channels = super_in_channels

        self.super_norm = Super_Norm(super_embed_dim=self.super_in_channels)
        self.super_global_reduce = Super_Linear(super_in_dim=self.super_in_channels,
                                                super_out_dim=self.reduce_channels)
        self.super_local_reduce = Super_Linear(super_in_dim=self.super_in_channels,
                                               super_out_dim=self.reduce_channels)

        self.super_channel_select = Super_Linear(super_in_dim=self.reduce_channels,
                                                 super_out_dim=self.super_in_channels)
        self.super_spatial_select = Super_Linear(super_in_dim=self.reduce_channels * 2,
                                                 super_out_dim=1)

        self.act_fn = act_fn()
        self.gate_fn = gate_fn()

        self.sample_in_channels = None

    def forward(self, x):
        ori_x = x
        x = self.super_norm(x)
        x_global = x.mean(1, keepdim=True)
        x_global = self.act_fn(self.super_global_reduce(x_global))
        x_local = self.act_fn(self.super_local_reduce(x))

        c_attn = self.gate_fn(self.super_channel_select(x_global))  # [B, 1, C]
        s_attn = self.gate_fn(
            self.super_spatial_select(
                torch.cat([x_local, x_global.expand(-1, x.shape[1], -1)], dim=-1)
            )
        )  # [B, N, 1]

        attn = c_attn * s_attn  # [B, N, C]
        return ori_x * attn

    def set_sample_config(self, sample_in_channels=None):
        if sample_in_channels is None:
            sample_in_channels = self.super_sample_in_channels
        self.sample_in_channels = sample_in_channels
        self.super_norm.set_sample_config(sample_embed_dim=self.sample_in_channels)
        self.super_global_reduce.set_sample_config(sample_in_dim=self.sample_in_channels,
                                                   sample_out_dim=self.reduce_channels)
        self.super_local_reduce.set_sample_config(sample_in_dim=self.sample_in_channels,
                                                  sample_out_dim=self.reduce_channels)
        self.super_channel_select.set_sample_config(sample_in_dim=self.reduce_channels,
                                                    sample_out_dim=self.sample_in_channels)
        self.super_spatial_select.set_sample_config(sample_in_dim=self.reduce_channels * 2,
                                                    sample_out_dim=1)

    def calc_sampled_param_num(self):
        total_param_num = 0
        total_param_num += self.super_norm.calc_sampled_param_num()
        total_param_num += self.super_global_reduce.calc_sampled_param_num()
        total_param_num += self.super_local_reduce.calc_sampled_param_num()
        total_param_num += self.super_channel_select.calc_sampled_param_num()
        total_param_num += self.super_spatial_select.calc_sampled_param_num()
        return total_param_num

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += self.super_norm.get_complexity(sequence_length=sequence_length)
        total_flops += self.super_global_reduce.get_complexity(sequence_length=sequence_length)
        total_flops += self.super_local_reduce.get_complexity(sequence_length=sequence_length)
        total_flops += self.super_channel_select.get_complexity(sequence_length=sequence_length)
        total_flops += self.super_spatial_select.get_complexity(sequence_length=sequence_length)
        return total_flops
