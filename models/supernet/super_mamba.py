import math
import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
from typing import Optional
from einops import rearrange, repeat
from timm.models.layers import DropPath
from models.supernet.super_ops import *
from models.mamba_local.ops.selective_scan_interface import mamba_inner_fn_no_out_proj


class Super_GLambaBlock(nn.Module):
    def __init__(
            self,
            embed_dim,
            d_state=16,
            c_kernel_size=16,
            expand_ratio=2.,
            num_head=8,
            mix_type="mh",
            mamba_ratio=0.5,
            directions=None,
            kernel_size=4,
            conv_bias=True,
            bias=False,
            layer_idx=None,
            token_size=(14, 14),
            norm_epsilon=1e-5,
            drop_path=0.,
            rms_norm=False,
            scale=False,
    ):
        super().__init__()
        assert mix_type in ["none", "og", "mh"]
        self.super_embed_dim = embed_dim
        self.super_expand_ratio = expand_ratio
        self.super_ffn_embed_dim_this_layer = int(expand_ratio * embed_dim)
        self.super_dropout = drop_path
        self.super_d_state = d_state
        self.super_kernel_size = kernel_size
        self.super_num_head = num_head

        self.super_mamba_ratio = mamba_ratio
        self.super_c_kernel_size = c_kernel_size

        self.is_identity_layer = None
        self.scale = scale

        self.sample_embed_dim = None
        self.sample_expand_ratio = None
        self.sample_ffn_embed_dim_this_layer = None
        self.sample_scale = None
        self.sample_dropout = None
        self.sample_d_state = None
        self.sample_kernel_size = None

        self.sample_mamba_ratio = None
        self.sample_c_kernel_size = None
        self.sample_num_head = None

        self.super_norm = Super_Norm(self.super_embed_dim, eps=norm_epsilon, rms_norm=rms_norm)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mix_type = mix_type
        if self.mix_type == "none":
            self.super_mamba = Super_MultiMamba(
                embed_dim=self.super_embed_dim,
                d_state=self.super_d_state,
                kernel_size=self.super_kernel_size,
                expand_ratio=self.super_expand_ratio,
                directions=directions,
                conv_bias=conv_bias,
                bias=bias,
                layer_idx=layer_idx,
                token_size=token_size,
            )
        else:
            self.super_mamba = Super_GLamba(
                embed_dim=self.super_embed_dim,
                d_state=self.super_d_state,
                mamba_type=self.mix_type,
                num_head=self.super_num_head,
                kernel_size=self.super_kernel_size,
                expand_ratio=self.super_expand_ratio,
                directions=directions,
                conv_bias=conv_bias,
                bias=bias,
                layer_idx=layer_idx,
                token_size=token_size,
                mamba_ratio=self.super_mamba_ratio,
                c_kernel_size=self.super_c_kernel_size
            )

    def set_sample_config(self, is_identity_layer, sample_embed_dim=None, sample_expand_ratio=None,
                          sample_d_state=None, sample_kernel_size=4, sample_dt_rank="auto",
                          sample_c_kernel_size=None, sample_mamba_ratio=None, sample_num_head=None):
        if is_identity_layer:
            self.is_identity_layer = True
            return
        self.is_identity_layer = False
        self.sample_num_head = sample_num_head or self.super_num_head
        self.sample_embed_dim = sample_embed_dim or self.super_embed_dim
        self.sample_expand_ratio = sample_expand_ratio or self.super_mamba_ratio
        self.sample_kernel_size = sample_kernel_size or self.super_kernel_size
        self.sample_d_state = sample_d_state or self.super_d_state
        self.sample_mamba_ratio = sample_mamba_ratio or self.super_mamba_ratio
        self.sample_c_kernel_size = sample_c_kernel_size or self.super_c_kernel_size
        self.super_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)
        if self.mix_type == "none":
            self.super_mamba.set_sample_config(
                sample_embed_dim=self.sample_embed_dim, sample_expand_ratio=self.sample_expand_ratio,
                sample_kernel_size=self.sample_kernel_size, sample_d_state=self.sample_d_state,
                sample_dt_rank=sample_dt_rank
            )
        elif self.mix_type == "og":
            self.super_mamba.set_sample_config(
                sample_embed_dim=self.sample_embed_dim, sample_expand_ratio=self.sample_expand_ratio,
                sample_kernel_size=self.sample_kernel_size, sample_d_state=self.sample_d_state,
                sample_dt_rank=sample_dt_rank, sample_mamba_ratio=self.sample_mamba_ratio,
                sample_c_kernel_size=self.sample_c_kernel_size
            )
        elif self.mix_type == "mh":
            self.super_mamba.set_sample_config(
                sample_embed_dim=self.sample_embed_dim, sample_expand_ratio=self.sample_expand_ratio,
                sample_kernel_size=self.sample_kernel_size, sample_d_state=self.sample_d_state,
                sample_dt_rank=sample_dt_rank, sample_mamba_ratio=self.sample_mamba_ratio,
                sample_c_kernel_size=self.sample_c_kernel_size, sample_num_head=self.sample_num_head
            )
        else:
            raise NotImplementedError

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        if self.is_identity_layer:
            return hidden_states

        if residual is not None:
            hidden_states = self.drop_path(hidden_states)

        hidden_states, residual = self.super_norm(
            hidden_states,
            residual=residual,
            prenorm=True,
        )
        out = self.super_mamba(hidden_states)
        return out, residual

    def calc_sampled_param_num(self):
        total_param_num = 0
        total_param_num += self.super_mamba.calc_sampled_param_num()
        total_param_num += self.super_norm.calc_sampled_param_num()
        return total_param_num

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += self.super_mamba.get_complexity(sequence_length=sequence_length)
        total_flops += self.super_norm.get_complexity(sequence_length=sequence_length)
        return total_flops


class Super_MultiMamba(nn.Module):
    def __init__(
            self,
            embed_dim,
            d_state=16,
            kernel_size=4,
            expand_ratio=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            layer_idx=None,
            directions=None,
            token_size=(14, 14),
    ):
        super().__init__()
        self.super_embed_dim = embed_dim
        self.super_d_state = d_state
        self.super_expand_ratio = expand_ratio
        self.super_d_inner = int(self.super_expand_ratio * self.super_embed_dim)
        self.super_kernel_size = kernel_size

        self.super_dt_rank = dt_rank
        self.super_dt_rank = math.ceil(
            self.super_embed_dim / 16) if self.super_dt_rank == "auto" else self.super_dt_rank

        self.layer_idx = layer_idx
        self.token_size = token_size

        self.in_proj = Super_Linear(self.super_embed_dim, self.super_d_inner * 2, bias=bias)
        self.out_proj = Super_Linear(self.super_d_inner, self.super_embed_dim, bias=bias)

        self.activation = "silu"
        self.act = nn.SiLU()

        self.super_multi_scan = Super_MultiScan(super_in_dim=self.super_d_inner, choices=directions,
                                                token_size=token_size)
        '''new for search'''
        A = repeat(
            torch.arange(1, self.super_d_state + 1),
            "n -> d n",
            d=self.super_d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        for i in range(len(self.super_multi_scan.choices)):
            setattr(self, f'A_log_{i}', nn.Parameter(A_log))
            getattr(self, f'A_log_{i}')._no_weight_decay = True

            conv1d = Super_Conv1d(
                channels=self.super_d_inner,
                kernel_size=self.super_kernel_size,
                bias=conv_bias,
            )
            setattr(self, f'conv1d_{i}', conv1d)

            x_proj = Super_Linear(
                self.super_d_inner, self.super_dt_rank + self.super_d_state * 2, bias=False
            )
            setattr(self, f'x_proj_{i}', x_proj)

            dt_proj = Super_Linear(self.super_dt_rank, self.super_d_inner, bias=True)

            # Initialize special dt projection to preserve variance at initialization
            dt_init_std = self.super_dt_rank ** -0.5 * dt_scale
            if dt_init == "constant":
                nn.init.constant_(dt_proj.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
            else:
                raise NotImplementedError

            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(self.super_d_inner) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            dt_proj.bias._no_reinit = True

            setattr(self, f'dt_proj_{i}', dt_proj)

            D = nn.Parameter(torch.ones(self.super_d_inner))  # Keep in fp32
            D._no_weight_decay = True
            setattr(self, f'D_{i}', D)

        # self.super_attn = Super_BiAttn(self.super_d_inner)

        self.sample_embed_dim = None
        self.sample_kernel_size = None
        self.sample_d_state = None
        self.sample_dt_rank = None
        self.sample_expand_ratio = None
        self.sample_d_inner = None
        for i in range(len(self.super_multi_scan.choices)):
            setattr(self, f'A_log_{i}_sample', None)
            setattr(self, f'D_{i}_sample', None)

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        xz = self.in_proj(hidden_states)

        xs = self.super_multi_scan.multi_scan(xz)  # [[BDL], [BDL], ...]
        outs = []
        for i, xz in enumerate(xs):
            # xz = rearrange(xz, "b l d -> b d l")
            A = -torch.exp(getattr(self, f'A_log_{i}_sample').float())
            conv1d = getattr(self, f'conv1d_{i}')
            x_proj = getattr(self, f'x_proj_{i}')
            dt_proj = getattr(self, f'dt_proj_{i}')
            D = getattr(self, f'D_{i}_sample')

            out = mamba_inner_fn_no_out_proj(
                xz,
                conv1d.samples['weight'],
                conv1d.samples['bias'],
                x_proj.samples['weight'],
                dt_proj.samples['weight'],
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                D,
                delta_bias=dt_proj.samples['bias'].float(),
            )
            outs.append(out)

        outs = self.super_multi_scan.multi_reverse(outs)
        # outs = [self.super_attn(rearrange(out, 'b d l -> b l d')) for out in outs]
        outs = [rearrange(out, 'b d l -> b l d') for out in outs]
        out = self.super_multi_scan(outs)
        out = self.out_proj(out)

        return out

    def set_sample_config(self, sample_embed_dim=None, sample_expand_ratio=None, sample_d_state=None,
                          sample_kernel_size=None, sample_dt_rank=None):
        self.sample_embed_dim = sample_embed_dim or self.super_embed_dim
        self.sample_d_state = sample_d_state or self.super_d_state
        self.sample_expand_ratio = sample_expand_ratio or self.super_expand_ratio
        self.sample_kernel_size = sample_kernel_size or self.super_kernel_size
        self.sample_d_inner = int(self.sample_embed_dim * self.sample_expand_ratio)
        self.sample_dt_rank = sample_dt_rank or self.super_dt_rank
        self.sample_dt_rank = math.ceil(
            self.sample_embed_dim / 16) if self.sample_dt_rank == "auto" else self.sample_dt_rank

        self.in_proj.set_sample_config(sample_in_dim=self.sample_embed_dim,
                                       sample_out_dim=self.sample_d_inner * 2)
        self.out_proj.set_sample_config(sample_in_dim=self.sample_d_inner,
                                        sample_out_dim=self.sample_embed_dim)
        for i in range(len(self.super_multi_scan.choices)):
            setattr(self, f'A_log_{i}_sample', getattr(self, f'A_log_{i}')[:self.sample_d_inner, :self.sample_d_state])
            setattr(self, f'D_{i}_sample', getattr(self, f'D_{i}')[:self.sample_d_inner])

            getattr(self, f'conv1d_{i}').set_sample_config(sample_channels=self.sample_d_inner,
                                                           sample_channels_in=self.sample_d_inner,
                                                           sample_kernel_size=self.sample_kernel_size)

            getattr(self, f'x_proj_{i}').set_sample_config(sample_in_dim=self.sample_d_inner,
                                                           sample_out_dim=self.sample_dt_rank + self.sample_d_state * 2)

            getattr(self, f'dt_proj_{i}').set_sample_config(sample_in_dim=self.sample_dt_rank,
                                                            sample_out_dim=self.sample_d_inner)

        self.super_multi_scan.set_sample_config(sample_in_dim=self.sample_d_inner)
        # self.super_attn.set_sample_config(sample_in_channels=self.sample_d_inner)

    def calc_sampled_param_num(self):
        total_param_num = 0
        total_param_num += self.in_proj.calc_sampled_param_num()
        total_param_num += self.out_proj.calc_sampled_param_num()
        for i in range(len(self.super_multi_scan.choices)):
            total_param_num += getattr(self, f'A_log_{i}_sample').numel()
            total_param_num += getattr(self, f'D_{i}_sample').numel()
            total_param_num += getattr(self, f'conv1d_{i}').calc_sampled_param_num()
            total_param_num += getattr(self, f'x_proj_{i}').calc_sampled_param_num()
            total_param_num += getattr(self, f'dt_proj_{i}').calc_sampled_param_num()
        total_param_num += self.super_attn.calc_sampled_param_num()
        return total_param_num

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += self.in_proj.get_complexity(sequence_length=sequence_length)
        total_flops += self.out_proj.get_complexity(sequence_length=sequence_length)
        for i in range(len(self.super_multi_scan.choices)):
            total_flops += sequence_length * np.prod(getattr(self, f'A_log_{i}_sample').size())
            total_flops += sequence_length * np.prod(getattr(self, f'D_{i}_sample').size())
            total_flops += getattr(self, f'conv1d_{i}').get_complexity(sequence_length=sequence_length)
            total_flops += getattr(self, f'x_proj_{i}').get_complexity(sequence_length=sequence_length)
            total_flops += getattr(self, f'dt_proj_{i}').get_complexity(sequence_length=sequence_length)
        total_flops += self.super_attn.get_complexity(sequence_length=sequence_length)
        return total_flops


class Super_MultiHeadMamba(nn.Module):
    def __init__(
            self,
            embed_dim,
            d_state=16,
            kernel_size=4,
            expand_ratio=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            layer_idx=None,
            directions=None,
            token_size=(14, 14),
            num_head=8,
    ):
        super().__init__()
        assert embed_dim % num_head == 0
        self.super_embed_dim = embed_dim
        self.super_num_head = num_head
        for i in range(self.super_num_head):
            mamba = Super_MultiMamba(
                embed_dim=self.super_embed_dim, d_state=d_state, kernel_size=kernel_size, expand_ratio=expand_ratio,
                dt_rank=dt_rank, dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale,
                dt_init_floor=dt_init_floor, conv_bias=conv_bias, bias=bias, layer_idx=layer_idx,
                directions=directions, token_size=token_size
            )
            setattr(self, f"mamba_i", mamba)

        self.sample_embed_dim = None
        self.sample_num_head = None

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        B, L, D = hidden_states.size()
        hidden_states = hidden_states.view(B, L, self.sample_num_head, self.sample_embed_dim // self.sample_num_head)
        out = None
        for i in range(self.sample_num_head):
            mamba_out = getattr(self, f"mamba_i")(hidden_states[:, :, i, :])
            if out is None:
                out = mamba_out
            else:
                out = torch.cat([out, mamba_out], dim=-1)
        return out

    def set_sample_config(self, sample_embed_dim=None, sample_num_head=None, sample_expand_ratio=None,
                          sample_d_state=None, sample_kernel_size=None, sample_dt_rank=None):
        self.sample_embed_dim = sample_embed_dim or self.super_embed_dim
        self.sample_num_head = sample_num_head or self.super_num_head
        embed_dim_every_mamba = int(self.sample_embed_dim // self.sample_num_head)
        for i in range(self.sample_num_head):
            getattr(self, f"mamba_i").set_sample_config(sample_embed_dim=embed_dim_every_mamba,
                                                        sample_expand_ratio=sample_expand_ratio,
                                                        sample_d_state=sample_d_state,
                                                        sample_kernel_size=sample_kernel_size,
                                                        sample_dt_rank=sample_dt_rank)

    def calc_sampled_param_num(self):
        total_param_num = 0
        for i in range(self.sample_num_head):
            total_param_num += getattr(self, f"mamba_i").calc_sampled_param_num()
        return total_param_num

    def get_complexity(self, sequence_length):
        total_flops = 0
        for i in range(self.sample_num_head):
            total_flops += getattr(self, f"mamba_i").get_complexity(sequence_length=sequence_length)
        return total_flops


class Super_GLamba(nn.Module):
    def __init__(
            self,
            embed_dim,
            mamba_ratio=0.5,
            num_head=8,
            mamba_type="mh",
            d_state=16,
            kernel_size=4,
            c_kernel_size=8,
            expand_ratio=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            layer_idx=None,
            directions=None,
            token_size=(14, 14),
    ):
        super().__init__()
        assert mamba_type in ['og', 'mh']
        self.super_embed_dim = embed_dim
        self.super_mamba_ratio = mamba_ratio
        self.super_num_head = num_head

        self.super_mamba_dim = int(self.super_embed_dim * self.super_mamba_ratio)
        self.super_local_dim = self.super_embed_dim - self.super_mamba_dim
        self.super_c_kernel_size = c_kernel_size

        self.mamba_type = mamba_type
        if self.mamba_type == 'og':
            self.super_mamba = Super_MultiMamba(embed_dim=self.super_mamba_dim, d_state=d_state, dt_rank=dt_rank,
                                                kernel_size=kernel_size, expand_ratio=expand_ratio,
                                                dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale,
                                                dt_init_floor=dt_init_floor, conv_bias=conv_bias, bias=bias,
                                                layer_idx=layer_idx, directions=directions, token_size=token_size)
        elif self.mamba_type == 'mh':
            self.super_mamba = Super_MultiHeadMamba(embed_dim=self.super_mamba_dim, d_state=d_state,
                                                    dt_rank=dt_rank, num_head=self.super_num_head, bias=bias,
                                                    kernel_size=kernel_size, expand_ratio=expand_ratio,
                                                    dt_min=dt_min, dt_max=dt_max, dt_init=dt_init,
                                                    dt_scale=dt_scale, token_size=token_size,
                                                    dt_init_floor=dt_init_floor, conv_bias=conv_bias,
                                                    layer_idx=layer_idx, directions=directions, )
        self.super_local_block = Super_LocalBlock(embed_dim=self.super_local_dim, kernel_size=self.super_c_kernel_size)

        self.sample_embed_dim = None
        self.sample_mamba_ratio = None
        self.sample_mamba_dim = None
        self.sample_local_dim = None
        self.sample_c_kernel_size = None
        self.sample_num_head = None

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        mamba_in = hidden_states[:, :, :self.sample_mamba_dim]
        local_in = hidden_states[:, :, -self.sample_local_dim:]
        mamba_out = self.super_mamba(mamba_in)
        local_out = self.super_local_block(local_in)

        out = torch.cat([mamba_out, local_out], dim=2)

        return out

    def set_sample_config(self, sample_embed_dim=None, sample_mamba_ratio=None, sample_expand_ratio=None,
                          sample_d_state=None, sample_kernel_size=None, sample_dt_rank=None,
                          sample_c_kernel_size=None, sample_num_head=None):
        self.sample_embed_dim = sample_embed_dim or self.super_embed_dim
        self.sample_mamba_ratio = sample_mamba_ratio or self.super_mamba_ratio
        self.sample_mamba_dim = int(self.sample_embed_dim * self.sample_mamba_ratio)
        self.sample_local_dim = self.sample_embed_dim - self.sample_mamba_dim
        self.sample_c_kernel_size = sample_c_kernel_size or self.super_c_kernel_size
        self.sample_num_head = sample_num_head or self.super_num_head
        if self.mamba_type == 'og':
            self.super_mamba.set_sample_config(sample_embed_dim=self.sample_mamba_dim,
                                               sample_expand_ratio=sample_expand_ratio,
                                               sample_d_state=sample_d_state,
                                               sample_kernel_size=sample_kernel_size,
                                               sample_dt_rank=sample_dt_rank)
        if self.mamba_type == 'mh':
            self.super_mamba.set_sample_config(sample_embed_dim=self.sample_mamba_dim,
                                               sample_num_head=self.sample_num_head,
                                               sample_expand_ratio=sample_expand_ratio,
                                               sample_d_state=sample_d_state,
                                               sample_kernel_size=sample_kernel_size,
                                               sample_dt_rank=sample_dt_rank)
        self.super_local_block.set_sample_config(sample_embed_dim=self.sample_local_dim,
                                                 sample_kernel_size=self.sample_c_kernel_size)

    def calc_sampled_param_num(self):
        total_param_num = 0
        total_param_num += self.super_multi_mamba.calc_sampled_param_num()
        total_param_num += self.super_local_block.calc_sampled_param_num()

        return total_param_num

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += self.super_multi_mamba.get_complexity(sequence_length=sequence_length)
        total_flops += self.super_local_block.get_complexity(sequence_length=sequence_length)
        return total_flops


class Super_LocalBlock(nn.Module):
    def __init__(self, embed_dim, expand_ratio=1., kernel_size=16):
        super().__init__()
        self.super_embed_dim = embed_dim
        self.expand_ratio = expand_ratio
        self.super_inner_dim = int(self.super_embed_dim * self.expand_ratio)
        self.super_kernel_size = kernel_size

        self.super_conv1 = Super_Conv1d(channels=self.super_inner_dim * 2, kernel_size=1, mode='none')
        self.super_depthwise_conv1 = Super_Conv1d(channels=self.super_inner_dim,
                                                  kernel_size=self.super_kernel_size, mode='local')
        self.super_depthwise_conv2 = Super_Conv1d(channels=self.super_inner_dim,
                                                  kernel_size=self.super_kernel_size, mode='local')
        self.super_conv2 = Super_Conv1d(channels=self.super_embed_dim, kernel_size=1, mode='none')
        self.super_bn1 = Super_BatchNorm1d(self.super_inner_dim)
        self.super_bn2 = Super_BatchNorm1d(self.super_inner_dim)

        self.glu = GLU(dim=1)
        self.swish = nn.Hardswish()

        self.sample_embed_dim = None
        self.sample_kernel_size = None
        self.sample_inner_dim = None

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        x = self.super_conv1(x)
        x = self.glu(x)
        x = self.super_depthwise_conv1(x)
        x = self.super_bn1(x)
        x = self.swish(x)

        x = self.super_depthwise_conv2(x)
        x = self.super_bn2(x)
        x = self.swish(x)

        x = self.super_conv2(x)
        out = x.transpose(1, 2).contiguous()
        return out

    def set_sample_config(self, sample_embed_dim=None, sample_kernel_size=None):
        self.sample_embed_dim = sample_embed_dim
        self.sample_kernel_size = sample_kernel_size
        self.sample_inner_dim = int(self.sample_embed_dim * self.expand_ratio)
        self.super_conv1.set_sample_config(sample_channels_in=self.sample_inner_dim,
                                           sample_channels=self.sample_inner_dim * 2, sample_kernel_size=1)
        self.super_conv2.set_sample_config(sample_channels_in=self.sample_inner_dim,
                                           sample_channels=self.sample_embed_dim, sample_kernel_size=1)
        self.super_depthwise_conv1.set_sample_config(sample_channels=self.sample_inner_dim,
                                                     sample_kernel_size=self.sample_kernel_size)
        self.super_depthwise_conv2.set_sample_config(sample_channels=self.sample_inner_dim,
                                                     sample_kernel_size=self.sample_kernel_size)
        self.super_bn1.set_sample_config(sample_num_features=self.sample_inner_dim)
        self.super_bn2.set_sample_config(sample_num_features=self.sample_inner_dim)

    def calc_sampled_param_num(self):
        total_param_num = 0
        total_param_num += self.super_conv1.calc_sampled_param_num()
        total_param_num += self.super_conv2.calc_sampled_param_num()
        total_param_num += self.super_depthwise_conv1.calc_sampled_param_num()
        total_param_num += self.super_depthwise_conv2.calc_sampled_param_num()
        total_param_num += self.super_bn1.calc_sampled_param_num()
        total_param_num += self.super_bn2.calc_sampled_param_num()
        return total_param_num

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += self.super_conv1.get_complexity(sequence_length=sequence_length)
        total_flops += self.super_conv2.get_complexity(sequence_length=sequence_length)
        total_flops += self.super_depthwise_conv1.get_complexity(sequence_length=sequence_length)
        total_flops += self.super_bn1.get_complexity(sequence_length=sequence_length)
        total_flops += self.super_depthwise_conv2.get_complexity(sequence_length=sequence_length)
        total_flops += self.super_bn2.get_complexity(sequence_length=sequence_length)
        return total_flops


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


if __name__ == '__main__':
    device = 'cuda'
    x = torch.randn((4, 196, 192)).to(device)
    model = Super_GLambaBlock(embed_dim=512, num_head=8, mix_type="mh", mamba_ratio=0.5, c_kernel_size=24).to(device)
    model.set_sample_config(is_identity_layer=False, sample_embed_dim=192, sample_num_head=8,
                            sample_mamba_ratio=0.75, sample_c_kernel_size=4)
    # model1 = Super_GLamba(embed_dim=512, num_head=8, mamba_ratio=0.5, expand_ratio=2.5, mamba_type='mh').to(device)
    # model2 = Super_MultiMamba(embed_dim=512).to(device)
    # model1.set_sample_config(sample_embed_dim=128, sample_num_head=4, sample_mamba_ratio=0.75)
    # model2.set_sample_config(sample_embed_dim=128)
    y1, _ = model(x)
    print(y1.shape)
