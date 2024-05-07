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


class Super_MultiMambaBlock(nn.Module):
    def __init__(
            self,
            embed_dim,
            d_state=16,
            kernel_size=4,
            expand_ratio=2,
            conv_bias=True,
            bias=False,
            layer_idx=None,
            directions=None,
            token_size=(14, 14),
            norm_epsilon=1e-5,
            drop_path=0.,
            rms_norm=False,
            scale=False
    ):
        super().__init__()
        self.super_embed_dim = embed_dim
        self.super_expand_ratio = expand_ratio
        self.super_ffn_embed_dim_this_layer = int(expand_ratio * embed_dim)
        self.super_dropout = drop_path
        self.super_d_state = d_state
        self.super_kernel_size = kernel_size

        self.is_identity_layer = None
        self.scale = scale

        self.sample_embed_dim = None
        self.sample_expand_ratio = None
        self.sample_ffn_embed_dim_this_layer = None
        self.sample_scale = None
        self.sample_dropout = None
        self.sample_d_state = None
        self.sample_kernel_size = None

        self.super_norm = Super_Norm(self.super_embed_dim, eps=norm_epsilon, rms_norm=rms_norm)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

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

    def set_sample_config(self, is_identity_layer, sample_embed_dim=None, sample_expand_ratio=None,
                          sample_d_state=None, sample_kernel_size=None, sample_dt_rank="auto"):
        if is_identity_layer:
            self.is_identity_layer = True
            return
        self.is_identity_layer = False
        self.sample_embed_dim = sample_embed_dim
        self.sample_expand_ratio = sample_expand_ratio
        self.sample_kernel_size = sample_kernel_size
        self.sample_d_state = sample_d_state
        self.super_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)
        self.super_mamba.set_sample_config(
            sample_embed_dim=self.sample_embed_dim, sample_expand_ratio=self.sample_expand_ratio,
            sample_kernel_size=self.sample_kernel_size, sample_d_state=self.sample_d_state,
            sample_dt_rank=sample_dt_rank
        )

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
            expand_ratio=2,
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
        self.super_expand = expand_ratio
        self.super_d_inner = int(self.super_expand * self.super_embed_dim)
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

        self.super_attn = Super_BiAttn(self.super_d_inner)

        self.sample_embed_dim = None
        self.sample_kernel_size = None
        self.sample_d_state = None
        self.sample_dt_rank = None
        self.sample_expand_ratio = None
        self.sample_d_inner = None
        for i in range(len(self.super_multi_scan.choices)):
            setattr(self, f'A_log_{i}_sample', None)
            setattr(self, f'conv1d_{i}_sample', None)
            setattr(self, f'x_proj_{i}_sample', None)
            setattr(self, f'dt_proj_{i}_sample', None)
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
            conv1d = getattr(self, f'conv1d_{i}_sample')
            x_proj = getattr(self, f'x_proj_{i}_sample')
            dt_proj = getattr(self, f'dt_proj_{i}_sample')
            D = getattr(self, f'D_{i}_sample')

            out = mamba_inner_fn_no_out_proj(
                xz,
                conv1d.weight,
                conv1d.bias,
                x_proj.weight,
                dt_proj.weight,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                D,
                delta_bias=dt_proj.bias.float(),
            )
            outs.append(out)

        outs = self.super_multi_scan.multi_reverse(outs)
        outs = [self.super_attn(rearrange(out, 'b d l -> b l d')) for out in outs]
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
                                                           sample_kernel_size=self.sample_kernel_size)
            setattr(self, f'conv1d_{i}_sample', getattr(self, f'conv1d_{i}'))

            getattr(self, f'x_proj_{i}').set_sample_config(sample_in_dim=self.sample_d_inner,
                                                           sample_out_dim=self.sample_dt_rank + self.sample_d_state * 2)
            setattr(self, f'x_proj_{i}_sample', getattr(self, f'x_proj_{i}'))

            getattr(self, f'dt_proj_{i}').set_sample_config(sample_in_dim=self.sample_dt_rank,
                                                            sample_out_dim=self.sample_d_inner)
            setattr(self, f'dt_proj_{i}_sample', getattr(self, f'dt_proj_{i}'))

        self.super_multi_scan.set_sample_config(sample_in_dim=self.sample_d_inner)
        self.super_attn.set_sample_config(sample_in_channels=self.sample_d_inner)

    def calc_sampled_param_num(self):
        total_param_num = 0
        total_param_num += self.in_proj.calc_sampled_param_num()
        total_param_num += self.out_proj.calc_sampled_param_num()
        for i in range(len(self.super_multi_scan.choices)):
            total_param_num += getattr(self, f'A_log_{i}_sample').numel()
            total_param_num += getattr(self, f'D_{i}_sample').numel()
            total_param_num += getattr(self, f'conv1d_{i}_sample').calc_sampled_param_num()
            total_param_num += getattr(self, f'x_proj_{i}_sample').calc_sampled_param_num()
            total_param_num += getattr(self, f'dt_proj_{i}_sample').calc_sampled_param_num()
        total_param_num += self.super_attn.calc_sampled_param_num()
        return total_param_num

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += self.in_proj.get_complexity(sequence_length=sequence_length)
        total_flops += self.out_proj.get_complexity(sequence_length=sequence_length)
        for i in range(len(self.super_multi_scan.choices)):
            total_flops += sequence_length * np.prod(getattr(self, f'A_log_{i}_sample').size())
            total_flops += sequence_length * np.prod(getattr(self, f'D_{i}_sample').size())
            total_flops += getattr(self, f'conv1d_{i}_sample').get_complexity(sequence_length=sequence_length)
            total_flops += getattr(self, f'x_proj_{i}_sample').get_complexity(sequence_length=sequence_length)
            total_flops += getattr(self, f'dt_proj_{i}_sample').get_complexity(sequence_length=sequence_length)
        total_flops += self.super_attn.get_complexity(sequence_length=sequence_length)
        return total_flops
