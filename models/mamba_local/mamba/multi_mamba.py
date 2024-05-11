import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from models.mamba_local.ops.selective_scan_interface import mamba_inner_fn_no_out_proj
from models.mamba_local.mamba.multi_scan import MultiScan
from models.mamba_local.mamba.attn import BiAttn


class MultiMamba(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,  # Fused kernel options
            layer_idx=None,
            directions=None,
            token_size=(14, 14),
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.token_size = token_size

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        self.activation = "silu"
        self.act = nn.SiLU()

        self.multi_scan = MultiScan(self.d_inner, choices=directions, token_size=token_size)
        '''new for search'''
        A = repeat(
            torch.arange(1, self.d_state + 1),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        for i in range(len(self.multi_scan.choices)):
            setattr(self, f'A_log_{i}', nn.Parameter(A_log))
            getattr(self, f'A_log_{i}')._no_weight_decay = True

            conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
            )
            setattr(self, f'conv1d_{i}', conv1d)

            x_proj = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False
            )
            setattr(self, f'x_proj_{i}', x_proj)

            dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

            # Initialize special dt projection to preserve variance at initialization
            dt_init_std = self.dt_rank ** -0.5 * dt_scale
            if dt_init == "constant":
                nn.init.constant_(dt_proj.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
            else:
                raise NotImplementedError

            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            dt_proj.bias._no_reinit = True

            setattr(self, f'dt_proj_{i}', dt_proj)

            D = nn.Parameter(torch.ones(self.d_inner))  # Keep in fp32
            D._no_weight_decay = True
            setattr(self, f'D_{i}', D)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        self.attn = BiAttn(self.d_inner)

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        xz = self.in_proj(hidden_states)

        xs = self.multi_scan.multi_scan(xz)  # [[BDL], [BDL], ...]
        outs = []
        for i, xz in enumerate(xs):
            # xz = rearrange(xz, "b l d -> b d l")
            A = -torch.exp(getattr(self, f'A_log_{i}').float())
            conv1d = getattr(self, f'conv1d_{i}')
            x_proj = getattr(self, f'x_proj_{i}')
            dt_proj = getattr(self, f'dt_proj_{i}')
            D = getattr(self, f'D_{i}')

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

        outs = self.multi_scan.multi_reverse(outs)
        outs = [self.attn(rearrange(out, 'b d l -> b l d')) for out in outs]
        out = self.multi_scan(outs)
        out = F.linear(out, self.out_proj.weight, self.out_proj.bias)

        return out


class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return pad, pad - (kernel_size + 1) % 2


class Local_Block(nn.Module):
    def __init__(self, d_model, expand_c, kernel_size):
        super().__init__()
        inner_dim = d_model * expand_c
        padding = calc_same_padding(kernel_size)
        self.net = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Conv1d(d_model, inner_dim * 2, kernel_size=1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(inner_dim),
            Swish(),
            nn.Conv1d(inner_dim, d_model, 1),
            Rearrange('b c n -> b n c')
        )

    def forward(self, x):
        return self.net(x)


class MixMamba(nn.Module):
    def __init__(
            self,
            d_model,
            d_mamba_ratio=0.5,
            c_kernel_size=8,
            d_state=16,
            d_conv=4,
            expand_m=2,
            expand_c=1,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,  # Fused kernel options
            layer_idx=None,
            directions=None,
            token_size=(14, 14),
    ):
        super().__init__()
        self.mamba_dim = int(d_model * d_mamba_ratio)
        self.local_dim = d_model - self.mamba_dim
        self.mamba = MultiMamba(d_model=self.mamba_dim, d_state=d_state, d_conv=d_conv, expand=expand_m,
                                dt_rank=dt_rank, dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale,
                                dt_init_floor=dt_init_floor, conv_bias=conv_bias, bias=bias,
                                use_fast_path=use_fast_path, layer_idx=layer_idx, directions=directions,
                                token_size=token_size)
        self.local_block = Local_Block(d_model=self.local_dim, expand_c=expand_c, kernel_size=c_kernel_size)

    def forward(self, x):
        mamba_in = x[:, :, :self.mamba_dim]
        local_in = x[:, :, -self.local_dim:]
        mamba_out = self.mamba(mamba_in)
        local_out = self.local_block(local_in)

        out = torch.cat([mamba_out, local_out], dim=2)
        return out


if __name__ == '__main__':
    device = "cuda"
    x = torch.randn((2, 256, 128)).to(device)
    # model = MultiMamba(d_model=64, token_size=(16, 16)).to(device)
    model = MixMamba(d_model=128, token_size=(16, 16)).to(device)
    out = model(x)
    print(out.shape)