
import torch
import torch.nn as nn
from einops import rearrange

from models.mamba_local.mamba.local_scan import local_scan, local_scan_bchw, local_reverse


class MultiScan(nn.Module):
    ALL_CHOICES = ('h', 'h_flip', 'v', 'v_flip')
    # ALL_CHOICES = ('h', 'h_flip', 'v', 'v_flip', 'w2', 'w2_flip', 'w7', 'w7_flip')

    def __init__(self, dim, choices=None, token_size=(14, 14)):
        super().__init__()
        self.token_size = token_size
        if choices is None:
            self.choices = MultiScan.ALL_CHOICES
            self.norms = nn.ModuleList([nn.LayerNorm(dim, elementwise_affine=False) for _ in self.choices])
            self.weights = nn.Parameter(1e-3 * torch.randn(len(self.choices), 1, 1, 1))
            self.search = True
        else:
            self.choices = choices
            self.search = False

    def forward(self, xs):
        """
        Input @xs: [[B, L, D], ...]
        """
        if self.search:
            weights = self.weights.softmax(0)
            xs = [norm(x) for norm, x in zip(self.norms, xs)]
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
                # return LocalScanTriton.apply(x.transpose(-2, -1), K, flip, H, W)
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
                # return LocalScanTriton.apply(x, K, flip, H, W).flatten(2)
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
            # return LocalReverseTriton.apply(x, K, flip, H, W)
        else:
            raise RuntimeError(f'Direction {direction} not found.')

    def __repr__(self):
        scans = ', '.join(self.choices)
        return super().__repr__().replace(self.__class__.__name__, f'{self.__class__.__name__}[{scans}]')