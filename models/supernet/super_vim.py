import math
import torch
import torch.nn as nn
import numpy as np
from functools import partial

from thop import profile
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath

from models.supernet.super_mamba import Super_MultiMambaBlock
from models.supernet.super_ops import *


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.0)


class Super_VisionMamba(nn.Module):
    def __init__(
            self,
            img_size: int = 128,
            patch_size: int = 16,
            depth: int = 2,
            embed_dim: int = 192,
            in_chans: int = 3,
            num_classes: int = 1000,
            drop_rate: float = 0.,
            drop_path_rate: float = 0.1,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            final_pool_type: str = 'none',
            if_abs_pos_embed: bool = False,
            if_cls_token: bool = False,
            directions: list = None,
            expand_ratio: float = 2.,
            d_state: int = 16,
            kernel_size: int = 4,
    ):
        super().__init__()
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_cls_token = if_cls_token
        self.num_tokens = 1 if if_cls_token else 0
        self.patch_size = patch_size
        self.directions = directions
        self.num_classes = num_classes

        # supernet parameters
        self.super_embed_dim = embed_dim  # num_features for consistency with other models
        self.super_patch_embed = Super_PatchEmbed(img_size=img_size, patch_size=patch_size,
                                                  in_chans=in_chans, embed_dim=self.super_embed_dim)
        self.super_depth = depth
        self.super_expand_ratio = expand_ratio
        self.super_kernel_size = kernel_size
        self.super_d_state = d_state
        self.super_drop_rate = drop_rate

        self.num_patches = self.super_patch_embed.num_patches
        self.token_size = self.super_patch_embed.grid_size

        if self.if_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.super_embed_dim))

        if self.if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_tokens, self.super_embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.super_depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        self.inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.super_head = Super_Linear(self.super_embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # transformer blocks
        if directions is None:
            directions = [None] * self.super_depth
        self.layers = nn.ModuleList(
            [
                Super_MultiMambaBlock(
                    embed_dim=self.super_embed_dim,
                    d_state=self.super_d_state,
                    kernel_size=self.super_kernel_size,
                    expand_ratio=self.super_expand_ratio,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    layer_idx=i,
                    drop_path=self.inter_dpr[i],
                    directions=directions[i],
                    token_size=self.token_size,
                )
                for i in range(depth)
            ]
        )

        # output head
        self.super_norm = Super_Norm(self.super_embed_dim, eps=norm_epsilon, rms_norm=rms_norm)

        # original init
        self.apply(segm_init_weights)
        self.super_head.apply(segm_init_weights)
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=.02)

        # mamba init
        self.apply(partial(_init_weights, n_layer=depth))

        # sample
        self.sample_embed_dim = None
        self.sample_depth = None
        self.sample_d_state = None,
        self.sample_kernel_size = None
        self.sample_expand_ratio = None

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    def forward_features(self, x, train_supernet=True, sample_config=None):
        if train_supernet:
            self.set_super_config()
        else:
            assert sample_config is not None
            self.set_sample_config(config=sample_config)
        B, _, H, W = x.shape
        x = self.super_patch_embed(x)
        if self.if_cls_token:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_token, x), dim=1)

        if self.if_abs_pos_embed:
            H, W = math.ceil(H / self.patch_size), math.ceil(W / self.patch_size)
            for layer in self.layers:
                layer.mixer.multi_scan.token_size = (H, W)
            x = x + self.pos_embed
            x = self.pos_drop(x)

        residual = None
        hidden_states = x
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(hidden_states, residual)

        if residual is not None:
            hidden_states = self.drop_path(hidden_states)

        hidden_states, residual = self.super_norm(hidden_states, residual=residual, prenorm=True)

        # return only cls token if it exists
        if self.if_cls_token:
            return hidden_states[:, 0, :]

        if self.final_pool_type == 'none':
            return hidden_states[:, -1, :]
        elif self.final_pool_type == 'mean':
            return hidden_states.mean(dim=1)
        elif self.final_pool_type == 'max':
            return hidden_states.max(dim=1)
        elif self.final_pool_type == 'all':
            return hidden_states
        else:
            raise NotImplementedError

    def forward(self, x, train_supernet=True, sample_config=None, return_features=False):
        x = self.forward_features(x, train_supernet=train_supernet, sample_config=sample_config)
        if return_features:
            return x
        x = self.super_head(x)
        return x

    def set_sample_config(self, config: dict):
        self.sample_embed_dim = config['embed_dim']
        self.sample_depth = config['depth']
        self.sample_d_state = config['d_state']
        self.sample_kernel_size = config['kernel_size']
        self.sample_expand_ratio = config['expand_ratio']

        self.super_patch_embed.set_sample_config(sample_embed_dim=self.sample_embed_dim[0])
        self.super_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim[0])
        self.super_head.set_sample_config(sample_in_dim=self.sample_embed_dim[0],
                                          sample_out_dim=self.num_classes)
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx < self.sample_depths[0]:
                layer.set_sample_config(
                    is_identity_layer=False,
                    sample_embed_dim=self.sample_embed_dim[layer_idx],
                    sample_expand_ratio=self.sample_expand_ratio[layer_idx],
                    sample_d_state=self.sample_d_state[layer_idx],
                    sample_kernel_size=self.sample_kernel_size[layer_idx],
                )
            else:
                layer.set_sample_config(is_identity_layer=True)

    def set_super_config(self):
        self.super_patch_embed.set_sample_config(sample_embed_dim=self.super_embed_dim)
        self.super_norm.set_sample_config(sample_embed_dim=self.super_embed_dim)
        self.super_head.set_sample_config(sample_in_dim=self.super_embed_dim,
                                          sample_out_dim=self.num_classes)
        for layer_idx, layer in enumerate(self.layers):
            layer.set_sample_config(
                is_identity_layer=False,
                sample_embed_dim=self.super_embed_dim,
                sample_expand_ratio=self.super_expand_ratio,
                sample_d_state=self.super_d_state,
                sample_kernel_size=self.super_kernel_size,
            )

    def get_sampled_params_numel(self, config: dict = None):
        if config is None:
            self.set_super_config()
        else:
            self.set_sample_config(config)
        total_numels = 0
        total_numels += self.super_patch_embed.calc_sampled_param_num()
        if self.if_cls_token:
            total_numels += self.cls_token.numel()
        if self.if_abs_pos_embed:
            total_numels += self.pos_embed.numel()
        total_numels += self.super_head.calc_sampled_param_num()
        total_numels += self.super_norm.calc_sampled_param_num()
        for layer_idx, layer in enumerate(self.layers):
            total_numels += layer.calc_sampled_param_num()
        return total_numels

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += self.super_norm.get_complexity(sequence_length=sequence_length)

        total_flops += self.super_patch_embed.get_complexity(sequence_length=sequence_length)
        total_flops += self.super_head.get_complexity(sequence_length=sequence_length + 1)
        if self.if_abs_pos_embed:
            total_flops += np.prod(self.pos_embed[..., :self.sample_embed_dim[0]].size()) / 2.0
        for layer in self.layers:
            total_flops += layer.get_complexity(sequence_length=sequence_length + 1)
        return total_flops


