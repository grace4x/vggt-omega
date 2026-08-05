# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt_omega.models.layers import SelfAttentionBlock
from vggt_omega.models.layers.utils import named_apply
from vggt_omega.models.layers.vision_transformer import init_weights_vit


class CameraHead(nn.Module):
    """Camera head used by the released VGGT-Omega checkpoints."""

    def __init__(self, dim_in: int = 2048, num_heads: int = 16, trunk_depth: int = 4) -> None:
        super().__init__()

        self.token_norm = nn.LayerNorm(dim_in, eps=1e-5)
        # Head-local transformer blocks that mix camera and register tokens across frames.
        self.trunk = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=dim_in,
                    num_heads=num_heads,
                    ffn_ratio=4.0,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    init_values=1e-5,
                    use_qk_norm=False,
                    mask_k_bias=True,
                )
                for _ in range(trunk_depth)
            ]
        )
        self.trunk_norm = nn.LayerNorm(dim_in, eps=1e-5)
        self.camera_branch = nn.Sequential(
            nn.Linear(dim_in, dim_in // 2, bias=True),
            nn.GELU(),
            nn.Linear(dim_in // 2, 9, bias=True),
        )
        self.init_weights()

    def init_weights(self) -> None:
        # Clears the NaN `bias_mask` sentinel on the trunk's masked-K-bias
        # attention; see the note in `Aggregator.init_weights`.
        named_apply(init_weights_vit, self.trunk)

        # Start the regression at a sane camera instead of at zero. Two reasons,
        # both of which only show up when training from scratch:
        #   * FoV goes through `relu(x) + 0.01`. A unit whose pre-activation
        #     starts negative receives exactly zero gradient and is dead for the
        #     whole run -- the vertical FoV in particular never recovers.
        #   * A near-zero quaternion is a degenerate rotation; `quat_to_mat`
        #     divides by its squared norm.
        # Zeroing the final weight (as `proj_conf` in DenseHead already does)
        # makes the initial prediction exactly this bias: identity pose, ~57 deg FoV.
        final = self.camera_branch[-1]
        nn.init.zeros_(final.weight)
        with torch.no_grad():
            final.bias.zero_()
            final.bias[6] = 1.0  # quaternion w, in the XYZW order quat_to_mat expects
            final.bias[7:] = 1.0  # fov_h, fov_w in radians

    def forward(
        self,
        aggregated_tokens_list: list[torch.Tensor | None],
        patch_token_start: int,
    ) -> torch.Tensor:
        tokens = aggregated_tokens_list[-1]
        if tokens is None:
            raise ValueError("Aggregator did not cache the final layer, which CameraHead needs.")
        batch_size, num_frames, num_tokens, _ = tokens.shape

        if patch_token_start is None:
            raise ValueError("patch_token_start is required for CameraHead")
        if patch_token_start > num_tokens:
            raise ValueError(f"patch_token_start ({patch_token_start}) exceeds token length ({num_tokens})")

        if tokens.dtype != torch.float32:
            tokens = tokens.float()

        camera_and_register_tokens = tokens[:, :, :patch_token_start]
        camera_and_register_tokens = self.token_norm(camera_and_register_tokens)

        camera_and_register_tokens = camera_and_register_tokens.reshape(batch_size, num_frames * patch_token_start, -1)
        rope_sincos = None
        for block in self.trunk:
            camera_and_register_tokens = block(camera_and_register_tokens, rope_sincos)

        camera_and_register_tokens = camera_and_register_tokens.reshape(batch_size, num_frames, patch_token_start, -1)
        camera_tokens = self.trunk_norm(camera_and_register_tokens[:, :, 0])
        return _apply_camera_activation(self.camera_branch(camera_tokens))


def _apply_camera_activation(raw_camera: torch.Tensor) -> torch.Tensor:
    translation = raw_camera[..., :3]
    quaternion = raw_camera[..., 3:7]
    fov = F.relu(raw_camera[..., 7:]) + 0.01
    return torch.cat([translation, quaternion, fov], dim=-1)
