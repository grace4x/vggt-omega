# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
import torch.nn as nn

from vggt_omega.models.aggregator import Aggregator
from vggt_omega.models.heads import CameraHead, DenseHead, TextAlignmentHead


class VGGTOmega(nn.Module):
    """Minimal VGGT-Omega model for camera and depth prediction.

    The defaults reproduce the released 1B checkpoint exactly. Everything below
    `enable_alignment` exists so a smaller variant can be trained from scratch --
    see `training/model_config.py` for the presets.
    """

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        enable_camera: bool = True,
        enable_depth: bool = True,
        enable_alignment: bool = False,
        depth: int = 24,
        num_heads: int = 16,
        num_register_tokens: int = 16,
        register_attention_block_indices: list[int] | None = None,
        cached_layer_indices: tuple[int, ...] = (4, 11, 17, 23),
        patch_embed_config: dict | None = None,
        camera_head_config: dict | None = None,
        dense_head_config: dict | None = None,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        self.aggregator = Aggregator(
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_register_tokens=num_register_tokens,
            cached_layer_indices=cached_layer_indices,
            patch_embed_config=patch_embed_config,
            use_checkpoint=use_checkpoint,
            **(
                {}
                if register_attention_block_indices is None
                else {"register_attention_block_indices": register_attention_block_indices}
            ),
        )
        _warn_if_rope_not_max(self.aggregator)

        self.camera_head = (
            CameraHead(dim_in=2 * embed_dim, **(camera_head_config or {})) if enable_camera else None
        )
        # The dense head reads back the layers the aggregator was told to cache,
        # so keep the two in lockstep unless the caller overrides it explicitly.
        dense_config = {"intermediate_layer_idx": list(cached_layer_indices), **(dense_head_config or {})}
        self.dense_head = (
            DenseHead(dim_in=2 * embed_dim, patch_size=patch_size, **dense_config) if enable_depth else None
        )
        self.text_alignment_head = TextAlignmentHead(dim_in=2 * embed_dim) if enable_alignment else None

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            aggregated_tokens_list, patch_token_start = self.aggregator(images)

        final_tokens = aggregated_tokens_list[-1]
        if final_tokens is None:
            raise ValueError("Aggregator did not cache the final layer, which VGGTOmega needs.")

        predictions = {
            "camera_and_register_tokens": final_tokens[:, :, :patch_token_start].contiguous(),
        }
        with torch.autocast(device_type="cuda", enabled=False):
            if self.camera_head is not None:
                predictions["pose_enc"] = self.camera_head(
                    aggregated_tokens_list,
                    patch_token_start=patch_token_start,
                )

            if self.dense_head is not None:
                depth, depth_conf = self.dense_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_token_start=patch_token_start,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.text_alignment_head is not None:
                predictions.update(
                    self.text_alignment_head(
                        aggregated_tokens_list,
                        patch_token_start=patch_token_start,
                    )
                )

        if not self.training:
            predictions["images"] = images
        return predictions


def _warn_if_rope_not_max(aggregator: nn.Module) -> None:
    for name, module in (("aggregator.patch_embed", aggregator.patch_embed), ("aggregator", aggregator)):
        rope_embed = getattr(module, "rope_embed", None)
        normalize_coords = getattr(rope_embed, "normalize_coords", None)
        if normalize_coords != "max":
            warnings.warn(
                f"{name} RoPE normalize_coords is {normalize_coords!r}; "
                "the released VGGT-Omega checkpoint was trained with 'max'.",
                stacklevel=2,
            )
