"""Named VGGT-Omega size presets, plus DINOv3 backbone loading.

`large` reproduces the released 1B checkpoint bit-for-bit -- `build_model("large")`
and a bare `VGGTOmega()` are the same module. `small` and `base` pair the
aggregator with the DINOv3 ViT-S/16 and ViT-B/16 trunks, which is what makes them
trainable on a single 24 GB GPU.

    python training/model_config.py --list
    python training/model_config.py --preset small --benchmark
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import torch
import torch.nn as nn

from vggt_omega.models import VGGTOmega


@dataclass
class ModelConfig:
    """Everything that differs between size presets."""

    name: str
    embed_dim: int
    depth: int
    num_heads: int
    patch_size: int = 16

    # Which aggregator layers get cached for the heads. Must be `depth`-relative
    # and length 4 -- DenseHead consumes exactly four scales.
    cached_layer_indices: tuple[int, ...] = (4, 11, 17, 23)
    # Blocks whose inter-frame attention is restricted to camera+register tokens.
    register_attention_block_indices: list[int] = field(default_factory=lambda: [2, 6, 9, 14, 20])
    num_register_tokens: int = 16

    # DINOv3 trunk. `depth`/`num_heads` here are the trunk's, not the aggregator's,
    # though every preset happens to keep them equal.
    patch_embed_depth: int = 24
    patch_embed_num_heads: int = 16

    camera_head_num_heads: int = 16
    camera_head_trunk_depth: int = 4

    dense_head_features: int = 256
    dense_head_out_channels: list[int] = field(default_factory=lambda: [256, 512, 1024, 1024])

    # HF repo whose weights initialise the trunk (see convert_dinov3.py).
    dinov3_model_id: str | None = None

    def to_kwargs(self, use_checkpoint: bool = False, **overrides) -> dict:
        kwargs = dict(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            num_register_tokens=self.num_register_tokens,
            register_attention_block_indices=list(self.register_attention_block_indices),
            cached_layer_indices=tuple(self.cached_layer_indices),
            patch_embed_config={
                "depth": self.patch_embed_depth,
                "num_heads": self.patch_embed_num_heads,
                "use_checkpoint": use_checkpoint,
            },
            camera_head_config={
                "num_heads": self.camera_head_num_heads,
                "trunk_depth": self.camera_head_trunk_depth,
            },
            dense_head_config={
                "features": self.dense_head_features,
                "out_channels": list(self.dense_head_out_channels),
            },
            use_checkpoint=use_checkpoint,
        )
        kwargs.update(overrides)
        return kwargs


PRESETS: dict[str, ModelConfig] = {
    # DINOv3 ViT-S/16 trunk. ~100M params, trains at 24 frames in <8 GB.
    "small": ModelConfig(
        name="small",
        embed_dim=384,
        depth=12,
        num_heads=6,
        cached_layer_indices=(2, 5, 8, 11),
        register_attention_block_indices=[1, 3, 5, 8, 10],
        patch_embed_depth=12,
        patch_embed_num_heads=6,
        camera_head_num_heads=6,
        dense_head_features=128,
        dense_head_out_channels=[128, 256, 512, 512],
        dinov3_model_id="facebook/dinov3-vits16-pretrain-lvd1689m",
    ),
    # DINOv3 ViT-B/16 trunk.
    "base": ModelConfig(
        name="base",
        embed_dim=768,
        depth=12,
        num_heads=12,
        cached_layer_indices=(2, 5, 8, 11),
        register_attention_block_indices=[1, 3, 5, 8, 10],
        patch_embed_depth=12,
        patch_embed_num_heads=12,
        camera_head_num_heads=12,
        dense_head_features=256,
        dense_head_out_channels=[256, 512, 1024, 1024],
        dinov3_model_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
    ),
    # The released VGGT-Omega-1B config. Identical to a bare `VGGTOmega()`.
    "large": ModelConfig(
        name="large",
        embed_dim=1024,
        depth=24,
        num_heads=16,
        cached_layer_indices=(4, 11, 17, 23),
        register_attention_block_indices=[2, 6, 9, 14, 20],
        patch_embed_depth=24,
        patch_embed_num_heads=16,
        camera_head_num_heads=16,
        dense_head_features=256,
        dense_head_out_channels=[256, 512, 1024, 1024],
        dinov3_model_id="facebook/dinov3-vitl16-pretrain-lvd1689m",
    ),
}


def get_config(preset: str) -> ModelConfig:
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    return replace(PRESETS[preset])


def build_model(
    preset: str | ModelConfig = "small",
    *,
    use_checkpoint: bool = False,
    dinov3_checkpoint: str | Path | None = None,
    enable_depth: bool = True,
    enable_camera: bool = True,
    **overrides,
) -> VGGTOmega:
    config = get_config(preset) if isinstance(preset, str) else preset
    model = VGGTOmega(
        enable_camera=enable_camera,
        enable_depth=enable_depth,
        **config.to_kwargs(use_checkpoint=use_checkpoint, **overrides),
    )
    if dinov3_checkpoint is not None:
        load_dinov3_backbone(model, dinov3_checkpoint)
    return model


def load_dinov3_backbone(model: VGGTOmega, checkpoint: str | Path) -> None:
    """Load a `convert_dinov3.py` checkpoint into `model.aggregator.patch_embed`."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload["state_dict"] if "state_dict" in payload else payload
    trunk = model.aggregator.patch_embed

    expected = trunk.state_dict()
    mismatched = [
        k for k, v in state_dict.items() if k in expected and tuple(v.shape) != tuple(expected[k].shape)
    ]
    if mismatched:
        raise ValueError(
            f"DINOv3 checkpoint does not match this preset (shape mismatch on {mismatched[:3]}...). "
            f"Checkpoint came from {payload.get('source')}; the trunk here is "
            f"embed_dim={trunk.embed_dim}, depth={trunk.n_blocks}, num_heads={trunk.num_heads}."
        )

    missing, unexpected = trunk.load_state_dict(state_dict, strict=False)
    # `rope_embed.periods` and the qkv `bias_mask` buffers are non-persistent and
    # already set by `init_weights()`; anything else missing is a real problem.
    unresolved = [k for k in missing if not (k.endswith("bias_mask") or k == "rope_embed.periods")]
    if unresolved or unexpected:
        raise ValueError(f"unexpected DINOv3 key mismatch: missing={unresolved}, unexpected={unexpected}")
    print(f"loaded DINOv3 trunk from {checkpoint} ({payload.get('source', 'unknown source')})")


def parameter_summary(model: nn.Module) -> dict[str, float]:
    def count(module: nn.Module | None) -> float:
        return 0.0 if module is None else sum(p.numel() for p in module.parameters()) / 1e6

    trunk = count(model.aggregator.patch_embed)
    return {
        "patch_embed": trunk,
        "aggregator_blocks": count(model.aggregator) - trunk,
        "camera_head": count(getattr(model, "camera_head", None)),
        "dense_head": count(getattr(model, "dense_head", None)),
        "total": count(model),
    }


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="small")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--benchmark", action="store_true", help="time a fwd+bwd+step on cuda")
    parser.add_argument("--frames", type=int, nargs="*", default=[8, 16, 24])
    parser.add_argument("--hw", type=int, nargs=2, default=[288, 512])
    parser.add_argument("--checkpointing", action="store_true")
    args = parser.parse_args()

    presets = sorted(PRESETS) if args.list else [args.preset]
    for name in presets:
        model = build_model(name, use_checkpoint=args.checkpointing)
        summary = parameter_summary(model)
        print(
            f"{name:6s} "
            + "  ".join(f"{k}={v:.1f}M" for k, v in summary.items())
            + f"  (aggregator depth={model.aggregator.depth}, cached={sorted(model.aggregator.cached_layer_indices)})"
        )

        if not args.benchmark:
            del model
            continue

        model = model.cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        H, W = args.hw
        for S in args.frames:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            images = torch.rand(1, S, 3, H, W, device="cuda")
            try:
                start = time.time()
                predictions = model(images)
                loss = predictions["pose_enc"].abs().mean() + predictions["depth"].abs().mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                print(
                    f"       S={S:3d} {H}x{W}  {time.time() - start:5.2f}s  "
                    f"peak {torch.cuda.max_memory_allocated() / 2**30:5.2f} GiB"
                )
            except torch.cuda.OutOfMemoryError:
                print(f"       S={S:3d} {H}x{W}  OOM")
                optimizer.zero_grad(set_to_none=True)
            del images
        del model, optimizer
        torch.cuda.empty_cache()
