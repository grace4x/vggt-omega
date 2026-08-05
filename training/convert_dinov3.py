#!/usr/bin/env python3
"""Convert a HuggingFace DINOv3 ViT checkpoint into this repo's `DinoVisionTransformer`.

`facebook/dinov3-*-pretrain-lvd1689m` ships the `transformers` layout, which
names things differently from the reference DINOv3 code that VGGT-Omega's
`vggt_omega/models/layers/vision_transformer.py` follows, and keeps q/k/v as
three separate projections instead of one fused `qkv`.

The HF config sets `key_bias: false`; this repo achieves the same thing with
`LinearKMaskedBias`, which multiplies the K third of the fused bias by zero. So
the K bias slot is filled with zeros here and the mask makes it a no-op either way.

Usage:

    python training/convert_dinov3.py --model-id facebook/dinov3-vits16-pretrain-lvd1689m \
        --out checkpoints/dinov3_vits16.pt

    # then, in your model builder:
    patch_embed.load_state_dict(torch.load("checkpoints/dinov3_vits16.pt")["state_dict"], strict=False)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def convert(hf_state: dict[str, torch.Tensor], num_layers: int) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}

    direct = {
        "embeddings.cls_token": "cls_token",
        "embeddings.mask_token": "mask_token",
        "embeddings.register_tokens": "storage_tokens",
        "embeddings.patch_embeddings.weight": "patch_embed.proj.weight",
        "embeddings.patch_embeddings.bias": "patch_embed.proj.bias",
        "norm.weight": "norm.weight",
        "norm.bias": "norm.bias",
    }
    for src, dst in direct.items():
        if src in hf_state:
            out[dst] = hf_state[src].clone()

    # HF stores mask_token as (1, 1, D); this repo declares it as (1, D).
    if "mask_token" in out and out["mask_token"].dim() == 3:
        out["mask_token"] = out["mask_token"][0]

    for i in range(num_layers):
        s, d = f"layer.{i}.", f"blocks.{i}."
        out[d + "norm1.weight"] = hf_state[s + "norm1.weight"].clone()
        out[d + "norm1.bias"] = hf_state[s + "norm1.bias"].clone()
        out[d + "norm2.weight"] = hf_state[s + "norm2.weight"].clone()
        out[d + "norm2.bias"] = hf_state[s + "norm2.bias"].clone()

        q = hf_state[s + "attention.q_proj.weight"]
        k = hf_state[s + "attention.k_proj.weight"]
        v = hf_state[s + "attention.v_proj.weight"]
        out[d + "attn.qkv.weight"] = torch.cat([q, k, v], dim=0)

        qb = hf_state.get(s + "attention.q_proj.bias")
        vb = hf_state.get(s + "attention.v_proj.bias")
        if qb is not None:
            kb = hf_state.get(s + "attention.k_proj.bias", torch.zeros_like(qb))
            out[d + "attn.qkv.bias"] = torch.cat([qb, kb, vb], dim=0)

        out[d + "attn.proj.weight"] = hf_state[s + "attention.o_proj.weight"].clone()
        out[d + "attn.proj.bias"] = hf_state[s + "attention.o_proj.bias"].clone()

        out[d + "ls1.gamma"] = hf_state[s + "layer_scale1.lambda1"].clone()
        out[d + "ls2.gamma"] = hf_state[s + "layer_scale2.lambda1"].clone()

        # HF calls the MLP projections up/down; this repo calls them fc1/fc2.
        out[d + "mlp.fc1.weight"] = hf_state[s + "mlp.up_proj.weight"].clone()
        out[d + "mlp.fc1.bias"] = hf_state[s + "mlp.up_proj.bias"].clone()
        out[d + "mlp.fc2.weight"] = hf_state[s + "mlp.down_proj.weight"].clone()
        out[d + "mlp.fc2.bias"] = hf_state[s + "mlp.down_proj.bias"].clone()

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--local-dir", type=Path, default=None, help="use an already-downloaded repo instead")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from safetensors.torch import load_file

    if args.local_dir:
        weights_path, config_path = args.local_dir / "model.safetensors", args.local_dir / "config.json"
    else:
        from huggingface_hub import hf_hub_download

        weights_path = Path(hf_hub_download(args.model_id, "model.safetensors"))
        config_path = Path(hf_hub_download(args.model_id, "config.json"))

    config = json.loads(config_path.read_text())
    hf_state = load_file(str(weights_path))
    state_dict = convert(hf_state, config["num_hidden_layers"])

    # The kwargs `DinoVisionTransformer` needs to match this checkpoint.
    vit_kwargs = {
        "patch_size": config["patch_size"],
        "embed_dim": config["hidden_size"],
        "depth": config["num_hidden_layers"],
        "num_heads": config["num_attention_heads"],
        "ffn_ratio": config["intermediate_size"] / config["hidden_size"],
        "n_storage_tokens": config["num_register_tokens"],
        "layerscale_init": config["layerscale_value"],
        "pos_embed_rope_base": config["rope_theta"],
        "qkv_bias": config.get("query_bias", True),
        "proj_bias": config.get("proj_bias", True),
        "ffn_bias": config.get("mlp_bias", True),
        "mask_k_bias": not config.get("key_bias", False),
        # VGGT-Omega overrides these two relative to stock DINOv3.
        "pos_embed_rope_normalize_coords": "max",
        "pos_embed_rope_dtype": "fp32",
        "norm_layer": "layernormbf16",
        "ffn_layer": "mlp",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict, "vit_kwargs": vit_kwargs, "source": args.model_id}, args.out)

    total = sum(t.numel() for t in state_dict.values())
    print(f"wrote {args.out}: {len(state_dict)} tensors, {total / 1e6:.1f}M params")
    print(f"vit_kwargs = {json.dumps(vit_kwargs, indent=1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
