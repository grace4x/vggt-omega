"""Training losses and eval metrics for VGGT-Omega.

Two losses, matching the two heads the model actually has:

* **Camera** -- Huber on the 9D pose encoding, split into translation /
  quaternion / FoV terms so they can be weighted separately. The quaternion term
  resolves the double cover (q and -q are the same rotation) before comparing.

* **Depth** -- the uncertainty-weighted regression the `depth_conf` head exists
  for: `conf * err - alpha * log(conf)`. The model predicts `conf = 1 + exp(x)`
  with `proj_conf` zero-initialised, so training starts at conf ~= 1.05 and the
  network has to earn any down-weighting. Error is measured in log-depth, which
  keeps near and far surfaces on comparable footing, and is Huberised because
  COLMAP's sparse points carry occasional gross outliers.

`gradient_matching_loss` is included for when dense depth is available (MVS or a
monocular teacher); it is a no-op on DL3DV's ~1%-coverage sparse maps and is off
by default.

Self-test on synthetic data:

    python training/losses.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt_omega.utils.rotation import quat_to_mat


# --------------------------------------------------------------------------- #
# camera
# --------------------------------------------------------------------------- #


def camera_loss(
    pred_pose_enc: torch.Tensor,
    gt_pose_enc: torch.Tensor,
    *,
    weight_translation: float = 1.0,
    weight_rotation: float = 1.0,
    weight_fov: float = 0.5,
    huber_delta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Huber loss on the (T, quat, fov) encoding. Both tensors are (B, S, 9)."""
    if pred_pose_enc.shape != gt_pose_enc.shape:
        raise ValueError(f"shape mismatch: {pred_pose_enc.shape} vs {gt_pose_enc.shape}")

    pred = pred_pose_enc.float()
    gt = gt_pose_enc.float()

    translation = F.huber_loss(pred[..., :3], gt[..., :3], delta=huber_delta, reduction="mean")

    # Quaternion double cover: q and -q are the same rotation, so align the sign
    # of the target to the prediction before measuring the difference. Without
    # this, half the targets pull the head in the wrong direction.
    pred_quat, gt_quat = pred[..., 3:7], gt[..., 3:7]
    sign = torch.where((pred_quat * gt_quat).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    rotation = F.huber_loss(pred_quat, gt_quat * sign, delta=huber_delta, reduction="mean")

    fov = F.huber_loss(pred[..., 7:], gt[..., 7:], delta=huber_delta, reduction="mean")

    total = weight_translation * translation + weight_rotation * rotation + weight_fov * fov
    return total, {"cam_trans": translation.detach(), "cam_rot": rotation.detach(), "cam_fov": fov.detach()}


# --------------------------------------------------------------------------- #
# depth
# --------------------------------------------------------------------------- #


def depth_loss(
    pred_depth: torch.Tensor,
    pred_conf: torch.Tensor,
    gt_depth: torch.Tensor,
    mask: torch.Tensor,
    *,
    alpha: float = 0.2,
    huber_delta: float = 0.5,
    log_space: bool = True,
    conf_max: float = 100.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Confidence-weighted depth regression over the valid pixels only.

    `pred_depth` is (B, S, H, W) or (B, S, H, W, 1) as the dense head returns it;
    `gt_depth` and `mask` are (B, S, H, W).

    This is a negative log-likelihood, so the value is **expected to go negative**
    once the model fits well -- the optimum for a pixel is `conf = alpha / err`,
    which drives `-alpha*log(conf)` below zero. `conf_max` caps that: without it,
    confidence grows without bound on pixels the model has memorised (an overfit
    run reaches conf ~800 in a few hundred steps) and eventually overflows.
    """
    if pred_depth.dim() == 5:
        pred_depth = pred_depth.squeeze(-1)
    if pred_conf.dim() == 5:
        pred_conf = pred_conf.squeeze(-1)

    mask = mask & torch.isfinite(gt_depth) & (gt_depth > eps)
    num_valid = mask.sum()
    if num_valid == 0:
        zero = pred_depth.sum() * 0.0
        return zero, {"depth_err": zero.detach(), "depth_conf": zero.detach(), "depth_valid": zero.detach()}

    pred = pred_depth.float()[mask].clamp_min(eps)
    gt = gt_depth.float()[mask].clamp_min(eps)
    conf = pred_conf.float()[mask].clamp(max=conf_max)

    residual = torch.log(pred) - torch.log(gt) if log_space else pred - gt
    error = F.huber_loss(residual, torch.zeros_like(residual), delta=huber_delta, reduction="none")

    # conf * err - alpha * log(conf): the head can discount a pixel, but pays
    # a log penalty for doing so, so it cannot drive the loss to zero for free.
    loss = (conf * error - alpha * torch.log(conf)).mean()

    return loss, {
        "depth_err": error.mean().detach(),
        "depth_conf": conf.mean().detach(),
        "depth_valid": (num_valid / mask.numel()).detach(),
    }


def gradient_matching_loss(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    mask: torch.Tensor,
    *,
    num_scales: int = 4,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Multi-scale gradient matching on log depth. Only meaningful for dense GT --
    a pixel contributes only where both it and its neighbour are valid, which on
    sparse COLMAP depth is almost never."""
    if pred_depth.dim() == 5:
        pred_depth = pred_depth.squeeze(-1)

    B, S, H, W = pred_depth.shape
    pred = torch.log(pred_depth.float().clamp_min(eps)).reshape(B * S, 1, H, W)
    gt = torch.log(gt_depth.float().clamp_min(eps)).reshape(B * S, 1, H, W)
    valid = mask.reshape(B * S, 1, H, W).float()

    total = pred.sum() * 0.0
    scales_used = 0
    for scale in range(num_scales):
        step = 2**scale
        if H <= step or W <= step:
            break
        scales_used += 1
        for slice_a, slice_b in (
            ((..., slice(None), slice(step, None)), (..., slice(None), slice(None, -step))),
            ((..., slice(step, None), slice(None)), (..., slice(None, -step), slice(None))),
        ):
            # A pair contributes only when both of its endpoints are observed.
            pair_valid = valid[slice_a] * valid[slice_b]
            d_pred = pred[slice_a] - pred[slice_b]
            d_gt = gt[slice_a] - gt[slice_b]
            residual = (d_pred - d_gt).abs() * pair_valid
            total = total + residual.sum() / pair_valid.sum().clamp_min(1.0)

    return total / max(scales_used, 1)


# --------------------------------------------------------------------------- #
# combined
# --------------------------------------------------------------------------- #


class VGGTOmegaLoss(nn.Module):
    """Sums the camera and depth terms. Returns (loss, scalars-for-logging)."""

    def __init__(
        self,
        weight_camera: float = 5.0,
        weight_depth: float = 1.0,
        weight_gradient: float = 0.0,
        camera_kwargs: dict | None = None,
        depth_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.weight_camera = weight_camera
        self.weight_depth = weight_depth
        self.weight_gradient = weight_gradient
        self.camera_kwargs = camera_kwargs or {}
        self.depth_kwargs = depth_kwargs or {}

    def forward(self, predictions: dict, batch: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        device = next(iter(predictions.values())).device
        total = torch.zeros((), device=device)
        logs: dict[str, torch.Tensor] = {}

        if "pose_enc" in predictions and self.weight_camera > 0:
            loss, parts = camera_loss(predictions["pose_enc"], batch["pose_enc"], **self.camera_kwargs)
            total = total + self.weight_camera * loss
            logs.update(parts)
            logs["loss_camera"] = loss.detach()

        if "depth" in predictions and self.weight_depth > 0:
            loss, parts = depth_loss(
                predictions["depth"],
                predictions["depth_conf"],
                batch["depth"],
                batch["depth_mask"],
                **self.depth_kwargs,
            )
            total = total + self.weight_depth * loss
            logs.update(parts)
            logs["loss_depth"] = loss.detach()

            if self.weight_gradient > 0:
                grad = gradient_matching_loss(predictions["depth"], batch["depth"], batch["depth_mask"])
                total = total + self.weight_gradient * grad
                logs["loss_grad"] = grad.detach()

        logs["loss"] = total.detach()
        return total, logs


# --------------------------------------------------------------------------- #
# eval metrics
# --------------------------------------------------------------------------- #


@torch.no_grad()
def pose_metrics(pred_pose_enc: torch.Tensor, gt_pose_enc: torch.Tensor) -> dict[str, float]:
    """Relative-pose errors over every frame pair, plus the AUC@30 that the VGGT
    line of work reports. Inputs are (B, S, 9)."""
    pred, gt = pred_pose_enc.float(), gt_pose_enc.float()
    B, S, _ = pred.shape

    # `quat_to_mat` divides by the squared norm, so guard against an early-training
    # prediction that has collapsed towards zero.
    R_pred = quat_to_mat(F.normalize(pred[..., 3:7], dim=-1, eps=1e-8))  # (B, S, 3, 3)
    R_gt = quat_to_mat(F.normalize(gt[..., 3:7], dim=-1, eps=1e-8))
    t_pred, t_gt = pred[..., :3], gt[..., :3]

    i, j = torch.triu_indices(S, S, offset=1)
    # Relative rotation i->j: R_j @ R_i^T for camera-from-world matrices.
    rel_pred = R_pred[:, j] @ R_pred[:, i].transpose(-1, -2)
    rel_gt = R_gt[:, j] @ R_gt[:, i].transpose(-1, -2)
    cos = ((rel_pred.transpose(-1, -2) @ rel_gt).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    # Clamp to the exact endpoints: nudging inwards puts a ~0.03 deg floor under
    # every reported error, which is enough to mask a genuinely perfect prediction.
    rotation_deg = torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0)))

    # Translation direction, which is what is comparable under scale ambiguity.
    tp = t_pred[:, j] - (rel_pred @ t_pred[:, i].unsqueeze(-1)).squeeze(-1)
    tg = t_gt[:, j] - (rel_gt @ t_gt[:, i].unsqueeze(-1)).squeeze(-1)
    cos_t = F.cosine_similarity(tp, tg, dim=-1).clamp(-1.0, 1.0)
    translation_deg = torch.rad2deg(torch.acos(cos_t))

    worst = torch.maximum(rotation_deg, translation_deg)
    thresholds = torch.arange(1, 31, device=worst.device, dtype=worst.dtype)
    accuracy = (worst.reshape(-1, 1) < thresholds.reshape(1, -1)).float().mean(0)

    return {
        "rot_err_deg_mean": rotation_deg.mean().item(),
        "rot_err_deg_median": rotation_deg.median().item(),
        "trans_err_deg_mean": translation_deg.mean().item(),
        "trans_err_deg_median": translation_deg.median().item(),
        "trans_err_abs_mean": (t_pred - t_gt).norm(dim=-1).mean().item(),
        "rra_at_5": (rotation_deg < 5).float().mean().item(),
        "rta_at_5": (translation_deg < 5).float().mean().item(),
        "auc_at_30": accuracy.mean().item(),
        "fov_err_deg": torch.rad2deg((pred[..., 7:] - gt[..., 7:]).abs()).mean().item(),
    }


@torch.no_grad()
def depth_metrics(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    mask: torch.Tensor,
    *,
    align_median: bool = False,
    eps: float = 1e-6,
) -> dict[str, float]:
    if pred_depth.dim() == 5:
        pred_depth = pred_depth.squeeze(-1)
    mask = mask & torch.isfinite(gt_depth) & (gt_depth > eps)
    if mask.sum() == 0:
        return {"abs_rel": float("nan"), "delta_1.25": float("nan"), "scale_ratio": float("nan")}

    pred = pred_depth.float()[mask].clamp_min(eps)
    gt = gt_depth.float()[mask].clamp_min(eps)
    ratio = (gt.median() / pred.median()).item()
    if align_median:
        pred = pred * ratio

    return {
        "abs_rel": ((pred - gt).abs() / gt).mean().item(),
        "delta_1.25": (torch.maximum(pred / gt, gt / pred) < 1.25).float().mean().item(),
        "scale_ratio": ratio,
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    B, S, H, W = 2, 6, 32, 48

    gt_pose = torch.randn(B, S, 9) * 0.1
    gt_pose[..., 3:7] = F.normalize(gt_pose[..., 3:7] + torch.tensor([0.0, 0.0, 0.0, 1.0]), dim=-1)
    gt_pose[..., 7:] = 1.0

    # A rotation is unchanged by negating its quaternion, so the loss must be
    # blind to that flip -- this is the term that silently ruins training if wrong.
    flipped = torch.cat([gt_pose[..., :3], -gt_pose[..., 3:7], gt_pose[..., 7:]], dim=-1)
    print(f"camera loss  exact match          {camera_loss(gt_pose, gt_pose)[0].item():.3e}")
    print(f"camera loss  quaternion negated   {camera_loss(flipped, gt_pose)[0].item():.3e}   <- must also be ~0")
    print(f"camera loss  noisy                {camera_loss(gt_pose + 0.1 * torch.randn_like(gt_pose), gt_pose)[0].item():.3e}")

    gt_depth = torch.rand(B, S, H, W) * 4 + 0.5
    mask = torch.rand(B, S, H, W) < 0.01  # DL3DV-like sparsity
    conf = torch.full((B, S, H, W), 1.05)
    base = -0.2 * torch.log(torch.tensor(1.05))  # the conf penalty floor at conf=1.05
    print(f"\ndepth loss   exact match          {depth_loss(gt_depth, conf, gt_depth, mask)[0].item():.4f}"
          f"   (floor from -alpha*log(conf) = {base.item():.4f})")
    print(f"depth loss   2x too large         {depth_loss(gt_depth * 2, conf, gt_depth, mask)[0].item():.4f}")
    print(f"depth loss   2x too small         {depth_loss(gt_depth / 2, conf, gt_depth, mask)[0].item():.4f}"
          f"   <- log-space, so symmetric with the line above")
    print(f"depth loss   empty mask           {depth_loss(gt_depth, conf, gt_depth, torch.zeros_like(mask))[0].item():.4f}")
    high_conf = torch.full((B, S, H, W), 5.0)
    print(f"depth loss   wrong + high conf    {depth_loss(gt_depth * 2, high_conf, gt_depth, mask)[0].item():.4f}"
          f"   <- penalised for being confidently wrong")

    dense = torch.ones_like(mask)
    print(f"\ngradient loss  exact match        {gradient_matching_loss(gt_depth, gt_depth, dense).item():.4f}")
    print(f"gradient loss  uniform 1.1x scale {gradient_matching_loss(gt_depth * 1.1, gt_depth, dense).item():.4f}"
          f"   <- 0 by design: it only sees structure, not scale")
    print(f"gradient loss  structure wrong    "
          f"{gradient_matching_loss(gt_depth.flip(-1), gt_depth, dense).item():.4f}")
    print(f"gradient loss  sparse mask        "
          f"{gradient_matching_loss(gt_depth.flip(-1), gt_depth, mask).item():.4f}"
          f"   <- near-useless at 1% coverage, hence weight_gradient=0")

    print("\npose metrics  exact  ", {k: round(v, 4) for k, v in pose_metrics(gt_pose, gt_pose).items()})
    noisy = gt_pose.clone()
    noisy[..., 3:7] = F.normalize(noisy[..., 3:7] + 0.02 * torch.randn_like(noisy[..., 3:7]), dim=-1)
    print("pose metrics  noisy  ", {k: round(v, 4) for k, v in pose_metrics(noisy, gt_pose).items()})
    print("pose metrics  flipped", {k: round(v, 4) for k, v in pose_metrics(flipped, gt_pose).items()},
          "  <- must match 'exact'")
    print("depth metrics        ", {k: round(v, 4) for k, v in depth_metrics(gt_depth * 1.1, gt_depth, mask).items()})
