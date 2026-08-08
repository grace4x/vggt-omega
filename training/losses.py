"""Training losses and eval metrics for VGGT-Omega.

Follows the paper's objective (arXiv:2605.15195):

    L = w_cam * L_cam + w_depth * L_depth + w_point * L_point + w_match * L_match

* **Camera** -- Huber on the 9D pose encoding, split into translation /
  quaternion / FoV terms so they can be weighted separately. The quaternion term
  resolves the double cover (q and -q are the same rotation) before comparing.

* **Depth** and **Point map** share one shape, the uncertainty-weighted
  regression the `depth_conf` head exists for:

      || c ⊙ (1 + D^-1) ⊙ e ||  +  || c ⊙ ∇e ||  -  α Σ log c

  with `e` the residual. `(1 + D^-1)` upweights near surfaces; the `∇e` term
  supervises local structure, which needs dense GT to do anything -- it is why
  the DA3 depth maps are worth fetching. `-α log c` stops the head from
  discounting a pixel for free. `conf = 1 + exp(x)` with `proj_conf`
  zero-initialised, so training starts at conf ~= 1.05 and any down-weighting
  has to be earned.

* **Point map** is not a head: VGGT-Omega runs a single dense head and derives
  point maps by unprojecting predicted depth through the predicted camera, so
  `L_point` is what couples the depth and camera heads geometrically. The same
  `depth_conf` serves as its confidence.

`L_match` needs a tracking head, which this model does not have, so it is absent.

Self-test on synthetic data:

    python training/losses.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt_omega.utils.pose_enc import encoding_to_camera
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
# depth and point map
# --------------------------------------------------------------------------- #


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _uncertainty_loss(
    residual: torch.Tensor,
    conf: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    *,
    alpha: float,
    weight_gradient: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`|| c ⊙ w ⊙ e || + || c ⊙ ∇e || - α Σ log c`, shared by depth and points.

    `residual` is (B, S, H, W, C) -- C=1 for depth, C=3 for point maps -- so both
    the norm and the spatial difference are taken on the vector rather than on its
    magnitude. `∇` is a forward difference along W and along H; a pair contributes
    only where both endpoints are valid -- ~100% of pairs on dense GT against ~0.01%
    on a 1%-coverage sparse map. Being a masked mean, its *magnitude* does not fall
    off on sparse GT; its coverage does, which is what turns it from a structural
    constraint into a few noisy samples.

    Returns (loss, per-pixel error magnitude) so the caller can log the raw error
    separately from the confidence-weighted objective.
    """
    mask_f = mask.float()
    error = residual.norm(dim=-1)
    loss = _masked_mean(conf * weight * error, mask_f)

    if weight_gradient > 0:
        # W direction, then H. `residual` is (..., H, W, C); `conf`/`mask` (..., H, W).
        d_w = residual[..., :, 1:, :] - residual[..., :, :-1, :]
        valid_w = (mask[..., :, 1:] & mask[..., :, :-1]).float()
        d_h = residual[..., 1:, :, :] - residual[..., :-1, :, :]
        valid_h = (mask[..., 1:, :] & mask[..., :-1, :]).float()
        grad = _masked_mean(conf[..., :, 1:] * d_w.norm(dim=-1), valid_w) + _masked_mean(
            conf[..., 1:, :] * d_h.norm(dim=-1), valid_h
        )
        loss = loss + weight_gradient * grad

    # The head may discount a pixel, but pays a log penalty for doing so, so it
    # cannot drive the loss to zero for free.
    return loss - alpha * _masked_mean(torch.log(conf), mask_f), error


def depth_loss(
    pred_depth: torch.Tensor,
    pred_conf: torch.Tensor,
    gt_depth: torch.Tensor,
    mask: torch.Tensor,
    *,
    alpha: float = 0.2,
    weight_gradient: float = 1.0,
    conf_max: float = 100.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """`L_depth` from the paper. `pred_depth` is (B, S, H, W) or (B, S, H, W, 1) as
    the dense head returns it; `gt_depth` and `mask` are (B, S, H, W).

    This is a negative log-likelihood, so the value is **expected to go negative**
    once the model fits well -- the optimum for a pixel is `conf = alpha / err`,
    which drives `-alpha*log(conf)` below zero. `conf_max` caps that: without it,
    confidence grows without bound on pixels the model has memorised (an overfit
    run reaches conf ~800 in a few hundred steps) and eventually overflows.
    """
    pred, conf, gt, mask = _prepare(pred_depth, pred_conf, gt_depth, mask, conf_max, eps)
    if not mask.any():
        return _empty(pred, ("depth_err", "depth_conf", "depth_valid"))

    loss, error = _uncertainty_loss(
        (pred - gt).unsqueeze(-1),
        conf,
        mask,
        1.0 + 1.0 / gt,  # (1 + D^-1): near surfaces matter more than far ones
        alpha=alpha,
        weight_gradient=weight_gradient,
    )
    return loss, {
        "depth_err": _masked_mean(error, mask.float()).detach(),
        "depth_conf": _masked_mean(conf, mask.float()).detach(),
        "depth_valid": (mask.sum() / mask.numel()).detach(),
    }


def point_loss(
    pred_points: torch.Tensor,
    pred_conf: torch.Tensor,
    gt_points: torch.Tensor,
    gt_depth: torch.Tensor,
    mask: torch.Tensor,
    *,
    alpha: float = 0.2,
    weight_gradient: float = 1.0,
    conf_max: float = 100.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """`L_point`: the same objective on (B, S, H, W, 3) point maps in the reference
    camera's frame. `gt_depth` only supplies the `(1 + D^-1)` weighting, so the
    near/far emphasis matches `depth_loss` instead of keying off the reference
    frame's z axis, which is not a depth once the cameras move.
    """
    _, conf, gt_z, mask = _prepare(gt_depth, pred_conf, gt_depth, mask, conf_max, eps)
    mask = mask & torch.isfinite(gt_points).all(dim=-1)
    if not mask.any():
        return _empty(pred_points, ("point_err",))

    loss, error = _uncertainty_loss(
        pred_points.float() - gt_points.float(),
        conf,
        mask,
        1.0 + 1.0 / gt_z,
        alpha=alpha,
        weight_gradient=weight_gradient,
    )
    return loss, {"point_err": _masked_mean(error, mask.float()).detach()}


def unproject_depth(
    depth: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor
) -> torch.Tensor:
    """(B, S, H, W) depth -> (B, S, H, W, 3) points in the world frame.

    `extrinsics` is (B, S, 3, 4) camera-from-world. Our batches put the world
    frame at the first camera, so the result is directly comparable to the
    `point_map` the loader builds. Differentiable in depth and in both cameras,
    which is what makes `L_point` couple the two heads.
    """
    if depth.dim() == 5:
        depth = depth.squeeze(-1)
    B, S, H, W = depth.shape
    y, x = torch.meshgrid(
        torch.arange(H, device=depth.device, dtype=depth.dtype),
        torch.arange(W, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )

    fx = intrinsics[..., 0, 0][..., None, None]
    fy = intrinsics[..., 1, 1][..., None, None]
    cx = intrinsics[..., 0, 2][..., None, None]
    cy = intrinsics[..., 1, 2][..., None, None]
    cam = torch.stack([(x - cx) / fx * depth, (y - cy) / fy * depth, depth], dim=-1)

    R = extrinsics[..., :3, :3]
    t = extrinsics[..., :3, 3]
    return torch.einsum("bsij,bshwj->bshwi", R.transpose(-1, -2), cam - t[:, :, None, None, :])


def _prepare(pred_depth, pred_conf, gt_depth, mask, conf_max: float, eps: float):
    """Squeeze the head's trailing axis, clamp confidence, and drop invalid GT.

    `gt` is forced to 1 outside the mask so that the `1/gt` weighting cannot
    produce a NaN that survives multiplication by a zero mask.
    """
    if pred_depth.dim() == 5:
        pred_depth = pred_depth.squeeze(-1)
    if pred_conf.dim() == 5:
        pred_conf = pred_conf.squeeze(-1)

    mask = mask & torch.isfinite(gt_depth) & (gt_depth > eps)
    gt = torch.where(mask, gt_depth.float(), torch.ones_like(gt_depth, dtype=torch.float32))
    conf = pred_conf.float().clamp(min=eps, max=conf_max)
    return pred_depth.float(), conf, gt, mask


def _empty(reference: torch.Tensor, keys) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = reference.sum() * 0.0
    return zero, {k: zero.detach() for k in keys}


# --------------------------------------------------------------------------- #
# combined
# --------------------------------------------------------------------------- #


class VGGTOmegaLoss(nn.Module):
    """`w_cam * L_cam + w_depth * L_depth + w_point * L_point`.

    Returns (loss, scalars-for-logging). `weight_gradient` is the `||c ⊙ ∇e||`
    sub-term inside both `L_depth` and `L_point`; leave it at 1.0 with dense GT and
    set it to 0 for a sparse-only run, where no neighbouring pixel pair is ever
    both-valid and the term just adds noise.
    """

    def __init__(
        self,
        weight_camera: float = 5.0,
        weight_depth: float = 1.0,
        weight_point: float = 1.0,
        weight_gradient: float = 1.0,
        camera_kwargs: dict | None = None,
        depth_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.weight_camera = weight_camera
        self.weight_depth = weight_depth
        self.weight_point = weight_point
        self.weight_gradient = weight_gradient
        self.camera_kwargs = camera_kwargs or {}
        self.depth_kwargs = depth_kwargs or {}

    def forward(self, predictions: dict, batch: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        device = next(iter(predictions.values())).device
        total = torch.zeros((), device=device)
        logs: dict[str, torch.Tensor] = {}
        depth_kwargs = {"weight_gradient": self.weight_gradient, **self.depth_kwargs}

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
                **depth_kwargs,
            )
            total = total + self.weight_depth * loss
            logs.update(parts)
            logs["loss_depth"] = loss.detach()

        # No point head: derive the point map from predicted depth and the
        # predicted camera, which is what ties the two heads together.
        if self.weight_point > 0 and {"depth", "pose_enc"} <= predictions.keys():
            depth = predictions["depth"]
            H, W = depth.shape[2:4]
            extrinsics, intrinsics = encoding_to_camera(predictions["pose_enc"].float(), (H, W))
            pred_points = unproject_depth(depth.float(), extrinsics, intrinsics)
            loss, parts = point_loss(
                pred_points,
                predictions["depth_conf"],
                batch["point_map"],
                batch["depth"],
                batch["depth_mask"],
                **depth_kwargs,
            )
            total = total + self.weight_point * loss
            logs.update(parts)
            logs["loss_point"] = loss.detach()

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
    sparse = torch.rand(B, S, H, W) < 0.01  # DL3DV-without-DA3 sparsity
    dense = torch.ones_like(sparse)
    conf = torch.full((B, S, H, W), 1.05)
    base = -0.2 * torch.log(torch.tensor(1.05))  # the conf penalty floor at conf=1.05
    print(f"\ndepth loss   exact match          {depth_loss(gt_depth, conf, gt_depth, dense)[0].item():.4f}"
          f"   (floor from -alpha*log(conf) = {base.item():.4f})")
    print(f"depth loss   2x too large         {depth_loss(gt_depth * 2, conf, gt_depth, dense)[0].item():.4f}")
    print(f"depth loss   2x too small         {depth_loss(gt_depth / 2, conf, gt_depth, dense)[0].item():.4f}")
    print(f"depth loss   empty mask           {depth_loss(gt_depth, conf, gt_depth, torch.zeros_like(dense))[0].item():.4f}")
    high_conf = torch.full((B, S, H, W), 5.0)
    print(f"depth loss   wrong + high conf    {depth_loss(gt_depth * 2, high_conf, gt_depth, dense)[0].item():.4f}"
          f"   <- penalised for being confidently wrong")
    # What the dense GT actually buys. The grad term is a masked *mean*, so on
    # sparse GT its magnitude does not shrink -- what collapses is how many pixel
    # pairs it is averaged over, which is what makes it a structural constraint
    # rather than a handful of noisy samples.
    for name, m in (("dense ", dense), ("sparse", sparse)):
        off = depth_loss(gt_depth.flip(-1), conf, gt_depth, m, weight_gradient=0.0)[0].item()
        on = depth_loss(gt_depth.flip(-1), conf, gt_depth, m, weight_gradient=1.0)[0].item()
        pairs = (m[..., 1:] & m[..., :-1]).float().mean().item()
        print(f"depth loss   flipped, {name}       {off:.4f} -> {on:.4f}   "
              f"grad {on - off:+.4f} over {pairs * 100:6.2f}% of pixel pairs")

    # Unprojecting GT depth through the GT camera must reproduce the GT point map,
    # so L_point is at its floor exactly when depth and camera are both right.
    gt_ext, gt_int = encoding_to_camera(gt_pose, (H, W))
    gt_points = unproject_depth(gt_depth, gt_ext, gt_int)
    print(f"\npoint loss   exact match          "
          f"{point_loss(gt_points, conf, gt_points, gt_depth, dense)[0].item():.4f}")
    print(f"point loss   depth 2x too large   "
          f"{point_loss(unproject_depth(gt_depth * 2, gt_ext, gt_int), conf, gt_points, gt_depth, dense)[0].item():.4f}")
    wrong_ext, wrong_int = encoding_to_camera(gt_pose + 0.05 * torch.randn_like(gt_pose), (H, W))
    print(f"point loss   camera perturbed     "
          f"{point_loss(unproject_depth(gt_depth, wrong_ext, wrong_int), conf, gt_points, gt_depth, dense)[0].item():.4f}"
          f"   <- depth is exact; only the camera is wrong")

    print("\npose metrics  exact  ", {k: round(v, 4) for k, v in pose_metrics(gt_pose, gt_pose).items()})
    noisy = gt_pose.clone()
    noisy[..., 3:7] = F.normalize(noisy[..., 3:7] + 0.02 * torch.randn_like(noisy[..., 3:7]), dim=-1)
    print("pose metrics  noisy  ", {k: round(v, 4) for k, v in pose_metrics(noisy, gt_pose).items()})
    print("pose metrics  flipped", {k: round(v, 4) for k, v in pose_metrics(flipped, gt_pose).items()},
          "  <- must match 'exact'")
    print("depth metrics        ", {k: round(v, 4) for k, v in depth_metrics(gt_depth * 1.1, gt_depth, dense).items()})
