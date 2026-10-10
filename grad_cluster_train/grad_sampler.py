"""Gradient-cluster sampler mirroring G-DIG / LESS / TAGCOS.

Pipeline (per-epoch cluster recomputation)
------------------------------------------

1. **JL projection** (mirroring G-DIG §3.2 and LESS §2):
   For each training scene run one forward + backward pass (no weight update).
   Clip the gradient to the training `--clip-grad` L2 cap, then project the
   full parameter-gradient vector g ∈ R^P into R^d using a fixed Gaussian
   random matrix R ~ N(0, 1) / sqrt(d), d=400 by default.
   R is generated chunk-by-chunk on the GPU so it is never materialised in full
   (peak extra memory: chunk_size × d × 4 bytes ≈ 50 MB at the defaults).
   The same seed → same projection matrix → comparable coordinates across scenes.

2. **K-means**, selected by `metric`:
   * `euclidean` (default, matching G-DIG): plain Euclidean k-means on the
     unnormalised projected vectors.  No spherical normalisation, no PCA,
     no CountSketch.  G-DIG uses 512 clusters.
   * `cosine`: L2-normalise each projected vector, then spherical k-means
     (assignment by cosine similarity; each centre is the renormalised mean
     of its members).  Because the JL map is linear, this is the same
     partition as normalising the full gradient, projecting, and clustering
     by cosine — magnitude is ignored and only direction is clustered.
     Scene-token embeddings are clustered with the same metric (they are
     already `embed_dim`, so they are not projected again).

   Cluster assignments are saved for inspection.  By default training keeps
   drawing scenes uniformly from the full dataset.

3. **Optional token-complexity weights** (arXiv:2401.04578, Eq. 1-2): each
   token cluster j gets C_j = d_inter,j * d_intra,j in cosine distance and a
   weight softmax(C / tau)_j.  The sampler then draws cluster j with that
   probability and a member uniformly within it.

References
----------
- G-DIG (gradient diversity for MT data selection): projects to 400 dims,
  512 clusters, Euclidean k-means.
- LESS (Xia & Chen): JL-style gradient sketches for influence estimation.
- TAGCOS: random-projected gradient features → k-means → per-cluster coreset.

Usage::

    sampler = GradClusterSampler(n_total=N, epoch_size=N, rank=0, world_size=1)
    loader  = DataLoader(dataset, sampler=sampler, ...)

    for epoch in ...:
        sampler.set_epoch(epoch)
        for batch in loader: ...           # normal training

        if epoch % grad_cluster_every == 0:
            recompute_clusters(
                model=model, criterion=criterion,
                grad_dataset=grad_dataset, device=device,
                k=512, proj_dim=400, proj_seed=42,
                proj_workers=4, rank=rank, world_size=world_size,
                epoch=epoch, out_dir=out_dir, metric="euclidean",
            )
"""

from __future__ import annotations

import math
import time
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler


# --------------------------------------------------------------------------- #
# Gaussian JL random projection of model gradients
# --------------------------------------------------------------------------- #


@torch.no_grad()
def jl_project_grad(
    model: torch.nn.Module,
    proj_dim: int,
    proj_seed: int,
    *,
    chunk_size: int = 32768,
    device: torch.device | None = None,
) -> np.ndarray:
    """Project all model gradients to `proj_dim` dims via a Gaussian random matrix.

    Mirrors the dimensionality-reduction step in G-DIG and LESS: a fixed random
    Gaussian matrix R ∈ R^{P × d} (P = total parameters, d = proj_dim) maps the
    gradient g ∈ R^P to g_proj = R^T g / sqrt(d) ∈ R^d.

    The full matrix R is never materialised in memory — it is re-generated
    deterministically in parameter-order chunks from `proj_seed`.  The same seed
    always produces the same projection matrix, so every scene's output lives in
    the same coordinate space.

    Args:
        model:      Model with populated `.grad` tensors (after .backward()).
                    Must NOT be DDP-wrapped (use the raw inner model).
        proj_dim:   Output dimension (d).  G-DIG uses 400.
        proj_seed:  Fixed RNG seed for R.  Same seed = same projection matrix
                    across all scenes and all recomputation epochs.
        chunk_size: Parameter elements handled per inner iteration.  Controls
                    peak extra GPU memory: chunk_size × proj_dim × 4 bytes.
        device:     Computation device.  Defaults to the model's device.

    Returns:
        (proj_dim,) float32 numpy array — the projected gradient.
    """
    if device is None:
        try:
            device = next(p.device for p in model.parameters() if p.grad is not None)
        except StopIteration:
            return np.zeros(proj_dim, dtype=np.float32)

    # Re-seed each call so every scene uses the same projection matrix.
    gen = torch.Generator(device=device)
    gen.manual_seed(proj_seed)

    proj = torch.zeros(proj_dim, device=device, dtype=torch.float32)

    for _, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().to(device=device, dtype=torch.float32).view(-1)
        n = g.numel()
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            # R_chunk: (c, proj_dim) — the rows of R for elements [start:end].
            R_chunk = torch.randn(end - start, proj_dim,
                                  device=device, dtype=torch.float32,
                                  generator=gen)
            # proj += R_chunk^T @ g_chunk  (contribution from this slice)
            proj.add_(torch.mv(R_chunk.t(), g[start:end]))

    # Scale so E[||proj||²] ≈ ||g||² (standard JL normalisation).
    proj.mul_(proj_dim ** -0.5)
    return proj.cpu().numpy()


# --------------------------------------------------------------------------- #
# Per-scene gradient collection
# --------------------------------------------------------------------------- #


def compute_gradient_projections(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    dataset,
    device: torch.device,
    *,
    proj_dim: int = 400,
    proj_seed: int = 42,
    chunk_size: int = 32768,
    workers: int = 4,
    clip_grad: float = 50.0,
    verbose: bool = True,
    capture_scene_tokens: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, list[int]]:
    """One forward+backward per scene; returns JL-projected gradient matrix and
    (optionally) mean scene-token features from the final aggregator layer.

    Args:
        model:     Raw (non-DDP) model in train mode.  No optimizer step is
                   taken; gradients are zeroed before returning.
        criterion: Same loss used for training.
        dataset:   Full training set (augment=False recommended for stability).
        device:    GPU/CPU.
        proj_dim:  Projection dimension.  G-DIG default: 400.
        proj_seed: Seed for the fixed Gaussian projection matrix.
        chunk_size: Chunk size for the JL projection loop.
        workers:   DataLoader worker count.
        clip_grad: Global L2 clip applied after backward, before projection.
                   Same semantics as training `--clip-grad`.  0 disables.
        verbose:   Print progress every 100 scenes.
        capture_scene_tokens: If True (default), also capture mean-pooled
                   register tokens (the 16 learned scene-summary tokens that
                   sit between the camera token and the patch grid) from the
                   last cached aggregator layer for each scene.  These are
                   extracted via a forward hook during the same pass so no
                   extra cost is incurred.

    Returns:
        G:              (N, proj_dim) float32 — projected gradients, one row
                        per valid scene.  **Not normalised** (plain Euclidean
                        k-means).  Rows come from clipped grads when
                        `clip_grad > 0`.
        T:              (N, embed_dim) float32 — per-scene register-token
                        feature vectors from the last cached aggregator layer.
                        Each cached layer tensor has shape
                        (batch, F, num_tokens, 2*embed_dim): the codebase
                        stores cat([frame_tokens, tokens], dim=-1), where
                        frame_tokens = post-intra-frame-attention (pre-global
                        mixing) and tokens = post-inter-frame-attention (fully
                        fused).  Only the second half (tokens, post-fusion) is
                        used here.  The 16 register tokens per frame
                        (indices 1:patch_token_start) are averaged across both
                        frames and token positions → (embed_dim,) per scene.
                        None if `capture_scene_tokens=False` or if the model
                        has no `.aggregator` attribute.
        valid_indices:  Dataset indices for which loss was finite.
    """
    from training.mixed_dataset import collate_mixed

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_mixed,
        drop_last=False,
    )

    model.train()
    model.zero_grad(set_to_none=True)

    projections: list[np.ndarray] = []
    token_feats: list[np.ndarray] = []
    valid_indices: list[int] = []
    n_skip = 0
    n_clipped = 0
    t0 = time.time()

    # ---- scene-token capture via forward hook ----------------------------
    # We use a single-element list as a mutable cell so the hook closure can
    # write into it and we can read it after each forward pass.  The hook is
    # registered on model.aggregator (if it exists) and fires synchronously
    # during the forward pass, so the captured tensor is always up-to-date
    # before we check loss finiteness.
    _tok_buf: list[np.ndarray | None] = [None]
    hook_handle = None

    if capture_scene_tokens and hasattr(model, "aggregator"):
        def _aggregator_hook(
            module: torch.nn.Module,
            input: tuple,
            output: tuple,
        ) -> None:
            agg_list, patch_start = output
            # Find the last non-None cached layer tensor.
            last_tokens: torch.Tensor | None = None
            for t in reversed(agg_list):
                if t is not None:
                    last_tokens = t
                    break
            if last_tokens is None:
                _tok_buf[0] = None
                return
            # last_tokens: (batch=1, num_frames, num_tokens, 2*embed_dim)
            # The cache stores cat([frame_tokens, tokens], dim=-1) where:
            #   first half  [..., :D] = post-intra-frame-attention (pre-global mixing)
            #   second half [..., D:] = post-inter-frame-attention (fully fused)
            # We want only the second half — the fully cross-view-fused state.
            D = last_tokens.shape[-1] // 2  # embed_dim
            # Token layout: [camera(1)] [register×16] [patch grid...]
            # Register tokens are indices [1:patch_start].
            reg_toks = last_tokens[0, :, 1:patch_start, D:]  # (F, 16, embed_dim)
            # Mean across both frames and the 16 register tokens → (embed_dim,)
            mean_vec = reg_toks.detach().float().mean(dim=(0, 1)).cpu().numpy()
            _tok_buf[0] = mean_vec

        hook_handle = model.aggregator.register_forward_hook(_aggregator_hook)
    # ----------------------------------------------------------------------

    for i, batch in enumerate(loader):
        tensors = {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        model.zero_grad(set_to_none=True)
        _tok_buf[0] = None  # reset before each forward

        preds = model(tensors["images"])
        loss, _ = criterion(preds, tensors)

        if not math.isfinite(float(loss.detach())):
            n_skip += 1
            del preds, loss, tensors
            continue

        # Commit the token feature for this scene (hook has already fired).
        if _tok_buf[0] is not None:
            token_feats.append(_tok_buf[0])

        loss.backward()

        if clip_grad and clip_grad > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip_grad
            )
            if float(total_norm) > clip_grad:
                n_clipped += 1

        gproj = jl_project_grad(model, proj_dim, proj_seed,
                                 chunk_size=chunk_size, device=device)
        projections.append(gproj)
        valid_indices.append(i)

        del preds, loss, tensors

        if verbose and ((i + 1) % 100 == 0 or i == len(loader) - 1):
            rate = (i + 1) / max(time.time() - t0, 1e-9)
            print(
                f"  [grad-proj] {i + 1}/{len(dataset)}  {rate:.2f} it/s"
                + (f"  skipped {n_skip}" if n_skip else "")
                + (f"  clipped {n_clipped}" if n_clipped else ""),
                flush=True,
            )

    if hook_handle is not None:
        hook_handle.remove()

    model.zero_grad(set_to_none=True)

    if not projections:
        raise RuntimeError(
            "compute_gradient_projections: every scene had non-finite loss"
        )

    G = np.stack(projections, axis=0).astype(np.float32)

    T: np.ndarray | None = None
    if token_feats:
        if len(token_feats) == len(projections):
            T = np.stack(token_feats, axis=0).astype(np.float32)
        else:
            # Shouldn't happen, but warn rather than crash.
            print(
                f"  [scene-tokens] WARNING: captured {len(token_feats)} token "
                f"features but {len(projections)} gradient projections — "
                "skipping token clustering.",
                flush=True,
            )

    return G, T, valid_indices


# --------------------------------------------------------------------------- #
# Intra-cluster Euclidean distance stats (diagnostic only)
# --------------------------------------------------------------------------- #


def _stats_from_per_cluster(per_cluster: list[float]) -> dict:
    arr = np.array(per_cluster)
    return {
        "per_cluster": per_cluster,
        "mean":   float(arr.mean()),
        "median": float(np.median(arr)),
        "min":    float(arr.min()),
        "max":    float(arr.max()),
    }


def cluster_euclidean_stats(G: np.ndarray, labels: np.ndarray) -> dict:
    """Per-cluster mean Euclidean distance to the cluster centroid.

    For cluster C with centroid μ this is mean_{x in C} ||x − μ|| — the
    (unsquared) quantity Euclidean k-means is grouping by.  Vectors are
    **not** L2-normalised, so magnitude in the projected space counts.
    Singleton clusters report 0.
    """
    k = int(labels.max()) + 1
    per_cluster: list[float] = []
    for ci in range(k):
        vecs = G[labels == ci]
        m = len(vecs)
        if m <= 1:
            per_cluster.append(0.0)
            continue
        centroid = vecs.mean(axis=0)
        dists = np.linalg.norm(vecs - centroid, axis=1)
        per_cluster.append(float(dists.mean()))
    return _stats_from_per_cluster(per_cluster)


def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation.  Near-zero rows stay zero."""
    X = np.asarray(X, dtype=np.float64)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(nrm, 1e-12)


def cluster_cosine_stats(G: np.ndarray, labels: np.ndarray) -> dict:
    """Per-cluster mean cosine distance (1 − cos) to the unit centroid.

    Rows are L2-normalised first.  The centroid is the mean of the unit
    vectors, then re-normalised, matching spherical k-means.  Singleton
    clusters report 0.
    """
    X = l2_normalize_rows(G)
    k = int(labels.max()) + 1
    per_cluster: list[float] = []
    for ci in range(k):
        vecs = X[labels == ci]
        m = len(vecs)
        if m <= 1:
            per_cluster.append(0.0)
            continue
        centroid = vecs.mean(axis=0)
        cn = float(np.linalg.norm(centroid))
        if cn <= 1e-12:
            per_cluster.append(0.0)
            continue
        cos = vecs @ (centroid / cn)
        per_cluster.append(float((1.0 - cos).mean()))
    return _stats_from_per_cluster(per_cluster)


# --------------------------------------------------------------------------- #
# K-means clustering
# --------------------------------------------------------------------------- #


def run_spherical_kmeans(
    G: np.ndarray,
    k: int,
    *,
    seed: int = 0,
    n_init: int = 10,
    max_iter: int = 300,
) -> np.ndarray:
    """Spherical k-means: L2-normalise rows, assign by cosine similarity.

    Same update as `clustering/cluster.py`: each centre is the mean of its
    assigned unit vectors, re-normalised.  k-means++ init on the sphere,
    `n_init` restarts, best run kept by total cosine to the assigned centre.
    Empty clusters are re-seeded at the worst-assigned point.
    """
    X = l2_normalize_rows(G)
    n, d = X.shape
    k = min(int(k), n)
    rng = np.random.default_rng(seed)

    best_obj = -np.inf
    best_labels = np.zeros(n, dtype=np.int32)
    for _ in range(n_init):
        centers = np.empty((k, d), dtype=X.dtype)
        centers[0] = X[rng.integers(n)]
        closest = np.full(n, np.inf)
        for c in range(1, k):
            # ||x - c||^2 = 2(1 - cos) on the unit sphere
            d2 = np.maximum(0.0, 2.0 - 2.0 * (X @ centers[c - 1]))
            closest = np.minimum(closest, d2)
            s = float(closest.sum())
            centers[c] = X[rng.integers(n)] if s <= 0 else X[rng.choice(n, p=closest / s)]
        centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)

        labels = np.full(n, -1)
        sim_c = None
        for _it in range(max_iter):
            sim_c = X @ centers.T
            new_labels = sim_c.argmax(1)
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels
            new_centers = np.empty_like(centers)
            for c in range(k):
                m = labels == c
                if m.any():
                    mu = X[m].mean(0)
                    mu_n = np.linalg.norm(mu)
                    new_centers[c] = mu / mu_n if mu_n > 1e-12 else centers[c]
                else:
                    new_centers[c] = X[sim_c.max(1).argmin()]
            shift = 1.0 - float((new_centers * centers).sum(1).mean())
            centers = new_centers
            if shift < 1e-6:
                break

        obj = float((X * centers[labels]).sum())
        if obj > best_obj:
            best_obj = obj
            best_labels = labels.astype(np.int32, copy=True)

    return best_labels


def run_kmeans(
    G: np.ndarray,
    k: int,
    *,
    seed: int = 0,
    n_init: int = 10,
    metric: str = "euclidean",
) -> tuple[np.ndarray, dict]:
    """K-means on (N, D) vectors.

    `metric="euclidean"` matches G-DIG: no normalisation, sklearn KMeans
    with Euclidean (L2) distance.

    `metric="cosine"` L2-normalises each row and runs spherical k-means, so
    assignment is by cosine similarity.

    Args:
        G:      (N, D) float matrix (projected gradients or token embeddings).
        k:      Number of clusters.  G-DIG uses 512.
        seed:   Random seed.
        n_init: Number of initialisations.
        metric: `"euclidean"` or `"cosine"`.

    Returns:
        labels: (N,) int32 cluster assignments in [0, k).
        stats:  Intra-cluster distance dict.  Euclidean: mean L2 distance to
                the centroid.  Cosine: mean cosine distance (1 − cos) to the
                unit centroid.  Diagnostic only.
    """
    if metric not in ("euclidean", "cosine"):
        raise ValueError(f"metric must be 'euclidean' or 'cosine', got {metric!r}")

    n = len(G)
    if k >= n:
        labels = np.arange(n, dtype=np.int32)
        stats_fn = cluster_cosine_stats if metric == "cosine" else cluster_euclidean_stats
        return labels, stats_fn(G, labels)

    if metric == "cosine":
        labels = run_spherical_kmeans(G, k, seed=seed, n_init=n_init)
        return labels, cluster_cosine_stats(G, labels)

    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError("scikit-learn is required: pip install scikit-learn")

    km = KMeans(n_clusters=k, random_state=seed, n_init=n_init, max_iter=300)
    labels = km.fit_predict(G).astype(np.int32)
    return labels, cluster_euclidean_stats(G, labels)


# --------------------------------------------------------------------------- #
# Cluster complexity weights (arXiv:2401.04578, Eq. 1-2)
# --------------------------------------------------------------------------- #


def cluster_complexity(
    X: np.ndarray, labels: np.ndarray, neighbors: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cluster (ids, d_inter, d_intra), both mean cosine distances in [0, 2].

    Same geometry as `multi_clustering/cluster.py::cluster_geometry`: rows are
    L2-normalised, a cluster centroid is the renormalised mean of its members,
    d_intra is the mean cosine distance from members to their own centroid, and
    d_inter is the mean cosine distance to the `neighbors` nearest other centroids.
    Cosine is used whatever metric the k-means ran with, as in the paper.
    """
    X = np.asarray(X, dtype=np.float64)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    ids = np.unique(labels)
    centers = np.stack([X[labels == c].mean(0) for c in ids])
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
    d_intra = np.array([float(np.mean(1.0 - X[labels == c] @ centers[j]))
                        for j, c in enumerate(ids)])
    l = min(int(neighbors), len(ids) - 1)
    if l < 1:
        return ids, np.ones(len(ids)), d_intra
    cc = centers @ centers.T
    np.fill_diagonal(cc, -np.inf)
    nearest = np.sort(cc, axis=1)[:, -l:]
    return ids, 1.0 - nearest.mean(axis=1), d_intra


def complexity_weights(d_inter: np.ndarray, d_intra: np.ndarray, tau: float) -> np.ndarray:
    """Eq. 1-2: softmax over cluster complexity C_j = d_inter,j * d_intra,j."""
    z = (np.asarray(d_inter) * np.asarray(d_intra)) / tau
    e = np.exp(z - z.max())
    return e / e.sum()


def cluster_balance_weights(
    labels: np.ndarray, groups: np.ndarray, alpha: float,
) -> np.ndarray:
    """Per-scene draw probabilities that flatten cluster sizes inside each group.

    A scene in a cluster of n scenes gets weight n**(alpha - 1), so a cluster's
    total mass is proportional to n**alpha: alpha=1 leaves the pool unchanged,
    alpha=0 gives every cluster the same mass, so a small cluster's scenes are
    drawn more often.  Each group (dataset) is then rescaled to its own share of
    the pool, so only the mix *inside* a dataset moves and the dataset
    proportions stay as they are in the pool.  Returns weights summing to 1.
    """
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    w = counts[inverse].astype(np.float64) ** (float(alpha) - 1.0)
    for g in np.unique(groups):
        m = groups == g
        w[m] *= m.mean() / w[m].sum()
    return w / w.sum()


# --------------------------------------------------------------------------- #
# Epoch sampler
# --------------------------------------------------------------------------- #


class GradClusterSampler(Sampler):
    """Draws `epoch_size` indices per epoch.

    By default a random shuffle of every index.  After `set_scene_weights`,
    indices are drawn with replacement from those per-index probabilities
    (e.g. cluster weight / cluster size from the token-complexity softmax).
    Multi-GPU: the full `epoch_size` sequence is generated identically on every
    rank, then sliced by rank so each GPU sees a complementary subset.
    """

    def __init__(
        self,
        n_total: int,
        epoch_size: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
    ) -> None:
        self.n_total = n_total
        self.epoch_size = epoch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self._epoch: int = 0
        self.scene_weights: np.ndarray | None = None

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def set_scene_weights(self, weights: np.ndarray | None) -> None:
        """Per-index draw probabilities for the following epochs; None = shuffle."""
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)
            if weights.shape != (self.n_total,):
                raise ValueError(f"expected {self.n_total} weights, got shape {weights.shape}")
            if (weights < 0).any() or weights.sum() <= 0:
                raise ValueError("scene weights must be non-negative with a positive sum")
            weights = weights / weights.sum()
        self.scene_weights = weights

    def __len__(self) -> int:
        return self.epoch_size // self.world_size

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self._epoch * 7919)
        if self.scene_weights is None:
            epoch_indices = np.resize(rng.permutation(self.n_total), self.epoch_size)
        else:
            epoch_indices = rng.choice(self.n_total, self.epoch_size, p=self.scene_weights)
        per_rank = self.epoch_size // self.world_size
        start = self.rank * per_rank
        yield from epoch_indices[start: start + per_rank].tolist()


# --------------------------------------------------------------------------- #
# High-level recomputation helper
# --------------------------------------------------------------------------- #


def recompute_clusters(
    *,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    grad_dataset,
    device: torch.device,
    k: int,
    proj_dim: int,
    proj_seed: int,
    proj_workers: int,
    rank: int,
    world_size: int,
    epoch: int,
    out_dir,
    seed: int = 0,
    clip_grad: float = 50.0,
    metric: str = "euclidean",
    cosine_center: str = "tokens",
    complexity_tau: float | None = None,
    complexity_neighbors: int = 20,
    cluster_balance: float | None = None,
    scene_groups: np.ndarray | None = None,
) -> np.ndarray | None:
    """Compute JL-projected gradients, cluster them, and save assignments.

    Computation runs on rank 0 only.  Other ranks wait at the barrier so the
    training loop stays in lockstep.

    With `complexity_tau` set, the token clusters are also scored by
    complexity (arXiv:2401.04578, Eq. 1-2) and rank 0 returns per-scene draw
    probabilities over `grad_dataset`: cluster weight / cluster size, so a
    cluster's whole share is spread evenly over its members.  Otherwise, and on
    every other rank, returns None.

    Args:
        k:          Number of k-means clusters (G-DIG default: 512).
        proj_dim:   JL projection output dimension (G-DIG default: 400).
        proj_seed:  Fixed seed for the projection matrix R.
        out_dir:    If set, saves `grad_clusters/epoch_NNNN.npz` there.
        clip_grad:  Global L2 clip applied to each scene gradient before
                    projection (same as training `--clip-grad`).  0 disables.
        metric:     `"euclidean"` (default, G-DIG) or `"cosine"`.  Cosine
                    L2-normalises each projected gradient and each scene-token
                    embedding, then runs spherical k-means.
        complexity_tau:       Softmax temperature over token-cluster
                              complexity; None disables the weights.
        complexity_neighbors: Nearest centroids averaged into d_inter (the
                              paper's l).
        cluster_balance:      Exponent alpha for `cluster_balance_weights`; None
                              disables.  Needs `scene_groups`.  Cannot be
                              combined with `complexity_tau`.
        scene_groups:         (len(grad_dataset),) int source-dataset id of each
                              scene; the balance is applied inside each group.
        cosine_center: `"tokens"` (default), `"both"` or `"none"`: which
                       vectors have their mean over scenes subtracted before
                       anything cosine (spherical k-means, complexity).  The
                       mean register token is ~half of every token vector, so
                       uncentered cosines all crowd together.  Euclidean
                       k-means is unaffected.  Raw G and T are what is saved.
    """
    if cosine_center not in ("tokens", "both", "none"):
        raise ValueError(f"cosine_center must be 'tokens', 'both' or 'none', got {cosine_center!r}")
    import torch.distributed as dist

    if metric not in ("euclidean", "cosine"):
        raise ValueError(f"metric must be 'euclidean' or 'cosine', got {metric!r}")
    if cluster_balance is not None:
        if complexity_tau is not None:
            raise ValueError("cluster_balance and complexity_tau are mutually exclusive")
        if scene_groups is None:
            raise ValueError("cluster_balance needs scene_groups")

    scene_weights: np.ndarray | None = None
    if rank == 0 and grad_dataset is not None:
        print(f"\n[epoch {epoch}] --- gradient projection + clustering ---", flush=True)
        t0 = time.time()

        G, T, valid_indices = compute_gradient_projections(
            model, criterion, grad_dataset, device,
            proj_dim=proj_dim,
            proj_seed=proj_seed,
            workers=proj_workers,
            clip_grad=clip_grad,
            verbose=True,
            capture_scene_tokens=True,
        )
        dist_name = "L2 dist" if metric == "euclidean" else "cosine dist (1-cos)"
        dist_tag = "l2" if metric == "euclidean" else "cos"
        cluster_name = "k-means" if metric == "euclidean" else "spherical k-means (cosine)"

        n = len(valid_indices)
        k_actual = max(2, math.isqrt(n)) if k == 0 else min(k, n)
        print(
            f"  [grad-proj] {n} scenes → ({n}, {proj_dim})  k={k_actual}"
            f"  metric={metric}  running {cluster_name}...",
            flush=True,
        )

        G_fit = G - G.mean(0) if cosine_center == "both" else G
        T_cos = None if T is None else (T - T.mean(0) if cosine_center != "none" else T)
        labels, cluster_stats = run_kmeans(
            G_fit, k_actual, seed=seed + epoch, metric=metric,
        )
        elapsed = time.time() - t0

        print(
            f"  [grad-cluster] intra-cluster {dist_name}:"
            f"  mean={cluster_stats['mean']:.3f}"
            f"  median={cluster_stats['median']:.3f}"
            f"  min={cluster_stats['min']:.3f}"
            f"  max={cluster_stats['max']:.3f}"
            f"  ({elapsed:.0f}s)",
            flush=True,
        )
        sizes = [(labels == ci).sum() for ci in range(int(labels.max()) + 1)]
        per_lines = "  ".join(
            f"c{ci}(n={sizes[ci]},{dist_tag}={v:.3f})"
            for ci, v in enumerate(cluster_stats["per_cluster"])
        )
        print(f"  [grad-cluster] per-cluster: {per_lines}", flush=True)

        # ---- scene-token clustering (comparison) ----
        token_labels: np.ndarray | None = None
        token_cluster_stats: dict | None = None
        ari_score: float | None = None

        if T is not None:
            print(
                f"  [scene-tokens] {n} scenes → ({n}, {T.shape[1]})  "
                f"(16 register tokens averaged over frames → embed_dim)  "
                f"k={k_actual}  metric={metric}  running {cluster_name}"
                " on mean register tokens...",
                flush=True,
            )
            t1 = time.time()
            token_labels, token_cluster_stats = run_kmeans(
                T_cos if metric == "cosine" else T, k_actual, seed=seed + epoch, metric=metric,
            )
            print(
                f"  [scene-tokens] intra-cluster {dist_name}:"
                f"  mean={token_cluster_stats['mean']:.3f}"
                f"  median={token_cluster_stats['median']:.3f}"
                f"  min={token_cluster_stats['min']:.3f}"
                f"  max={token_cluster_stats['max']:.3f}"
                f"  ({time.time() - t1:.0f}s)",
                flush=True,
            )

            # Adjusted Rand Index between gradient clusters and token clusters.
            try:
                from sklearn.metrics import adjusted_rand_score
                ari_score = float(adjusted_rand_score(labels, token_labels))
                print(
                    f"  [scene-tokens] ARI(grad_clusters, token_clusters) = {ari_score:.4f}"
                    "  (1=identical, 0=random)",
                    flush=True,
                )
            except Exception as exc:
                print(f"  [scene-tokens] could not compute ARI: {exc}", flush=True)

        # ---- token-cluster complexity weights ----
        cw: dict | None = None
        if complexity_tau is not None:
            if token_labels is None:
                raise RuntimeError("complexity weights need scene tokens, but none were captured")
            ids, d_inter, d_intra = cluster_complexity(T_cos, token_labels, complexity_neighbors)
            p = complexity_weights(d_inter, d_intra, complexity_tau)
            csize = np.array([(token_labels == c).sum() for c in ids])
            per_valid = (p / csize)[np.searchsorted(ids, token_labels)]
            # Scenes whose gradient pass failed have no cluster; they get the mean
            # scene weight rather than dropping out of training.
            scene_weights = np.full(len(grad_dataset), 1.0 / n)
            scene_weights[np.asarray(valid_indices, dtype=np.int64)] = per_valid
            scene_weights /= scene_weights.sum()
            ess = 1.0 / (len(scene_weights) * float(np.sum(scene_weights ** 2)))
            C = d_inter * d_intra
            print(
                f"  [complexity] tau={complexity_tau:g}"
                f"  l={min(complexity_neighbors, len(ids) - 1)}"
                f"  C min/median/max={C.min():.4f}/{np.median(C):.4f}/{C.max():.4f}"
                f"  cluster weight min/median/max={p.min():.4f}/{np.median(p):.4f}/{p.max():.4f}"
                f"  (max/min {p.max() / p.min():.2f}x)"
                f"  scene max {scene_weights.max() * len(scene_weights):.1f}x uniform"
                f"  ESS {ess:.2f}",
                flush=True,
            )
            cw = dict(
                token_cluster_ids=ids.astype(np.int32),
                token_d_inter=d_inter.astype(np.float32),
                token_d_intra=d_intra.astype(np.float32),
                token_complexity=C.astype(np.float32),
                token_cluster_weights=p.astype(np.float64),
                scene_weights=scene_weights,
                complexity_tau=np.array(complexity_tau),
                complexity_neighbors=np.array(complexity_neighbors),
            )

        # ---- token-cluster balance weights ----
        if cluster_balance is not None:
            if token_labels is None:
                raise RuntimeError("cluster balance needs scene tokens, but none were captured")
            groups = np.asarray(scene_groups)
            if len(groups) != len(grad_dataset):
                raise RuntimeError(
                    f"scene_groups has {len(groups)} entries, grad_dataset has {len(grad_dataset)} scenes"
                )
            valid = np.asarray(valid_indices, dtype=np.int64)
            per_valid = cluster_balance_weights(token_labels, groups[valid], cluster_balance)
            # Scenes whose gradient pass failed have no cluster; they keep their
            # pool share (weight 1/N) instead of dropping out of training.
            scene_weights = np.full(len(grad_dataset), 1.0 / len(grad_dataset))
            scene_weights[valid] = per_valid * (len(valid) / len(grad_dataset))
            scene_weights /= scene_weights.sum()
            ess = 1.0 / (len(scene_weights) * float(np.sum(scene_weights ** 2)))
            csz = np.bincount(token_labels)
            csz = csz[csz > 0]
            print(
                f"  [cluster-balance] alpha={cluster_balance:g}"
                f"  token cluster size min/median/max={csz.min()}/{int(np.median(csz))}/{csz.max()}"
                f"  scene weight min/max={scene_weights.min() * len(scene_weights):.2f}x/"
                f"{scene_weights.max() * len(scene_weights):.2f}x uniform"
                f"  ESS {ess:.2f}",
                flush=True,
            )
            cw = dict(
                scene_weights=scene_weights,
                cluster_balance=np.array(cluster_balance),
            )

        if out_dir is not None:
            from pathlib import Path
            save_dir = Path(out_dir) / "grad_clusters"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"epoch_{epoch:04d}.npz"
            stat_prefix = "eucl" if metric == "euclidean" else "cos"
            save_kwargs: dict = dict(
                G=G,
                labels=labels,
                valid_indices=np.array(valid_indices, dtype=np.int64),
                epoch=np.array(epoch),
                k=np.array(k_actual),
                proj_dim=np.array(proj_dim),
                clip_grad=np.array(clip_grad),
                metric=np.array(metric),
                cosine_center=np.array(cosine_center),
            )
            save_kwargs[f"{stat_prefix}_mean"] = np.array(cluster_stats["mean"])
            save_kwargs[f"{stat_prefix}_median"] = np.array(cluster_stats["median"])
            save_kwargs[f"{stat_prefix}_min"] = np.array(cluster_stats["min"])
            save_kwargs[f"{stat_prefix}_max"] = np.array(cluster_stats["max"])
            save_kwargs[f"{stat_prefix}_per_cluster"] = np.array(
                cluster_stats["per_cluster"], dtype=np.float32
            )
            if T is not None:
                save_kwargs["T"] = T
            if token_labels is not None:
                save_kwargs["token_labels"] = token_labels
                save_kwargs[f"token_{stat_prefix}_mean"] = np.array(token_cluster_stats["mean"])
                save_kwargs[f"token_{stat_prefix}_median"] = np.array(token_cluster_stats["median"])
                save_kwargs[f"token_{stat_prefix}_min"] = np.array(token_cluster_stats["min"])
                save_kwargs[f"token_{stat_prefix}_max"] = np.array(token_cluster_stats["max"])
                save_kwargs[f"token_{stat_prefix}_per_cluster"] = np.array(
                    token_cluster_stats["per_cluster"], dtype=np.float32
                )
            if ari_score is not None:
                save_kwargs["ari_grad_vs_tokens"] = np.array(ari_score)
            if cw is not None:
                save_kwargs.update(cw)
            np.savez_compressed(save_path, **save_kwargs)
            print(f"  [grad-cluster] saved to {save_path}", flush=True)

    if world_size > 1:
        dist.barrier()
    return scene_weights
