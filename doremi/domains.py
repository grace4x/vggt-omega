"""The 50 patch-average clusters, read as DoReMi's *domains*.

`patch_avg_clustering/cluster.py` writes a clusters json whose every entry is a
`{dataset, subset, scene}` triple, and `collate_mixed` already puts `subset` and
`scene_id` on every batch -- so a batch can be labelled by domain without touching
the datasets at all. That is what `Domains.ids_for_batch` does, and it is the only
plumbing DoReMi needs on the data side.

Two choices here differ from the LLM setting and matter more than they look:

**The baseline mixture is proportional to cluster size, not uniform.** An epoch of
`build_concat_trainset` lists every scene once, so the mixture a plain run trains on
is `alpha_baseline[j] = |C_j| / N`. That is the distribution the reference model is
trained on, so it is the distribution the learned weights have to be measured
against -- `ratio = alpha / alpha_baseline` is the number this whole pipeline exists
to produce, and it is 1.0 everywhere for a plain run.

**Smoothing pulls toward the baseline, not toward uniform.** DoReMi mixes a little
uniform mass into the weights each step so no domain can be starved to zero. Uniform
over 50 clusters would hand the two 2-scene clusters 2% of all samples each, i.e.
~23x oversampling, purely as a floor. Mixing toward the baseline keeps the same
"nothing goes to zero" guarantee at the mixture a plain run already survives.

    python3 doremi/domains.py                      # the domain table
    python3 doremi/domains.py --min-size 10        # what --min-domain-size drops
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_CLUSTERS = REPO / "patch_avg_clustering" / "clusters_dl3dv_scannet_patchavg_k50_none.json"


def scene_key(subset: str, scene: str) -> str:
    """`subset/scene`, the same identifier `--scene-list` and `subset.py` use."""
    return f"{subset}/{scene}"


@dataclass
class Domains:
    """A contiguous 0..D-1 domain indexing over a clusters json.

    `ids[d]` is the original cluster number, which is what gets written back out --
    domain indices are an implementation detail of the optimiser, cluster ids are
    what the contact sheets and `get_cluster_loss_html.py` speak.
    """

    path: Path
    ids: list[int]
    sizes: list[int]
    meta: list[dict]
    scene_to_domain: dict[str, int]
    index_meta: dict = field(default_factory=dict)
    # `subset/scene` -> which dataset it came from. Kept per scene, not just as the
    # per-cluster mix, because the datasets differ in depth-GT quality and 46 of the
    # 48 domains are >95% one of them -- so any per-cluster statistic has to be
    # checkable against the dataset split (see report.py).
    scene_to_dataset: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def n_scenes(self) -> int:
        return len(self.scene_to_domain)

    def baseline(self) -> np.ndarray:
        """The mixture a plain shuffled epoch realises: proportional to cluster size."""
        sizes = np.asarray(self.sizes, dtype=np.float64)
        return sizes / sizes.sum()

    def uniform(self) -> np.ndarray:
        return np.full(len(self), 1.0 / len(self))

    def domain_of(self, subset: str, scene: str) -> int:
        """Domain index, or -1 for a scene the clusters json does not cover."""
        return self.scene_to_domain.get(scene_key(subset, scene), -1)

    def ids_for_batch(self, batch: dict) -> "np.ndarray":
        """(B,) domain indices for a `collate_mixed` batch; -1 where unknown."""
        return np.array(
            [self.domain_of(s, i) for s, i in zip(batch["subset"], batch["scene_id"])],
            dtype=np.int64,
        )

    def scene_list(self) -> list[str]:
        """Every covered scene, sorted -- writable straight out as a `--scene-list`."""
        return sorted(self.scene_to_domain)

    def label(self, d: int) -> str:
        """`c07 (n=61, scannet 60/dl3dv 1)`, for logs and tables."""
        mix = self.meta[d].get("datasets") or {}
        parts = "/".join(f"{n} {c}" for n, c in sorted(mix.items(), key=lambda kv: -kv[1]))
        return f"c{self.ids[d]:02d} (n={self.sizes[d]}{', ' + parts if parts else ''})"


def load_domains(path: Path | str | None = None, *, min_size: int = 1) -> Domains:
    """Read a `cluster.py` clusters json into a domain map.

    `min_size` drops clusters too small to estimate a loss on. Their scenes leave
    the domain map, which means the trainer drops them from the training pool too
    (see `doremi/train.py --min-domain-size`) -- a cluster of 2 scenes cannot
    support a weight, and leaving it in only adds a very loud, very noisy domain.
    """
    path = Path(path) if path is not None else DEFAULT_CLUSTERS
    if not path.exists():
        raise SystemExit(f"clusters json not found: {path}")
    index = json.loads(path.read_text())
    if "clusters" not in index:
        raise SystemExit(f"{path} has no 'clusters' key; is it a cluster.py output?")

    ids: list[int] = []
    sizes: list[int] = []
    meta: list[dict] = []
    scene_to_domain: dict[str, int] = {}
    scene_to_dataset: dict[str, str] = {}
    dropped = 0
    for cluster in index["clusters"]:
        scenes = cluster["scenes"]
        if len(scenes) < min_size:
            dropped += len(scenes)
            continue
        d = len(ids)
        ids.append(int(cluster["cluster"]))
        sizes.append(len(scenes))
        meta.append(
            {
                "datasets": cluster.get("datasets"),
                "cohesion": cluster.get("cohesion"),
                "complexity": cluster.get("complexity"),
                "d_inter": cluster.get("d_inter"),
                "d_intra": cluster.get("d_intra"),
            }
        )
        for s in scenes:
            key = scene_key(s["subset"], s["scene"])
            scene_to_domain[key] = d
            # `dataset` is written by patch_avg_clustering/cluster.py; a mapping from
            # clustering/ has only subset/scene, so fall back as subset.py does.
            scene_to_dataset[key] = s.get("dataset", s["subset"])

    if not ids:
        raise SystemExit(f"no cluster in {path.name} has >= {min_size} scenes")

    index_meta = {
        k: index[k]
        for k in ("k", "features", "center", "patch_layers", "model", "dedup", "num_frames", "split")
        if k in index
    }
    index_meta["clusters_scenes"] = int(index.get("num_scenes", len(scene_to_domain)))
    index_meta["dropped_by_min_size"] = dropped
    index_meta["min_size"] = min_size
    return Domains(path=path, ids=ids, sizes=sizes, meta=meta,
                   scene_to_domain=scene_to_domain, index_meta=index_meta,
                   scene_to_dataset=scene_to_dataset)


# --------------------------------------------------------------------------- #
# weights file
# --------------------------------------------------------------------------- #


def save_weights(path: Path, domains: Domains, alpha: np.ndarray, *, extra: dict | None = None) -> None:
    """Write the learned mixture, one row per domain, ordered by `ratio`.

    Ordered rather than by cluster id because the point of reading this file by eye
    is to see which visual modes DoReMi wants more of.
    """
    baseline = domains.baseline()
    rows = [
        {
            "cluster": domains.ids[d],
            "size": domains.sizes[d],
            "alpha": float(alpha[d]),
            "alpha_baseline": float(baseline[d]),
            "ratio": float(alpha[d] / baseline[d]) if baseline[d] > 0 else float("nan"),
            # Per-scene sampling weight, normalised to mean 1 -- the oversampling
            # factor `train.py --mode final` actually applies.
            "scene_factor": float(alpha[d] / baseline[d]) if baseline[d] > 0 else 0.0,
            "datasets": domains.meta[d].get("datasets"),
            "cohesion": domains.meta[d].get("cohesion"),
        }
        for d in range(len(domains))
    ]
    rows.sort(key=lambda r: -r["ratio"])
    payload = {
        "clusters": str(domains.path),
        "n_domains": len(domains),
        "n_scenes": domains.n_scenes,
        **(extra or {}),
        "index": domains.index_meta,
        "domains": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=str) + "\n")


def load_weights(path: Path, domains: Domains) -> np.ndarray:
    """Read a `save_weights` file back as an alpha vector aligned to `domains`.

    Keyed by cluster id, not by position, so a weights file written before a
    `--min-domain-size` change still lines up. A domain with no row gets its
    baseline share and a warning; the alternative -- silently zero -- would drop
    scenes from training on a filename mismatch.
    """
    payload = json.loads(Path(path).read_text())
    by_cluster = {int(r["cluster"]): float(r["alpha"]) for r in payload["domains"]}
    if payload.get("clusters") and Path(payload["clusters"]).name != domains.path.name:
        print(f"[doremi] warning: weights were learned on {Path(payload['clusters']).name}, "
              f"applying to {domains.path.name}")

    baseline = domains.baseline()
    alpha = np.zeros(len(domains))
    missing = []
    for d, cid in enumerate(domains.ids):
        if cid in by_cluster:
            alpha[d] = by_cluster[cid]
        else:
            alpha[d] = baseline[d]
            missing.append(cid)
    if missing:
        print(f"[doremi] warning: {len(missing)} domains absent from {Path(path).name} "
              f"(clusters {missing[:8]}{'...' if len(missing) > 8 else ''}); using their baseline share")
    return alpha / alpha.sum()


def scene_factors(domains: Domains, alpha: np.ndarray) -> dict[str, float]:
    """`subset/scene` -> per-scene sampling weight, normalised to mean 1.

    A scene's weight is `alpha_j / |C_j|`: the domain's target share of the samples,
    spread evenly over the scenes that carry it. Normalising to mean 1 makes the
    number readable as an oversampling factor, and makes the baseline mixture come
    out as all-ones.
    """
    sizes = np.asarray(domains.sizes, dtype=np.float64)
    per_scene = alpha / np.maximum(sizes, 1.0)
    per_scene = per_scene / (per_scene * sizes).sum() * sizes.sum()  # mean 1 over scenes
    return {key: float(per_scene[d]) for key, d in domains.scene_to_domain.items()}


def format_table(domains: Domains, alpha: np.ndarray, *, excess: np.ndarray | None = None,
                 limit: int = 0) -> str:
    """The domain table as text, sorted by ratio. `limit` prints the head and tail."""
    baseline = domains.baseline()
    order = np.argsort(-(alpha / np.maximum(baseline, 1e-12)))
    if limit and len(order) > 2 * limit:
        shown = list(order[:limit]) + [None] + list(order[-limit:])
    else:
        shown = list(order)

    head = f"{'cluster':>8s} {'n':>5s} {'alpha':>8s} {'base':>8s} {'ratio':>7s}"
    if excess is not None:
        head += f" {'excess':>9s}"
    lines = [head + "  datasets"]
    for d in shown:
        if d is None:
            lines.append(f"{'...':>8s}")
            continue
        mix = domains.meta[d].get("datasets") or {}
        mix_s = " ".join(f"{n}={c}" for n, c in sorted(mix.items(), key=lambda kv: -kv[1]))
        line = (f"{domains.ids[d]:>8d} {domains.sizes[d]:>5d} {alpha[d]:>8.5f} "
                f"{baseline[d]:>8.5f} {alpha[d] / max(baseline[d], 1e-12):>7.2f}")
        if excess is not None:
            line += f" {excess[d]:>9.4f}"
        lines.append(line + "  " + mix_s)
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clusters", type=Path, default=None)
    p.add_argument("--min-size", type=int, default=1)
    p.add_argument("--weights", type=Path, default=None, help="a save_weights json to table instead")
    p.add_argument("--scene-list", type=Path, default=None, help="write the covered scenes here")
    args = p.parse_args()

    domains = load_domains(args.clusters, min_size=args.min_size)
    alpha = load_weights(args.weights, domains) if args.weights else domains.baseline()
    print(f"{domains.path.name}: {len(domains)} domains over {domains.n_scenes} scenes "
          f"(min_size={args.min_size}, dropped {domains.index_meta['dropped_by_min_size']} scenes)")
    print(format_table(domains, alpha))

    factors = scene_factors(domains, alpha)
    values = np.array(list(factors.values()))
    print(f"per-scene factors: mean={values.mean():.3f} min={values.min():.3f} max={values.max():.3f}")

    if args.scene_list:
        args.scene_list.write_text("\n".join(domains.scene_list()) + "\n")
        print(f"wrote {args.scene_list} ({domains.n_scenes} scenes)")
