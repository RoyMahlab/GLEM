"""Shared plumbing for the cross-modal-transfer analyses.

Both ``analysis.transfer_harm`` (does uniform distillation *hurt* an identifiable
node population?) and ``analysis.gate_quality`` (which selection signal best
predicts *who should teach whom*?) need the same three things:

  1. the latest saved logits for a dataset, solo (pre-transfer) or EM (post-transfer);
  2. per-node signals -- kNN semantic ambiguity, local homophily, split masks;
  3. a fast AUROC that can be called thousands of times under a bootstrap.

Signals are cached to ``analysis_output/node_signals/<name>.npz`` because the
SBERT kNN pass over sportsfit (173K nodes) is the slow part and neither script
should pay it twice.

Nothing here trains anything -- every function reads artefacts already on disk.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from analysis.gnn_entropy_error import node_homophily
from data.data import TAGDataset

DATASETS = TAGDataset.AVAILABLE_DATASETS + TAGDataset.WEBKB_DATASETS
DATASETS_SET = set(DATASETS)


# ───────────────────────────────────────────── logits discovery
# Mirrors the helpers in datasets_analysis.ipynb so both entry points resolve the
# same files (latest by mtime; EM and solo runs live in differently-named dirs).


def _is_em_path(rel_path: Path) -> bool:
    """True if any path segment marks an EM run (``em-<name>`` dir or ``.../em/``)."""
    return any(seg == "em" or seg.startswith("em-") for seg in rel_path.parts)


def _dataset_of(rel_path: Path) -> str | None:
    """Map an ``analysis_output`` path to its dataset name, solo or EM layout."""
    for seg in rel_path.parts:
        s = seg[3:] if seg.startswith("em-") else seg
        base = s.split("-")[0]  # dataset names contain no hyphens
        if base in DATASETS_SET:
            return base
    return None


def latest_logits(analysis_root: Path, name: str, kind: str, em: bool = False):
    """``(path, tensor)`` of the most-recent ``{kind}_logits.pt`` for ``name``.

    ``kind`` is ``"gnn"`` or ``"llm"``; ``em`` picks the EM run rather than the
    solo baseline. Returns ``(None, None)`` when nothing matches.

    Prefer :func:`paired_run_logits` for before/after comparisons -- picking each
    side independently by mtime can straddle two different pipeline invocations.
    """
    cands = [
        p for p in analysis_root.rglob(f"{kind}_logits.pt")
        if _is_em_path(p.relative_to(analysis_root)) == em
        and _dataset_of(p.relative_to(analysis_root)) == name
    ]
    if not cands:
        return None, None
    latest = max(cands, key=lambda p: p.stat().st_mtime)
    return latest, torch.load(latest, weights_only=False).float()


def paired_run_logits(analysis_root: Path, name: str, uid: str | None = None):
    """Solo *and* EM logits from **one** pipeline invocation: ``<name>-<uid>/{,em/}``.

    A before/after comparison is only meaningful if both sides came from the same
    run -- same split, same seed, same config. ``main.py`` writes the solo logits to
    ``<name>-<uid>/`` and that run's EM logits to ``<name>-<uid>/em/``, so a uid dir
    holding all four files is a self-consistent pair.

    Picking each side independently by mtime does **not** give you this: EM-only
    runs (``run_solo=false``) write ``em-<name>/`` and newer uid dirs with no solo
    baseline, so "latest EM" and "latest solo" can be weeks and several configs
    apart. Returns ``(uid, {(kind, em): (path, tensor)})``, or ``(None, None)`` if
    no uid dir is complete.
    """
    cands = []
    for d in analysis_root.glob(f"{name}-*"):
        want = {(k, e): (d / "em" / f"{k}_logits.pt") if e else (d / f"{k}_logits.pt")
                for k in ("gnn", "llm") for e in (False, True)}
        if all(p.exists() for p in want.values()):
            cands.append((d.name.split("-", 1)[1], want))
    if not cands:
        return None, None
    if uid is not None:
        cands = [c for c in cands if c[0] == uid]
        if not cands:
            return None, None
    run_uid, paths = max(cands, key=lambda c: max(p.stat().st_mtime for p in c[1].values()))
    return run_uid, {k: (p, torch.load(p, weights_only=False).float())
                     for k, p in paths.items()}


# ───────────────────────────────────────────── per-node signals


def semantic_ambiguity_per_node(emb: np.ndarray, y: np.ndarray, train_idx: np.ndarray,
                                num_classes: int, k: int = 15,
                                batch: int = 2048) -> np.ndarray:
    """Per-node SBERT kNN label-entropy, normalized to [0, 1].

    Batched, memory-bounded equivalent of
    ``analysis.semantic_ambiguity.knn_label_entropy`` (identical per-node values):
    neighbours come from labelled *training* nodes, the node's own index is
    dropped, entropy is normalized by ``log(num_classes)``.
    """
    emb = emb.astype(np.float32)
    train_emb = emb[train_idx]
    log_k = np.log(num_classes)
    out = np.empty(len(emb))
    for start in range(0, len(emb), batch):
        end = min(start + batch, len(emb))
        sims = emb[start:end] @ train_emb.T
        kth = min(k, sims.shape[1] - 1)
        top = np.argpartition(-sims, kth=kth, axis=1)[:, :k + 1]
        for r, gi in enumerate(range(start, end)):
            cand = train_idx[top[r]]
            cand = cand[np.argsort(-sims[r, top[r]])]
            cand = cand[cand != gi][:k]
            p = np.bincount(y[cand], minlength=num_classes).astype(float)
            p /= p.sum()
            out[gi] = -(p * np.log(p + 1e-12)).sum() / log_k
    return out


def soft_local_homophily(edge_index: torch.Tensor, probs: torch.Tensor) -> np.ndarray:
    """GLANCE's label-free local-homophily estimate ``h_v = p_v . mean_{u in N(v)} p_u``.

    Unlike :func:`analysis.gnn_entropy_error.node_homophily` this never touches
    neighbour *labels*, so it is usable at inference time. Isolated nodes get NaN.
    """
    src, dst = edge_index[0], edge_index[1]
    n, c = probs.shape
    nbr_sum = torch.zeros(n, c).index_add_(0, src, probs[dst])
    deg = torch.zeros(n).index_add_(0, src, torch.ones(src.numel()))
    nbr_mean = nbr_sum / deg.clamp(min=1).unsqueeze(1)
    h = (probs * nbr_mean).sum(1)
    return torch.where(deg > 0, h, torch.tensor(float("nan"))).numpy()


def node_signals(cfg, name: str, k: int = 15, cache: bool = True) -> dict:
    """Label-derived per-node signals for ``name``, cached to ``analysis_output``.

    Returns ``y``, ``ambiguity`` (SBERT kNN label-entropy), ``homophily`` (true
    local homophily, NaN on isolated nodes), ``degree``, the three split masks and
    ``num_classes``. ``soft_homophily`` is *not* here -- it needs GNN probabilities
    and is computed by the caller via :func:`soft_local_homophily`.
    """
    root = Path(cfg.project_root) / "analysis_output" / "node_signals"
    path = root / f"{name}-k{k}.npz"
    if cache and path.exists():
        z = np.load(path)
        return {key: z[key] for key in z.files} | {"num_classes": int(z["num_classes"])}

    ds = TAGDataset(cfg, name=name)
    d = ds.data
    y = d.y.numpy()
    emb = ds.load_sbert_x.float().numpy()
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    train_idx = np.where(d.train_mask.numpy())[0]

    deg = np.bincount(d.edge_index[0].numpy(), minlength=len(y))
    out = {
        "y": y,
        "ambiguity": semantic_ambiguity_per_node(emb, y, train_idx, ds.num_classes, k=k),
        "homophily": node_homophily(d.edge_index, d.y, len(y)).numpy(),
        "degree": deg,
        "train_mask": d.train_mask.numpy(),
        "val_mask": d.val_mask.numpy(),
        "test_mask": d.test_mask.numpy(),
        "edge_index": d.edge_index.numpy(),
        "num_classes": np.array(ds.num_classes),
    }
    if cache:
        root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **out)
    out["num_classes"] = int(ds.num_classes)
    return out


def split_mask(sig: dict, split: str) -> np.ndarray:
    """Boolean node mask for ``train`` | ``val`` | ``test`` | ``all``."""
    if split == "all":
        return np.ones(len(sig["y"]), dtype=bool)
    return sig[f"{split}_mask"].astype(bool)


# ───────────────────────────────────────────── metrics


def fast_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney U), tie-corrected.

    Equivalent to ``sklearn.metrics.roc_auc_score`` but cheap enough to call
    inside a bootstrap loop. NaN when one class is absent.
    """
    labels = labels.astype(bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks within tied score groups so ties contribute 0.5 each
    s = scores[order]
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[j] == s[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return (ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def predictive_entropy(logits: torch.Tensor, num_classes: int) -> np.ndarray:
    """Normalized Shannon entropy of each row's softmax, in [0, 1]."""
    p = torch.softmax(logits, dim=-1)
    ent = -(p * (p + 1e-12).log()).sum(-1)
    return (ent / np.log(num_classes)).numpy()


def quantile_bins(values: np.ndarray, nbins: int) -> tuple[np.ndarray, np.ndarray]:
    """Quantile bin assignment tolerant of heavily-tied signals.

    Local homophily is very discrete on low-degree graphs (0, 1/2, 1 dominate), so
    plain ``np.quantile`` edges collapse. Duplicate edges are dropped and the
    realised bin count may be < ``nbins``; returns ``(bin_index, edges)`` with
    ``bin_index == -1`` for NaN entries.
    """
    finite = np.isfinite(values)
    edges = np.unique(np.quantile(values[finite], np.linspace(0, 1, nbins + 1)))
    if len(edges) < 2:
        edges = np.array([values[finite].min(), values[finite].max() + 1e-9])
    idx = np.full(len(values), -1, dtype=int)
    idx[finite] = np.clip(np.digitize(values[finite], edges[1:-1], right=False),
                          0, len(edges) - 2)
    return idx, edges
