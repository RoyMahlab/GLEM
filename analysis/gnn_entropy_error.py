"""Relate a node's GNN predictive entropy to its misclassification rate.

For each node we take the GNN's softmax over ``gnn_logits.pt`` and compute its
(base-K normalized) predictive entropy -- a model-internal uncertainty signal.
We then test whether high-entropy nodes are the ones the GNN gets wrong, and
plot the error rate as a function of entropy (a reliability-style curve).

Run (Hydra; overrides optional)::

    python -m analysis.gnn_entropy_error
    python -m analysis.gnn_entropy_error +gnn_logits=gnn_logits.pt +split=test +nbins=10

Outputs a printed report and ``figures/gnn_entropy_vs_error.png``.
"""
from pathlib import Path

import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from scipy.stats import pointbiserialr, spearmanr
from sklearn.metrics import roc_auc_score

from data.data import TAGDataset


def predictive_entropy(prob: np.ndarray, num_classes: int) -> np.ndarray:
    """Shannon entropy of each row, normalized to [0, 1] by log(K)."""
    return -(prob * np.log(prob + 1e-12)).sum(1) / np.log(num_classes)


def local_homophily(edge_index: np.ndarray, y: np.ndarray, num_nodes: int) -> np.ndarray:
    """Per-node fraction of graph neighbours sharing its label.

    Returns NaN for isolated nodes (no neighbours -> homophily undefined).
    """
    src, dst = edge_index[0], edge_index[1]
    same = (y[src] == y[dst]).astype(float)
    deg = np.bincount(src, minlength=num_nodes).astype(float)
    same_sum = np.bincount(src, weights=same, minlength=num_nodes)
    with np.errstate(invalid="ignore", divide="ignore"):
        h = same_sum / deg
    h[deg == 0] = np.nan
    return h

def node_homophily(edge_index: torch.Tensor, y: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Compute per-node homophily for a PyTorch Geometric Data object.
    
    Returns a tensor of shape [num_nodes] with homophily in [0, 1].
    Isolated nodes (no neighbors) get NaN.
    """
    src, dst = edge_index[0], edge_index[1]

    # 1 if the neighbor shares the same label, 0 otherwise
    same_label = (y[src] == y[dst]).float()

    # Sum of same-label neighbors per node
    same_count = torch.zeros(num_nodes).scatter_add(0, src, same_label)

    # Total degree per node - assuming undirected graph (each edge counts for both src and dst)
    degree = torch.zeros(num_nodes).scatter_add(
        0, src, torch.ones(src.size(0))
    )

    # Homophily = same_label_neighbors / degree (NaN for isolated nodes)
    homophily = torch.where(degree > 0, same_count / degree, torch.tensor(float('nan')))

    return homophily

def quartile_table(entropy: np.ndarray, is_wrong: np.ndarray, mask: np.ndarray):
    v, w = entropy[mask], is_wrong[mask]
    q = np.quantile(v, [0, .25, .5, .75, 1.0])
    logger.info("Error rate by predictive-entropy quartile:")
    for i in range(4):
        lo, hi = q[i], q[i + 1]
        sel = (v >= lo) & (v <= hi) if i == 3 else (v >= lo) & (v < hi)
        logger.info(f"  Q{i+1} [{lo:.3f},{hi:.3f}]  n={int(sel.sum()):4d}  "
                    f"err_rate={w[sel].mean():.3f}")


def make_figure(entropy: np.ndarray, is_wrong: np.ndarray, homophily: np.ndarray,
                mask: np.ndarray, nbins: int, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v, w = entropy[mask], is_wrong[mask]
    fig, ax = plt.subplots(1, 3, figsize=(19, 5))

    ax[0].hist(v[w == 0], bins=20, alpha=.6, density=True, label="correct")
    ax[0].hist(v[w == 1], bins=20, alpha=.6, density=True, label="misclassified")
    ax[0].set_xlabel("GNN predictive entropy")
    ax[0].set_ylabel("density")
    ax[0].set_title("Entropy by GNN outcome")
    ax[0].legend()

    # equal-count (quantile) bins -> error rate per bin
    edges = np.quantile(v, np.linspace(0, 1, nbins + 1))
    edges[-1] += 1e-9
    centers, rates, counts = [], [], []
    for i in range(nbins):
        sel = (v >= edges[i]) & (v < edges[i + 1])
        if sel.sum():
            centers.append((edges[i] + edges[i + 1]) / 2)
            rates.append(w[sel].mean())
            counts.append(int(sel.sum()))
    ax[1].plot(centers, rates, "o-")
    ax[1].axhline(w.mean(), ls="--", c="grey", label=f"overall err {w.mean():.2f}")
    ax[1].set_xlabel("GNN predictive entropy (quantile-binned)")
    ax[1].set_ylabel("error rate")
    ax[1].set_title("Error rate vs predictive entropy")
    ax[1].set_ylim(0, 1)
    ax[1].legend()

    # predictive entropy vs local homophily: mean entropy per homophily bin
    hm, vm = homophily[mask], v
    ok = ~np.isnan(hm)
    hm, vm = hm[ok], vm[ok]
    ax[2].scatter(hm, vm, s=8, alpha=.25, color="grey", label="nodes")
    edges = np.linspace(0, 1, nbins + 1)
    centers, means = [], []
    for i in range(nbins):
        hi = (edges[i + 1] + 1e-9) if i == nbins - 1 else edges[i + 1]
        sel = (hm >= edges[i]) & (hm < hi)
        if sel.sum():
            centers.append((edges[i] + edges[i + 1]) / 2)
            means.append(vm[sel].mean())
    ax[2].plot(centers, means, "o-", color="C3", label="mean entropy")
    ax[2].set_xlabel("local homophily (frac. neighbours sharing label)")
    ax[2].set_ylabel("GNN predictive entropy")
    ax[2].set_title("Predictive entropy vs local homophily")
    ax[2].legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    logger.info(f"Saved figure -> {out_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="main")
def main(cfg: DictConfig):
    name = cfg.get("dataset_name", cfg.llm.tag_dataset_name)
    logits_path = Path(cfg.project_root) / cfg.get("gnn_logits", "gnn_logits.pt")
    split = cfg.get("split", "test")  # test | val | train | all
    nbins = int(cfg.get("nbins", 10))

    logger.info(f"dataset={name}  logits={logits_path}  split={split}  nbins={nbins}")
    dataset = TAGDataset(cfg, name=name)
    data = dataset.data
    num_classes = dataset.num_classes
    y = data.y.numpy()

    logits = torch.load(logits_path, weights_only=False).float()
    prob = torch.softmax(logits, dim=1).numpy()
    pred = prob.argmax(1)
    is_wrong = (pred != y).astype(int)
    entropy = predictive_entropy(prob, num_classes)
    homophily = node_homophily(data.edge_index, data.y, len(y)).numpy()

    masks = {
        "test": data.test_mask.numpy(),
        "val": data.val_mask.numpy(),
        "train": data.train_mask.numpy(),
        "all": np.ones(len(y), bool),
    }
    mask = masks[split]
    w = is_wrong[mask]

    logger.info(f"{'='*72}")
    logger.info(f"{name.upper()} [{split}]  (n={int(mask.sum())}, errors={int(w.sum())}, "
                f"err_rate={w.mean():.3f})")
    logger.info(f"{'='*72}")
    r, p = pointbiserialr(entropy[mask], w)
    auc = roc_auc_score(w, entropy[mask]) if w.min() != w.max() else float("nan")
    logger.info(f"predictive entropy -> error:  point-biserial r={r:.4f} (p={p:.1e}), "
                f"AUROC={auc:.3f}")
    quartile_table(entropy, is_wrong, mask)

    # how entropy relates to local homophily (isolated nodes excluded)
    hm = homophily[mask]
    ok = ~np.isnan(hm)
    if ok.sum() > 1:
        rs, ps = spearmanr(hm[ok], entropy[mask][ok])
        logger.info(f"local homophily vs predictive entropy:  Spearman rho={rs:.4f} "
                    f"(p={ps:.1e}, n={int(ok.sum())})")

    fig_path = Path(cfg.project_root) / "figures" / "gnn_entropy_vs_error.png"
    make_figure(entropy, is_wrong, homophily, mask, nbins, fig_path)


if __name__ == "__main__":
    main()
