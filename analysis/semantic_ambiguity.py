"""Correlate text semantic ambiguity with LLM misclassification.

We measure each paper's *semantic ambiguity* purely from its SBERT text
embedding (independent of the LLM, so the correlation with LLM errors is not
circular), then test whether ambiguous texts are the ones the LLM gets wrong.

Three LLM-independent ambiguity scores (higher == more ambiguous), all derived
from SBERT embeddings of ``raw_texts``:

  * kNN label entropy  -- entropy of the true-class distribution among a node's
                          nearest neighbours in semantic space (neighbours drawn
                          from labelled *training* nodes). The local view.
  * centroid soft-entropy -- entropy of softmax(cosine-sim to per-class
                          centroids). The global view.
  * centroid neg-margin   -- negated top-2 centroid cosine margin.

Each is correlated against the LLM error indicator ``argmax(logits) != y`` via
point-biserial r, AUROC and a Mann-Whitney U test, on the held-out test split.

Run (Hydra; paths/overrides optional)::

    python -m analysis.semantic_ambiguity
    python -m analysis.semantic_ambiguity +llm_logits=llm_logits.pt +k=15
    python -m analysis.semantic_ambiguity +dataset_name=cora +split=test

Outputs a printed report and ``figures/semantic_ambiguity_vs_llm_error.png``.
"""
from pathlib import Path

import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from scipy.stats import mannwhitneyu, pointbiserialr, spearmanr
from sklearn.metrics import roc_auc_score

from data.data import TAGDataset


# ----------------------------------------------------------------- ambiguity metrics
def _normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def knn_label_entropy(emb: np.ndarray, y: np.ndarray, train_idx: np.ndarray,
                      num_classes: int, k: int = 15) -> np.ndarray:
    """Entropy of neighbours' true labels for each node (local ambiguity)."""
    sims = emb @ emb[train_idx].T  # (N, n_train) cosine sim to labelled nodes
    out = np.empty(len(emb))
    log_k = np.log(num_classes)
    for i in range(len(emb)):
        order = np.argsort(-sims[i])
        nbr = train_idx[order]
        nbr = nbr[nbr != i][:k]  # drop self if node is itself a training node
        p = np.bincount(y[nbr], minlength=num_classes).astype(float)
        p /= p.sum()
        out[i] = -(p * np.log(p + 1e-12)).sum() / log_k
    return out


def centroid_metrics(emb: np.ndarray, y: np.ndarray, train_idx: np.ndarray,
                     num_classes: int, temp: float = 0.05):
    """Softmax-entropy and top-2 margin of cosine sim to per-class centroids."""
    cent = np.stack([emb[train_idx][y[train_idx] == c].mean(0)
                     for c in range(num_classes)])
    cent = _normalize(cent)
    sims = emb @ cent.T  # (N, K)
    z = sims / temp
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    entropy = -(p * np.log(p + 1e-12)).sum(1) / np.log(num_classes)
    s = np.sort(sims, axis=1)
    margin = s[:, -1] - s[:, -2]  # small == ambiguous
    return entropy, margin


# ----------------------------------------------------------------- reporting
def report(metrics: dict, is_wrong: np.ndarray, llm_entropy: np.ndarray,
           mask: np.ndarray, title: str):
    w = is_wrong[mask]
    logger.info(f"{'='*72}")
    logger.info(f"{title}  (n={int(mask.sum())}, errors={int(w.sum())}, "
                f"err_rate={w.mean():.3f})")
    logger.info(f"{'='*72}")
    logger.info(f"{'metric':<26}{'point-biserial r':>18}{'p':>11}{'AUROC':>8}{'MWU p':>10}")
    for name, vals in metrics.items():
        v = vals[mask]
        r, p = pointbiserialr(v, w)
        auc = roc_auc_score(w, v) if w.min() != w.max() else float("nan")
        mw_p = mannwhitneyu(v[w == 1], v[w == 0], alternative="greater").pvalue
        logger.info(f"{name:<26}{r:>18.4f}{p:>11.1e}{auc:>8.3f}{mw_p:>10.1e}")
    # mediation: ambiguity -> LLM uncertainty -> error
    primary = next(iter(metrics.values()))[mask]
    rs, _ = spearmanr(primary, llm_entropy[mask])
    r_llm, p_llm = pointbiserialr(llm_entropy[mask], w)
    auc_llm = roc_auc_score(w, llm_entropy[mask]) if w.min() != w.max() else float("nan")
    logger.info(f"[mediation] Spearman(ambiguity, LLM pred-entropy) = {rs:.3f}")
    logger.info(f"[reference] LLM pred-entropy -> error: r={r_llm:.3f} "
                f"(p={p_llm:.1e}), AUROC={auc_llm:.3f}")


def quartile_table(vals: np.ndarray, is_wrong: np.ndarray, mask: np.ndarray, name: str):
    v, w = vals[mask], is_wrong[mask]
    q = np.quantile(v, [0, .25, .5, .75, 1.0])
    logger.info(f"Error rate by {name} quartile:")
    for i in range(4):
        lo, hi = q[i], q[i + 1]
        sel = (v >= lo) & (v <= hi) if i == 3 else (v >= lo) & (v < hi)
        logger.info(f"  Q{i+1} [{lo:.3f},{hi:.3f}]  n={int(sel.sum()):4d}  "
                    f"err_rate={w[sel].mean():.3f}")


def make_figure(vals: np.ndarray, is_wrong: np.ndarray, mask: np.ndarray,
                metric_name: str, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v, w = vals[mask], is_wrong[mask]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].hist(v[w == 0], bins=20, alpha=.6, density=True, label="correct")
    ax[0].hist(v[w == 1], bins=20, alpha=.6, density=True, label="misclassified")
    ax[0].set_xlabel(metric_name)
    ax[0].set_ylabel("density")
    ax[0].set_title("Ambiguity by LLM outcome")
    ax[0].legend()

    nb = 8
    edges = np.quantile(v, np.linspace(0, 1, nb + 1))
    edges[-1] += 1e-9
    centers, rates = [], []
    for i in range(nb):
        sel = (v >= edges[i]) & (v < edges[i + 1])
        if sel.sum():
            centers.append((edges[i] + edges[i + 1]) / 2)
            rates.append(w[sel].mean())
    ax[1].plot(centers, rates, "o-")
    ax[1].axhline(w.mean(), ls="--", c="grey", label=f"overall err {w.mean():.2f}")
    ax[1].set_xlabel(f"{metric_name} (binned)")
    ax[1].set_ylabel("LLM error rate")
    ax[1].set_title("Error rate vs semantic ambiguity")
    ax[1].legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    logger.info(f"Saved figure -> {out_path}")


# ----------------------------------------------------------------- entry point
@hydra.main(version_base=None, config_path="../configs", config_name="main")
def main(cfg: DictConfig):
    name = cfg.get("dataset_name", cfg.llm.tag_dataset_name)
    logits_path = Path(cfg.project_root) / cfg.get("llm_logits", "llm_logits.pt")
    split = cfg.get("split", "test")  # test | val | train | all
    k = int(cfg.get("k", 15))

    logger.info(f"dataset={name}  logits={logits_path}  split={split}  k={k}")
    dataset = TAGDataset(cfg, name=name)
    data = dataset.data
    num_classes = dataset.num_classes

    y = data.y.numpy()
    raw_texts = list(data.raw_texts)
    # TAGDataset.data drops label_texts; read class names from the processed tensor.
    label_texts = list(getattr(dataset.load_processed_data, "label_texts",
                               list(range(num_classes))))
    sbert = dataset.load_sbert_x.float().numpy()

    logits = torch.load(logits_path, weights_only=False).float()
    prob = torch.softmax(logits, dim=1).numpy()
    pred = prob.argmax(1)
    is_wrong = (pred != y).astype(int)
    llm_entropy = -(prob * np.log(prob + 1e-12)).sum(1) / np.log(num_classes)

    emb = _normalize(sbert)
    train_idx = np.where(data.train_mask.numpy())[0]

    knn_ent = knn_label_entropy(emb, y, train_idx, num_classes, k=k)
    cent_ent, cent_margin = centroid_metrics(emb, y, train_idx, num_classes)
    metrics = {
        f"kNN label entropy (k={k})": knn_ent,
        "centroid soft-entropy": cent_ent,
        "centroid neg-margin": -cent_margin,
    }

    masks = {
        "test": data.test_mask.numpy(),
        "val": data.val_mask.numpy(),
        "train": data.train_mask.numpy(),
        "all": np.ones(len(y), bool),
    }
    eval_mask = masks[split]

    report(metrics, is_wrong, llm_entropy, eval_mask, f"{name.upper()} [{split}]")
    report(metrics, is_wrong, llm_entropy, masks["all"], f"{name.upper()} [all nodes]")
    quartile_table(knn_ent, is_wrong, eval_mask, f"kNN label entropy [{split}]")

    fig_path = Path(cfg.project_root) / "figures" / "semantic_ambiguity_vs_llm_error.png"
    make_figure(knn_ent, is_wrong, eval_mask, "kNN label entropy", fig_path)

    # qualitative: most-ambiguous misclassified texts
    logger.info(f"{'='*72}")
    logger.info(f"Most semantically-ambiguous MISCLASSIFIED texts [{split}]")
    logger.info(f"{'='*72}")
    eval_idx = np.where(eval_mask)[0]
    wrong = eval_idx[is_wrong[eval_idx] == 1]
    for i in wrong[np.argsort(-knn_ent[wrong])][:6]:
        txt = raw_texts[i].replace("\n", " ").replace("\t", " ")[:280]
        logger.info(f"[node {i}] ambiguity={knn_ent[i]:.3f}  "
                    f"true={label_texts[y[i]]}  pred={label_texts[pred[i]]}  "
                    f"(p_pred={prob[i].max():.2f})")
        logger.info(f"  {txt}")


if __name__ == "__main__":
    main()
