"""Which selection signal actually knows who should teach whom?

Every LLM+GNN co-training method decides *which nodes* cross the modality boundary,
and every one of them decides it with an **endogenous** signal -- a function of the
models' own fitted outputs (GLEM: nothing, uniform; Golden Teacher: each model's
self-confidence; GNN-as-Judge: agreement plus the GNN's probability margin). The
alternative is an **exogenous** signal: a property of the node's position in each
modality's space, computable before either model trains (kNN semantic ambiguity;
local homophily -- GLANCE's routing insight, never used for training exchange).

This script scores those signals against ground truth on the decision that matters.

**Primary test (disagreement-restricted).** Restrict to nodes where the two models
disagree and ask: *is the teacher the one who is right?* Label = ``teacher correct``.
This is the honest framing -- on agreement nodes there is nothing to decide, and
because ``teacher correct AND student wrong`` implies disagreement, the unrestricted
version rewards any signal that merely detects disagreement.

**Secondary test (unrestricted).** Over all nodes, label = ``teacher correct AND
student wrong`` -- i.e. "this node would benefit from transfer". Reported for
completeness; read it knowing the confound above.

Both are evaluated per direction (GNN->LLM and LLM->GNN) on *solo* (pre-transfer)
logits, so nothing here is contaminated by the co-training being evaluated. AUROC
gets a percentile bootstrap CI, and the direction-appropriate exogenous signal is
compared head-to-head against GNN-as-Judge's ``s_pref`` with a **paired** bootstrap
(same resampled nodes for both), which is the comparison a reviewer will want.

Run::

    python -m analysis.gate_quality
    python -m analysis.gate_quality +split=test +n_boot=1000

Outputs ``analysis_output/gate_quality/`` -- ``gate_auroc_{split}.csv`` (every
dataset x direction x selector) and ``gate_auroc_{split}.png``.
"""
from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib
import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.transfer_common import (  # noqa: E402
    DATASETS,
    fast_auroc,
    latest_logits,
    node_signals,
    paired_run_logits,
    predictive_entropy,
    soft_local_homophily,
    split_mask,
)

# Which exogenous signal is the *hypothesised* gate for each direction, and which
# endogenous rule it is benchmarked against in the paired bootstrap.
DIRECTIONS = {
    "gnn->llm": dict(teacher="gnn", student="llm", exo="ambiguity"),
    "llm->gnn": dict(teacher="llm", student="gnn", exo="neg_homophily"),
}
BASELINE = "s_pref"

# Selector family, for grouping in the report.
FAMILY = {
    "student_entropy": "endogenous", "teacher_conf": "endogenous",
    "conf_margin": "endogenous", "s_pref": "endogenous", "jsd": "endogenous",
    "ambiguity": "exogenous", "neg_homophily": "exogenous (uses nbr labels)",
    "neg_soft_homophily": "exogenous (label-free)",
    "random": "baseline",
}


def build_selectors(p_t: np.ndarray, p_s: np.ndarray, sig: dict,
                    soft_h: np.ndarray, num_classes: int, rng) -> dict[str, np.ndarray]:
    """All candidate gate signals, oriented so **higher = follow the teacher**."""
    yt, ys = p_t.argmax(1), p_s.argmax(1)
    n = len(p_t)
    idx = np.arange(n)

    log_c = np.log(num_classes)
    h_s = -(p_s * np.log(p_s + 1e-12)).sum(1) / log_c
    m = 0.5 * (p_t + p_s)
    kl = lambda a, b: (a * (np.log(a + 1e-12) - np.log(b + 1e-12))).sum(1)  # noqa: E731

    return {
        # -- endogenous: functions of the models' own fitted outputs -------------
        "student_entropy": h_s,                                   # student unsure
        "teacher_conf": p_t.max(1),                               # teacher sure
        "conf_margin": p_t.max(1) - p_s.max(1),                   # teacher surer than student
        "s_pref": p_t[idx, yt] - p_t[idx, ys],                    # GNN-as-Judge Eq. filter
        "jsd": 0.5 * (kl(p_t, m) + kl(p_s, m)),                   # symmetric disagreement
        # -- exogenous: properties of the node, computable before training -------
        "ambiguity": sig["ambiguity"],                            # text uninformative
        "neg_homophily": 1.0 - sig["homophily"],                  # structure misleading
        "neg_soft_homophily": 1.0 - soft_h,                       # GLANCE label-free estimate
        # -- sanity floor --------------------------------------------------------
        "random": rng.random(n),
    }


def bootstrap(selectors: dict[str, np.ndarray], labels: np.ndarray, n_boot: int,
              rng) -> tuple[dict[str, tuple[float, float]], np.ndarray | None]:
    """Percentile CIs per selector, plus the paired ``exo - s_pref`` difference draws.

    All selectors are scored on the *same* resampled node set each rep, so the
    difference distribution is paired and the CI is a valid test of "does the
    exogenous signal beat GNN-as-Judge's rule on this dataset".
    """
    n = len(labels)
    names = list(selectors)
    draws = {k: [] for k in names}
    for _ in range(n_boot):
        take = rng.integers(0, n, n)
        lab = labels[take]
        if lab.all() or not lab.any():
            continue
        for k in names:
            draws[k].append(fast_auroc(selectors[k][take], lab))
    cis = {}
    for k in names:
        d = np.array(draws[k], dtype=float)
        d = d[np.isfinite(d)]
        cis[k] = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))) if len(d) else (np.nan, np.nan)
    return cis, {k: np.array(v, dtype=float) for k, v in draws.items()}


def evaluate(name: str, dname: str, spec: dict, logits: dict, sig: dict,
             mask: np.ndarray, num_classes: int, n_boot: int, rng) -> list[dict]:
    """One dataset x direction: both the restricted and unrestricted AUROC tables."""
    p_t = torch.softmax(logits[spec["teacher"]], dim=-1).numpy()
    p_s = torch.softmax(logits[spec["student"]], dim=-1).numpy()
    y = sig["y"]
    t_correct = p_t.argmax(1) == y
    s_correct = p_s.argmax(1) == y
    disagree = p_t.argmax(1) != p_s.argmax(1)

    soft_h = soft_local_homophily(torch.as_tensor(sig["edge_index"]).long(),
                                  torch.as_tensor(p_t if spec["teacher"] == "gnn" else p_s))
    selectors = build_selectors(p_t, p_s, sig, soft_h, num_classes, rng)

    # Drop nodes where any selector is undefined (isolated nodes have NaN homophily)
    # so every selector is scored on an identical node set.
    finite = np.all([np.isfinite(v) for v in selectors.values()], axis=0)

    rows = []
    for regime, keep, labels in (
        ("disagree", mask & disagree & finite, t_correct),
        ("all", mask & finite, t_correct & ~s_correct),
    ):
        n = int(keep.sum())
        lab = labels[keep]
        if n < 20 or lab.all() or not lab.any():
            logger.warning(f"[{name}/{dname}/{regime}] n={n}, pos={int(lab.sum())} -- skipped")
            continue
        sub = {k: v[keep] for k, v in selectors.items()}
        cis, draws = bootstrap(sub, lab, n_boot, rng)
        diff = draws[spec["exo"]] - draws[BASELINE]
        diff = diff[np.isfinite(diff)]
        for k, v in sub.items():
            rows.append({
                "dataset": name, "direction": dname, "regime": regime,
                "selector": k, "family": FAMILY[k],
                "n": n, "base_rate": round(float(lab.mean()), 4),
                "auroc": round(float(fast_auroc(v, lab)), 4),
                "ci_lo": round(cis[k][0], 4), "ci_hi": round(cis[k][1], 4),
                # paired exogenous-minus-s_pref, identical on every row of this block
                "exo_minus_spref": round(float(diff.mean()), 4) if len(diff) else np.nan,
                "exo_minus_spref_lo": round(float(np.percentile(diff, 2.5)), 4) if len(diff) else np.nan,
                "exo_minus_spref_hi": round(float(np.percentile(diff, 97.5)), 4) if len(diff) else np.nan,
            })
    return rows


def plot_summary(df: pd.DataFrame, out_path: Path):
    """Per-selector AUROC across datasets, faceted by direction (restricted regime)."""
    sub = df[df["regime"] == "disagree"]
    dirs = list(sub["direction"].unique())
    fig, axes = plt.subplots(1, len(dirs), figsize=(8 * len(dirs), 5), squeeze=False)
    colors = {"endogenous": "#F44336", "exogenous": "#4CAF50",
              "exogenous (uses nbr labels)": "#4CAF50",
              "exogenous (label-free)": "#2196F3", "baseline": "#9E9E9E"}
    for ax, dname in zip(axes[0], dirs):
        g = sub[sub["direction"] == dname]
        order = g.groupby("selector")["auroc"].mean().sort_values()
        for i, sel in enumerate(order.index):
            vals = g[g["selector"] == sel]["auroc"].values
            fam = g[g["selector"] == sel]["family"].iloc[0]
            ax.scatter(vals, np.full(len(vals), i), alpha=.55, s=28, color=colors[fam])
            ax.scatter([vals.mean()], [i], marker="|", s=400, lw=2.5, color="black")
        ax.axvline(0.5, color="grey", ls="--", lw=1)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order.index)
        ax.set_xlabel("AUROC  (does this signal know the teacher is right?)")
        ax.set_title(f"{dname}   |  disagreement nodes only\n"
                     f"dots = datasets, bar = mean")
        ax.set_xlim(0, 1)
        ax.grid(axis="x", ls="--", alpha=.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


@hydra.main(version_base=None, config_path="../configs", config_name="main")
def main(cfg: DictConfig):
    root = Path(cfg.project_root)
    analysis_root = root / "analysis_output"
    out_dir = analysis_root / "gate_quality"
    split = str(cfg.get("split", "all"))
    n_boot = int(cfg.get("n_boot", 300))
    k = int(cfg.analysis.knn_k)
    names = list(cfg.get("datasets_override", None) or DATASETS)
    rng = np.random.default_rng(int(cfg.seed))

    logger.info(f"split={split}  n_boot={n_boot}  k={k}  datasets={names}")
    rows = []
    for name in names:
        # Prefer the solo logits of the same uid the harm analysis pairs on, so both
        # analyses describe one run; fall back to latest-by-mtime solo logits.
        uid, paired = paired_run_logits(analysis_root, name, uid=cfg.get("run_uid", None))
        if paired is not None:
            logits = {kind: paired[(kind, False)][1] for kind in ("gnn", "llm")}
        else:
            logits = {kind: latest_logits(analysis_root, name, kind, em=False)[1]
                      for kind in ("gnn", "llm")}
            uid = "latest-by-mtime"
        if any(v is None for v in logits.values()):
            logger.warning(f"[{name}] skipped -- missing solo logits")
            continue
        logger.info(f"[{name}] uid={uid}")

        sig = node_signals(cfg, name, k=k)
        mask = split_mask(sig, split)
        for dname, spec in DIRECTIONS.items():
            rows += evaluate(name, dname, spec, logits, sig, mask,
                             sig["num_classes"], n_boot, rng)
        logger.info(f"[{name}] done")

    if not rows:
        logger.error("no dataset had solo logits for both models -- nothing to report")
        return

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f"gate_auroc_{split}.csv"
    df.to_csv(csv, index=False)
    plot_summary(df, out_dir / f"gate_auroc_{split}.png")

    for regime in ("disagree", "all"):
        sub = df[df["regime"] == regime]
        if not len(sub):
            continue
        logger.info(f"\n{'='*82}\nREGIME = {regime}   "
                    f"({'disagreement nodes, label = teacher correct' if regime == 'disagree' else 'all nodes, label = teacher right & student wrong'})\n{'='*82}")
        for dname, g in sub.groupby("direction"):
            # mean AUROC and mean rank across datasets (GLANCE-style rank summary)
            piv = g.pivot_table(index="selector", columns="dataset", values="auroc")
            rank = piv.rank(ascending=False, axis=0).mean(axis=1)
            summary = pd.DataFrame({
                "family": [FAMILY[s] for s in piv.index],
                "mean_auroc": piv.mean(axis=1).round(4),
                "min": piv.min(axis=1).round(4),
                "max": piv.max(axis=1).round(4),
                "mean_rank": rank.round(2),
                "n_datasets": piv.notna().sum(axis=1),
            }).sort_values("mean_auroc", ascending=False)
            logger.info(f"\n--- {dname} ---\n{summary.to_string()}")
            paired = g.drop_duplicates("dataset")[
                ["dataset", "n", "base_rate", "exo_minus_spref",
                 "exo_minus_spref_lo", "exo_minus_spref_hi"]]
            wins = int((paired["exo_minus_spref_lo"] > 0).sum())
            loss = int((paired["exo_minus_spref_hi"] < 0).sum())
            logger.info(f"paired bootstrap {DIRECTIONS[dname]['exo']} - {BASELINE}: "
                        f"{wins} datasets significantly better, {loss} significantly worse, "
                        f"{len(paired) - wins - loss} inconclusive\n{paired.to_string(index=False)}")

    logger.info(f"\nSaved -> {csv}\n        -> {out_dir / f'gate_auroc_{split}.png'}")


if __name__ == "__main__":
    main()
