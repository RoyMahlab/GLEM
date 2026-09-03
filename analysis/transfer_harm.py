"""Does cross-modal distillation *harm* an identifiable population of nodes?

The claim every LLM+GNN co-training paper makes implicitly is that letting the
peer model teach is net-positive. That is a *dataset-level* average. This script
asks the node-level question: **where does the transfer flip a node from right to
wrong, and is that population predictable from an exogenous property of the node?**

For each direction we compare a *pre-transfer* prediction (the solo-trained model)
against a *post-transfer* one (the same model after co-training), per node:

    correction : wrong  -> correct
    corruption : correct -> wrong
    NCS        : (corrections - corruptions) / n      [GLANCE's Net Correction Score]

Each transfer is binned on **two different axes**, which answer two different
questions and are easy to conflate:

  * ``axis=student`` -- where the *student* is out of its inductive bias. For
    ``llm->gnn`` that is low local homophily (structure misleading); for
    ``gnn->llm`` it is high kNN semantic ambiguity (text uninformative). Rising
    NCS here is the case for gating **at all**: transfer pays off where the
    student is weak.
  * ``axis=teacher`` -- where the *teacher* is out of its bias, i.e. the **other**
    signal (``llm->gnn`` is binned by ambiguity, because the LLM teacher is the
    one that fails on ambiguous text). This is the **harm hypothesis**: an
    unreliable teacher should corrupt nodes the student already had right.

The 2x2 ``quadrant_frame`` crosses the two, isolating the cell the harm
hypothesis actually predicts -- **teacher weak, student fine**.

A per-bin exact McNemar test (binomial on the discordant pairs) says whether a
bin's net effect is real rather than churn. **If NCS is flat across bins, the
gating premise has no support and there is no paper** -- that negative result is
the point of running this before any new training. Note that running this on a
*gated* EM run cannot establish the counterfactual: absence of harm is then the
gate working, not evidence that uniform distillation is safe. For that, compare
against a uniform run (``em.tau_gnn=0 em.tau_knn=0``).

Run::

    python -m analysis.transfer_harm                       # every dataset, all nodes
    python -m analysis.transfer_harm +split=test +nbins=6
    python -m analysis.transfer_harm '+datasets_override=[cora,cornell]'

Outputs ``analysis_output/transfer_harm/`` -- ``harm_bins_<split>.csv`` (every bin),
``harm_quadrants_<split>.csv``, ``<name>_harm.png`` per dataset, ``pooled_harm.png``.
"""
from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig
from scipy.stats import binomtest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.transfer_common import (  # noqa: E402
    DATASETS,
    node_signals,
    paired_run_logits,
    quantile_bins,
    split_mask,
)

# Each transfer has TWO relevant difficulty axes, and they answer different
# questions. Binning by the *student's* axis asks "does the transfer help most
# where the student is weak?" -- the case for gating at all. Binning by the
# *teacher's* axis asks "does the transfer hurt where the teacher is weak?" --
# the harm hypothesis. They are different signals and must not be conflated:
# for llm->gnn the student (GNN) is weak on heterophilous nodes while the teacher
# (LLM) is weak on semantically ambiguous ones.
SIGNALS = {
    "neg_homophily": "1 - local homophily  (structure less reliable ->)",
    "ambiguity": "kNN semantic ambiguity  (text less reliable ->)",
}
TRANSFERS = {
    "llm->gnn": dict(student="gnn", teacher="llm",
                     student_signal="neg_homophily", teacher_signal="ambiguity"),
    "gnn->llm": dict(student="llm", teacher="gnn",
                     student_signal="ambiguity", teacher_signal="neg_homophily"),
}
# (transfer, axis) pairs actually plotted/tabulated.
AXES = ("student", "teacher")


def signal_values(sig: dict, which: str) -> np.ndarray:
    return (1.0 - sig["homophily"]) if which == "neg_homophily" else sig["ambiguity"]


def _cell(pre, post, tch, sel) -> dict:
    """Correction / corruption / NCS / exact-McNemar for one node subset."""
    corrections = int((~pre[sel] & post[sel]).sum())
    corruptions = int((pre[sel] & ~post[sel]).sum())
    discordant = corrections + corruptions
    # Exact McNemar: under "the transfer is neutral here", each discordant node is
    # a fair coin. Two-sided, so a cell can be flagged as helping or hurting.
    p = binomtest(corrections, discordant, 0.5).pvalue if discordant else float("nan")
    n = int(sel.sum())
    return {
        "n": n, "corrections": corrections, "corruptions": corruptions,
        "ncs": round((corrections - corruptions) / n, 4) if n else np.nan,
        "corruption_rate": round(corruptions / n, 4) if n else np.nan,
        "acc_before": round(float(pre[sel].mean()), 4) if n else np.nan,
        "acc_after": round(float(post[sel].mean()), 4) if n else np.nan,
        "teacher_acc": round(float(tch[sel].mean()), 4) if n else np.nan,
        "mcnemar_p": float(p),
    }


def direction_frame(name: str, tname: str, spec: dict, axis: str, sig: dict,
                    mask: np.ndarray, before, after, teacher,
                    nbins: int) -> pd.DataFrame | None:
    """Per-bin correction/corruption table, binned on the student- or teacher-side axis."""
    y = sig["y"]
    which = spec[f"{axis}_signal"]
    values = signal_values(sig, which)

    keep = mask & np.isfinite(values)
    if keep.sum() < 2 * nbins:
        logger.warning(f"[{name}/{tname}/{axis}] too few nodes ({keep.sum()}) -- skipped")
        return None

    pre = before.argmax(1).numpy() == y
    post = after.argmax(1).numpy() == y
    tch = teacher.argmax(1).numpy() == y

    bin_idx, edges = quantile_bins(np.where(keep, values, np.nan), nbins)
    rows = []
    for b in range(len(edges) - 1):
        sel = keep & (bin_idx == b)
        if not sel.any():
            continue
        rows.append({
            "dataset": name, "direction": tname, "axis": axis, "signal": which,
            "bin": b,
            "bin_lo": round(float(edges[b]), 4), "bin_hi": round(float(edges[b + 1]), 4),
            # quantile rank of the bin centre, so bins are comparable across
            # datasets whose signal distributions differ -- used by the pooled plot
            "q": round((b + 0.5) / (len(edges) - 1), 4),
            **_cell(pre, post, tch, sel),
        })
    return pd.DataFrame(rows)


def quadrant_frame(name: str, tname: str, spec: dict, sig: dict, mask: np.ndarray,
                   before, after, teacher) -> pd.DataFrame | None:
    """2x2 split on (teacher weak?) x (student weak?), each signal at its median.

    This is the cell the harm hypothesis actually predicts: **teacher weak while
    the student is fine**. If uniform distillation is dangerous anywhere it is
    there -- an unreliable teacher overwriting a student that had the node right.
    """
    y = sig["y"]
    t_val = signal_values(sig, spec["teacher_signal"])
    s_val = signal_values(sig, spec["student_signal"])
    keep = mask & np.isfinite(t_val) & np.isfinite(s_val)
    if keep.sum() < 40:
        return None

    pre = before.argmax(1).numpy() == y
    post = after.argmax(1).numpy() == y
    tch = teacher.argmax(1).numpy() == y
    t_weak = t_val > np.median(t_val[keep])
    s_weak = s_val > np.median(s_val[keep])

    rows = []
    for tw in (False, True):
        for sw in (False, True):
            sel = keep & (t_weak == tw) & (s_weak == sw)
            if not sel.any():
                continue
            rows.append({
                "dataset": name, "direction": tname,
                "teacher_weak": tw, "student_weak": sw,
                **_cell(pre, post, tch, sel),
            })
    return pd.DataFrame(rows)


def plot_dataset(name: str, frames: dict[tuple[str, str], pd.DataFrame], out_path: Path):
    """Correction/corruption bars + NCS line: one panel per (transfer, axis)."""
    fig, axes = plt.subplots(1, len(frames), figsize=(7 * len(frames), 5), squeeze=False)
    for ax, ((tname, axis), df) in zip(axes[0], frames.items()):
        spec = TRANSFERS[tname]
        x = np.arange(len(df))
        ax.bar(x, df["corrections"] / df["n"], color="#4CAF50", alpha=.8, label="corrected (wrong->right)")
        ax.bar(x, -df["corruptions"] / df["n"], color="#F44336", alpha=.8, label="corrupted (right->wrong)")
        ax.plot(x, df["ncs"], "o-", color="black", lw=2, label="net correction score")
        ax.axhline(0, color="grey", lw=1)

        ax2 = ax.twinx()
        ax2.plot(x, df["teacher_acc"], "s--", color="#2196F3", alpha=.7, label="teacher acc")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("teacher accuracy in bin", color="#2196F3")

        ax.set_xticks(x)
        ax.set_xticklabels([f"{lo:.2f}\n{hi:.2f}" for lo, hi in zip(df["bin_lo"], df["bin_hi"])],
                           fontsize=8)
        ax.set_xlabel(SIGNALS[spec[f"{axis}_signal"]])
        ax.set_ylabel("fraction of bin")
        ax.set_title(f"{name}  |  {spec['teacher'].upper()} teaches {spec['student'].upper()}"
                     f"\nbinned by {axis} weakness"
                     + ("  (harm hypothesis)" if axis == "teacher" else ""))
        for i, (n, p) in enumerate(zip(df["n"], df["mcnemar_p"])):
            star = "*" if np.isfinite(p) and p < 0.05 else ""
            ax.text(i, ax.get_ylim()[1] * 0.92, f"n={n}{star}", ha="center", fontsize=7, color="grey")
        ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_pooled(all_df: pd.DataFrame, out_path: Path):
    """NCS vs within-dataset quantile rank, one line per dataset, faceted by direction.

    Pooling on the *quantile rank* rather than the raw signal is what makes an
    11-dataset overlay meaningful -- ambiguity 0.5 means something different on
    pubmed (3 classes) than on bookchild (24).
    """
    panels = list(all_df.groupby(["direction", "axis"]).groups)
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5), squeeze=False)
    for ax, (dname, axis) in zip(axes[0], panels):
        sub = all_df[(all_df["direction"] == dname) & (all_df["axis"] == axis)]
        for ds, g in sub.groupby("dataset"):
            ax.plot(g["q"], g["ncs"], "o-", alpha=.6, lw=1.2, label=ds)
        # weighted mean NCS per quantile decile across datasets
        binned = sub.groupby(pd.cut(sub["q"], np.linspace(0, 1, 9)), observed=True)
        agg = binned.apply(lambda g: np.average(g["ncs"], weights=g["n"]) if len(g) else np.nan)
        centres = [iv.mid for iv in agg.index]
        ax.plot(centres, agg.values, "k-", lw=3, label="pooled (n-weighted)")
        ax.axhline(0, color="grey", lw=1)
        ax.set_xlabel(f"within-dataset quantile of {axis} weakness")
        ax.set_ylabel("net correction score")
        ax.set_title(f"NCS  |  {dname}  |  binned by {axis} weakness"
                     + ("  (harm hypothesis)" if axis == "teacher" else ""))
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


@hydra.main(version_base=None, config_path="../configs", config_name="main")
def main(cfg: DictConfig):
    root = Path(cfg.project_root)
    analysis_root = root / "analysis_output"
    out_dir = analysis_root / "transfer_harm"
    split = str(cfg.get("split", "all"))
    nbins = int(cfg.get("nbins", 8))
    k = int(cfg.analysis.knn_k)
    names = list(cfg.get("datasets_override", None) or DATASETS)

    logger.info(f"split={split}  nbins={nbins}  k={k}  datasets={names}")
    all_rows, quad_rows = [], []
    for name in names:
        # Same-uid pairing, not latest-by-mtime: before/after must come from one
        # pipeline invocation or the comparison straddles configs (see docstring
        # of paired_run_logits).
        uid, paired = paired_run_logits(analysis_root, name, uid=cfg.get("run_uid", None))
        if paired is None:
            logger.warning(f"[{name}] skipped -- no uid dir holds both solo and EM logits "
                           f"for both models")
            continue
        logits = {kk: t for kk, (_, t) in paired.items()}

        sig = node_signals(cfg, name, k=k)
        y = sig["y"]
        accs = {f"{'em' if e else 'solo'}-{kd}":
                round(float((logits[(kd, e)].argmax(1).numpy() == y)[sig["test_mask"].astype(bool)].mean()), 4)
                for kd in ("gnn", "llm") for e in (False, True)}
        logger.info(f"[{name}] uid={uid}  test acc {accs}")
        mask = split_mask(sig, split)
        frames = {}
        for tname, spec in TRANSFERS.items():
            kw = dict(before=logits[(spec["student"], False)],
                      after=logits[(spec["student"], True)],
                      teacher=logits[(spec["teacher"], False)])
            for axis in AXES:
                df = direction_frame(name, tname, spec, axis, sig, mask,
                                     nbins=nbins, **kw)
                if df is not None and len(df):
                    frames[(tname, axis)] = df
                    all_rows.append(df)
            q = quadrant_frame(name, tname, spec, sig, mask, **kw)
            if q is not None:
                quad_rows.append(q)
        if frames:
            plot_dataset(name, frames, out_dir / f"{name}_harm.png")
            logger.info(f"[{name}] done -> {out_dir / f'{name}_harm.png'}")

    if not all_rows:
        logger.error("no dataset had both solo and EM logits -- nothing to report")
        return

    all_df = pd.concat(all_rows, ignore_index=True)
    quad_df = pd.concat(quad_rows, ignore_index=True) if quad_rows else pd.DataFrame()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f"harm_bins_{split}.csv"
    all_df.to_csv(csv, index=False)
    quad_csv = out_dir / f"harm_quadrants_{split}.csv"
    if len(quad_df):
        quad_df.to_csv(quad_csv, index=False)
    plot_pooled(all_df, out_dir / f"pooled_harm_{split}.png")

    logger.info(f"\n{'='*90}\nNCS by bin, n-weighted over datasets\n"
                f"  axis=student -> 'does transfer help where the STUDENT is weak?' (case for gating)\n"
                f"  axis=teacher -> 'does transfer hurt where the TEACHER is weak?' (harm hypothesis)\n"
                f"{'='*90}")
    for (dname, axis), g in all_df.groupby(["direction", "axis"]):
        lo, hi = g[g["q"] <= 0.5], g[g["q"] > 0.5]
        harm = (g["mcnemar_p"] < 0.05) & (g["ncs"] < 0)
        logger.info(
            f"{dname:9s} axis={axis:7s} signal={g['signal'].iloc[0]:14s} "
            f"overall NCS={np.average(g['ncs'], weights=g['n']):+.4f}  "
            f"low={np.average(lo['ncs'], weights=lo['n']):+.4f}  "
            f"high={np.average(hi['ncs'], weights=hi['n']):+.4f}  "
            f"harmful bins (p<.05, NCS<0): {int(harm.sum())}/{len(g)}"
        )

    if len(quad_df):
        logger.info(f"\n{'='*90}\nQuadrants: the harm hypothesis predicts NCS < 0 in "
                    f"teacher_weak=True, student_weak=False\n{'='*90}")
        for dname, g in quad_df.groupby("direction"):
            agg = g.groupby(["teacher_weak", "student_weak"]).apply(
                lambda h: pd.Series({
                    "n": int(h["n"].sum()),
                    "ncs": round(float(np.average(h["ncs"], weights=h["n"])), 4),
                    "corruption_rate": round(float(np.average(h["corruption_rate"], weights=h["n"])), 4),
                    "teacher_acc": round(float(np.average(h["teacher_acc"], weights=h["n"])), 4),
                    "datasets_NCS<0": int((h["ncs"] < 0).sum()),
                    "harmful (p<.05)": int(((h["mcnemar_p"] < 0.05) & (h["ncs"] < 0)).sum()),
                }), include_groups=False)
            logger.info(f"\n--- {dname} ---\n{agg.to_string()}")

    logger.info(f"\nSaved -> {csv}\n        -> {quad_csv}"
                f"\n        -> {out_dir / f'pooled_harm_{split}.png'}")


if __name__ == "__main__":
    main()
