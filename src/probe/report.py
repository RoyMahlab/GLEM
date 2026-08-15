"""Apply the EXPERIMENT.md section 11 decision rule, and draw the figures.

This is deliberately separate from ``analyze.py``: that script computes per-bin
statistics and knows nothing about verdicts, this one only reads its CSVs and
applies the preregistered rule. Nothing here chooses a bin, a threshold or an arm.

The rule (§11), evaluated per (dataset, direction):

* **Supported** -- NCS significantly negative (McNemar p<0.05, sign stable across
  all seeds) in the teacher-out-of-bias bins while positive elsewhere, AND absent
  in the alpha=beta=0 control.
* **Weakly supported** -- NCS positive everywhere but significantly lower in the
  teacher-out-of-bias bins, with teacher accuracy degrading across the axis.
* **Not supported** -- NCS flat across the axis, OR the same pattern appears in the
  control.

Two §5/§10 gates are applied before the rule, not after: bins flagged ``n < 30`` do
not contribute, and a bin whose NCS sign is not stable across all seeds is reported
as unstable rather than as a result.

Run::

    .venv/bin/python src/probe/report.py --in temp/probe_analysis --out temp/probe_analysis
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Which median-split bin is the teacher-out-of-bias side, per direction. Mirrors
# analyze.py rather than being re-derived from the signal name.
OUT_OF_BIAS = {'gnn->lm': 'low',   # GNN teacher fails at LOW local homophily
               'lm->gnn': 'high'}  # LM  teacher fails at HIGH kNN ambiguity
CONTROL_ARMS = ('alpha0_li_T', 'alpha0_li_F')


def mcnemar_pooled(corrections, corruptions):
    """Exact McNemar on summed discordant counts.

    §10: descriptive only. Pooling reuses the same nodes across seeds and
    iterations, so the p-value is anti-conservative.
    """
    from probe.analyze import mcnemar_exact
    return mcnemar_exact(int(corrections), int(corruptions))


def sign_stability(sub):
    """Fraction of seeds sharing the modal NCS sign, and whether all agree."""
    per_seed = sub.groupby('seed')['ncs'].mean()
    signs = np.sign(per_seed.values)
    signs = signs[signs != 0]
    if len(signs) == 0:
        return 0.0, False, per_seed
    frac = max((signs > 0).mean(), (signs < 0).mean())
    return float(frac), bool(abs(signs.sum()) == len(signs)), per_seed


def evaluate(d, scheme='median'):
    """One verdict row per (dataset, direction), plus the evidence behind it.

    ``scheme`` selects the binning the rule is applied to. It defaults to
    ``'median'``, the only scheme §5 preregistered as primary, and ``main()`` never
    passes anything else -- so ``verdicts.csv`` is always the preregistered result.
    Other values are for the exploratory companions only (A11): a verdict computed
    on a non-preregistered binning is not a verdict, and must be labelled as such
    wherever it is shown.
    """
    rows = []
    gated = d[(d.axis == 'teacher') & (d.bin_scheme == scheme) & (~d.low_n_flag)]

    # Emit an explicit 'no test' row for every (dataset, direction) that exists in
    # the data but has no rows left after the §5/§10 gates. Grouping only over
    # survivors would drop those combinations from the table entirely, and a
    # direction that could not be tested must be visible as untested rather than
    # silently absent -- it is the difference between "no harm" and "no measurement".
    universe = d[d.axis == 'teacher'].groupby(['dataset', 'direction']).size().index
    survivors = set(gated.groupby(['dataset', 'direction']).size().index)
    for ds, direction in universe:
        if (ds, direction) in survivors:
            continue
        raw = d[(d.axis == 'teacher') & (d.dataset == ds) & (d.direction == direction)
                & (d.bin_scheme == scheme)]
        if raw.empty:
            why = (f'{scheme} split degenerate: one bin empty, so there is no contrast '
                   f'to test (see amendment A6)')
        else:
            why = (f'every {scheme} bin flagged n<30 (max n={int(raw.n.max())}); '
                   f'excluded from the verdict by §5')
        rows.append({'dataset': ds, 'direction': direction,
                     'oob_bin': OUT_OF_BIAS[direction], 'verdict': 'no test',
                     'reason': why,
                     'published_tacc_oob': np.nan, 'published_tacc_in': np.nan})

    for (ds, direction), grp in gated.groupby(['dataset', 'direction']):
        oob = OUT_OF_BIAS[direction]
        rec = {'dataset': ds, 'direction': direction, 'oob_bin': oob}
        for arm_label, arm_sel in (('published', grp[grp.arm == 'published']),
                                   ('control', grp[grp.arm.isin(CONTROL_ARMS)])):
            a = arm_sel[arm_sel['bin'] == oob]
            b = arm_sel[arm_sel['bin'] != oob]
            if len(a) == 0 or len(b) == 0:
                rec[f'{arm_label}_status'] = 'no contrast (a bin is absent or all n<30)'
                continue
            f_oob, stable_oob, per_seed = sign_stability(a)
            rec.update({
                # n per *distillation event*, which is the real unit: a node in this
                # bin is measured once per (seed, iteration). Summing those gives
                # node-observations, not nodes, and reads as far more independent
                # evidence than exists -- so the summed figure is kept but named
                # honestly, and the per-step range is what the table shows.
                f'{arm_label}_n_per_step': float(a.n.mean()),
                f'{arm_label}_n_per_step_min': int(a.n.min()),
                f'{arm_label}_n_per_step_max': int(a.n.max()),
                f'{arm_label}_n_obs': int(a.n.sum()),
                f'{arm_label}_n_in_per_step': float(b.n.mean()),
                f'{arm_label}_ncs_oob': per_seed.mean(),
                f'{arm_label}_ncs_in': b.groupby('seed').ncs.mean().mean(),
                f'{arm_label}_ncs_gap': per_seed.mean() - b.groupby('seed').ncs.mean().mean(),
                # §10 asks for across-SEED variance. Averaging within a seed first
                # keeps iteration-to-iteration wobble out of the seed spread. NaN at
                # one seed, which is the honest reading rather than a spurious 0.
                f'{arm_label}_ncs_oob_sd': per_seed.std(),
                f'{arm_label}_iters': int(a.iteration.nunique()),
                f'{arm_label}_seeds': len(per_seed),
                f'{arm_label}_sign_frac': f_oob,
                f'{arm_label}_sign_stable': stable_oob,
                f'{arm_label}_p_oob': mcnemar_pooled(a.corrections.sum(), a.corruptions.sum()),
                f'{arm_label}_tacc_oob': a.teacher_acc.mean(),
                f'{arm_label}_tacc_in': b.teacher_acc.mean(),
                f'{arm_label}_status': 'ok',
            })

        # Per-seed stability of the GAP itself. `{arm}_sign_stable` above is the
        # stability of NCS *within the out-of-bias bin*, a different quantity: on
        # arxiv the control's bin NCS is stably negative while its gap flips sign
        # twice. §11's disqualifier is about the gap, so it needs this.
        for arm_label, arm_sel in (('published', grp[grp.arm == 'published']),
                                   ('control', grp[grp.arm.isin(CONTROL_ARMS)])):
            aa = arm_sel[arm_sel['bin'] == oob]
            bb = arm_sel[arm_sel['bin'] != oob]
            if len(aa) == 0 or len(bb) == 0:
                continue
            per_seed_gap = (aa.groupby('seed').ncs.mean()
                            - bb.groupby('seed').ncs.mean()).dropna()
            rec[f'{arm_label}_gap_sd'] = float(per_seed_gap.std())
            sg = np.sign(per_seed_gap.values)
            sg = sg[sg != 0]
            rec[f'{arm_label}_gap_sign_stable'] = bool(
                len(sg) > 1 and abs(sg.sum()) == len(sg))

        rec['verdict'], rec['reason'] = _verdict(rec)
        rows.append(rec)
    return pd.DataFrame(rows)


def _verdict(r):
    """Apply §11 literally to one (dataset, direction) record."""
    if r.get('published_status') != 'ok':
        return 'no test', r.get('published_status', 'no published rows survived the gates')

    ncs_oob, gap = r['published_ncs_oob'], r['published_ncs_gap']
    stable, p = r['published_sign_stable'], r['published_p_oob']
    tacc_degrades = r['published_tacc_oob'] < r['published_tacc_in']

    # §10 requires the NCS sign to be stable *across seeds*. With a single seed
    # that test is vacuous -- one sign trivially agrees with itself -- so a
    # single-seed cell must not be able to reach Supported or Weakly supported no
    # matter how large its effect or how small its p-value. arxiv is deliberately
    # single-seed (A7), so without this guard it would report a verdict its own
    # amendment says it cannot support.
    if r.get('published_seeds', 0) < 2:
        return 'no test', (f"only {r.get('published_seeds', 0)} seed: §10 across-seed sign "
                           f"stability cannot be evaluated (see amendment A7). "
                           f"Descriptively: NCS {ncs_oob:+.4f} out-of-bias vs "
                           f"{r['published_ncs_in']:+.4f} in-bias, gap {gap:+.4f}, "
                           f"teacher acc {r['published_tacc_in']:.3f}->{r['published_tacc_oob']:.3f}")

    # §11's disqualifier needs the control to exist. Absent it, the pattern cannot
    # be attributed to the pseudo-label term rather than to retraining churn.
    if r.get('control_status') != 'ok':
        return 'no test', ('no alpha=beta=0 control available for this cell, so §11\'s '
                           'control clause cannot be evaluated')

    # §11's disqualifier: "the same pattern appears in the alpha=beta=0 control".
    # §11 never defined "same pattern" quantitatively and the two readings diverge on
    # arxiv, so the choice is stated rather than left implicit (amendment A17).
    #
    # Matching the SIGN of the mean gap alone would let an unstable control veto a
    # stable published effect: arxiv's control gaps are +0.0001 / +0.0103 / -0.0169 --
    # scatter about zero whose mean happens to land negative -- against a published
    # gap of -0.0089 / -0.0121 / -0.0114. A rule discarding a result on that basis
    # would reject almost any true effect, and §10 already makes across-seed sign
    # stability this study's standard for "real". So the control must reproduce the
    # pattern *stably* to disqualify.
    ctrl_same = (r.get('control_status') == 'ok'
                 and np.sign(r.get('control_ncs_gap', np.nan)) == np.sign(gap)
                 and bool(r.get('control_gap_sign_stable', False)))

    if ncs_oob < 0 and stable and p < 0.05 and r['published_ncs_in'] > 0:
        if ctrl_same:
            return 'not supported', ('negative NCS in the out-of-bias bin, but the '
                                     'control reproduces the same pattern')
        return 'supported', 'NCS significantly negative out-of-bias, positive elsewhere, absent in control'

    if r['published_ncs_oob'] > 0 and r['published_ncs_in'] > 0 and gap < 0 and stable and tacc_degrades:
        # §11's control clause is a general disqualifier ("... OR the same pattern
        # appears in the control"), not one attached only to Supported. It was
        # previously consulted for Supported alone -- an implementation gap against
        # the preregistered text, fixed here.
        if ctrl_same:
            return 'not supported', ('NCS lower out-of-bias, but the control '
                                     'reproduces the same gap stably')
        return 'weakly supported', 'NCS positive throughout but lower out-of-bias, teacher accuracy degrading'

    if not stable:
        return 'not supported', (f"NCS sign unstable across seeds "
                                 f"({r['published_sign_frac']:.0%} agreement, "
                                 f"n={r['published_seeds']} seeds)")
    if ctrl_same:
        return 'not supported', 'the same pattern appears in the alpha=beta=0 control'
    return 'not supported', f'NCS flat across the axis (gap {gap:+.4f}, p={p:.3f})'


# ───────────────────────────────────────────── figures


def per_dataset_figure(d, out):
    """Deliverable 4: correction/corruption bars + NCS line vs the teacher axis,
    teacher accuracy on a secondary axis, one panel per direction."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t = d[(d.axis == 'teacher') & (d.arm == 'published')]
    for ds, grp in t.groupby('dataset'):
        dirs = sorted(grp.direction.unique())
        if not dirs:
            continue
        fig, axes = plt.subplots(1, len(dirs), figsize=(6.2 * len(dirs), 4.2), squeeze=False)
        for ax, direction in zip(axes[0], dirs):
            # Prefer the quantile scheme for the axis plot: it shows the trend, and
            # is the reported scheme wherever the median split degenerated (A6).
            sub = grp[(grp.direction == direction) & (grp.bin_scheme == 'q5')]
            if sub.empty:
                sub = grp[(grp.direction == direction) & (grp.bin_scheme == 'median')]
            agg = sub.groupby('bin').agg(
                corrections=('corrections', 'sum'), corruptions=('corruptions', 'sum'),
                ncs=('ncs', 'mean'), tacc=('teacher_acc', 'mean'), n=('n', 'sum')).reset_index()
            if agg.empty:
                ax.set_axis_off()
                continue
            x = np.arange(len(agg))
            ax.bar(x - 0.2, agg.corrections, 0.4, label='corrections', color='#4C78A8')
            ax.bar(x + 0.2, agg.corruptions, 0.4, label='corruptions', color='#E45756')
            ax.set_xticks(x)
            ax.set_xticklabels([f'{b}\nn={int(n)}' for b, n in zip(agg['bin'], agg.n)], fontsize=8)
            ax.set_ylabel('nodes flipped')
            sig = sub.signal.iloc[0]
            ax.set_title(f'{ds}  {direction}\nbinned by teacher-side {sig}', fontsize=9)
            ax2 = ax.twinx()
            ax2.plot(x, agg.ncs, 'o-', color='#333', label='NCS')
            ax2.plot(x, agg.tacc, 's--', color='#54A24B', label='teacher acc')
            ax2.axhline(0, color='#999', lw=0.8, ls=':')
            ax2.set_ylabel('NCS / teacher accuracy')
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=7, loc='best')
        fig.tight_layout()
        fig.savefig(out / f'fig_{ds}.png', dpi=140)
        plt.close(fig)


def pooled_figure(d, out):
    """Deliverable 5: NCS vs within-dataset signal quantile, so datasets with
    different signal distributions are comparable."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t = d[(d.axis == 'teacher') & (d.bin_scheme == 'q5')]
    dirs = sorted(t.direction.unique())
    if not dirs:
        return
    fig, axes = plt.subplots(1, len(dirs), figsize=(6.2 * len(dirs), 4.2), squeeze=False)
    for ax, direction in zip(axes[0], dirs):
        sub = t[t.direction == direction]
        for arm, style in (('published', '-'), ('alpha0_li_T', '--'), ('alpha0_li_F', ':')):
            a = sub[sub.arm == arm]
            if a.empty:
                continue
            for ds, g in a.groupby('dataset'):
                g = g.groupby('bin').ncs.mean().reset_index().sort_values('bin')
                ax.plot(np.arange(len(g)), g.ncs, style, marker='o', ms=3, alpha=0.75,
                        label=f'{ds.split("_")[0]} {arm}')
        ax.axhline(0, color='#333', lw=1)
        ax.set_xlabel('within-dataset quantile of the teacher-side signal (low -> high)')
        ax.set_ylabel('NCS')
        ax.set_title(f'{direction}: NCS vs teacher-side signal quantile', fontsize=9)
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out / 'fig_pooled_quantile.png', dpi=140)
    plt.close(fig)


def per_iteration_figure(d, out):
    """Deliverable 6: does corruption accumulate over EM iterations, or get
    repaired later? NCS per iteration for the worst teacher-out-of-bias bin."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t = d[(d.axis == 'teacher') & (d.bin_scheme == 'median') & (~d.low_n_flag)].copy()
    t = t[t.apply(lambda r: r['bin'] == OUT_OF_BIAS[r['direction']], axis=1)]
    if t.empty:
        return
    dirs = sorted(t.direction.unique())
    fig, axes = plt.subplots(1, len(dirs), figsize=(6.2 * len(dirs), 4.0), squeeze=False)
    for ax, direction in zip(axes[0], dirs):
        sub = t[t.direction == direction]
        for (ds, arm), g in sub.groupby(['dataset', 'arm']):
            g = g.groupby('iteration').ncs.agg(['mean', 'std']).reset_index()
            ax.errorbar(g.iteration, g['mean'], yerr=g['std'], marker='o', ms=4, capsize=3,
                        alpha=0.8, ls='-' if arm == 'published' else '--',
                        label=f'{ds.split("_")[0]} {arm}')
        ax.axhline(0, color='#333', lw=1)
        ax.set_xlabel('EM iteration')
        ax.set_ylabel('NCS in teacher-out-of-bias bin')
        ax.set_title(f'{direction}: accumulation across EM iterations', fontsize=9)
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out / 'fig_per_iteration.png', dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    inp, out = Path(a.inp), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(inp / 'ncs_long.csv')

    v = evaluate(d)
    v.to_csv(out / 'verdicts.csv', index=False)

    cols = ['dataset', 'direction', 'oob_bin', 'published_n_per_step', 'published_seeds',
            'published_iters', 'published_ncs_oob', 'published_ncs_oob_sd',
            'published_ncs_in', 'published_ncs_gap', 'published_sign_stable',
            'published_p_oob', 'published_tacc_oob', 'published_tacc_in',
            'control_ncs_gap', 'verdict', 'reason']
    show = v.reindex(columns=[c for c in cols if c in v.columns])
    pd.set_option('display.width', 250)
    print('=== section 11 verdicts, per (dataset, direction) ===')
    print(show.to_string(index=False))
    print()
    print('=== verdict counts ===')
    print(v.verdict.value_counts().to_string())

    per_dataset_figure(d, out)
    pooled_figure(d, out)
    per_iteration_figure(d, out)
    print(f'\nwrote {out}/verdicts.csv and figures')


if __name__ == '__main__':
    main()
