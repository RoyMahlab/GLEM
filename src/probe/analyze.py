"""Turn the per-step archive into the NCS tables of EXPERIMENT.md.

For every distillation event in ``steps.jsonl`` this reconstructs the triple the
hypothesis needs -- student before, student after, teacher as consumed -- from the
archived snapshots, then bins the analysed nodes on the **teacher-side** axis and
tests each bin.

The before/after chain: the archive is keyed by iteration, and each side writes
exactly once per iteration, so a student's pre-step logits are its own snapshot
from the previous iteration (``iter<i-1>``, falling back to the pretrained
``iter-1`` baseline at the first iteration). The teacher is read from the copy made
at step time, never re-resolved from a path -- the live paths are overwritten as
the run proceeds, and ``direction``/``teacher``/``student`` are read from the step
record rather than re-derived, so the teacher/student axis cannot get swapped here.

Run::

    .venv/bin/python src/probe/analyze.py --probe-dir <dir> --out <dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np

MIN_BIN_N = 30  # EXPERIMENT.md section 5: smaller bins are flagged and excluded


def mcnemar_exact(corrections, corruptions):
    """Two-sided exact McNemar: binomial on the discordant pairs, p = 0.5.

    Concordant nodes (correct->correct, wrong->wrong) carry no information about
    direction and are excluded by construction. Returns NaN when nothing changed.
    """
    n = int(corrections) + int(corruptions)
    if n == 0:
        return float('nan')
    try:
        from scipy.stats import binomtest
        return float(binomtest(int(corrections), n, 0.5, alternative='two-sided').pvalue)
    except ImportError:
        from scipy.stats import binom_test
        return float(binom_test(int(corrections), n, 0.5, alternative='two-sided'))


def _acc(logits, labels, idx):
    if len(idx) == 0:
        return float('nan')
    return float((logits[idx].argmax(1) == labels[idx]).mean())


def bin_stats(before, after, teacher, labels, idx):
    """Correction/corruption counts and accuracies for one bin of node ids."""
    n = len(idx)
    if n == 0:
        return None
    ok_b = before[idx].argmax(1) == labels[idx]
    ok_a = after[idx].argmax(1) == labels[idx]
    corrections = int((~ok_b & ok_a).sum())
    corruptions = int((ok_b & ~ok_a).sum())
    return {
        'n': n,
        'corrections': corrections,
        'corruptions': corruptions,
        'ncs': (corrections - corruptions) / n,
        'corruption_rate': corruptions / n,
        'student_acc_before': float(ok_b.mean()),
        'student_acc_after': float(ok_a.mean()),
        'teacher_acc': _acc(teacher, labels, idx),
        'mcnemar_p': mcnemar_exact(corrections, corruptions),
        'low_n_flag': n < MIN_BIN_N,
    }


def median_bins(values, idx):
    """``{'low': ids, 'high': ids}`` by median split over the analysed population.

    The median is taken over the nodes actually analysed rather than over all
    nodes, so the two cells stay balanced in the population being tested.
    """
    v = values[idx]
    finite = np.isfinite(v)
    idx, v = idx[finite], v[finite]
    if len(idx) == 0:
        return {}
    med = float(np.median(v))
    return {'low': idx[v <= med], 'high': idx[v > med]}


def quantile_bins(values, idx, nbins=5):
    """Quantile bins tolerant of heavy ties (local homophily is very discrete).

    Duplicate quantile edges are dropped, so the realised bin count may be below
    ``nbins``; the label records the realised edges.
    """
    v = values[idx]
    finite = np.isfinite(v)
    idx, v = idx[finite], v[finite]
    if len(idx) == 0:
        return {}
    edges = np.unique(np.quantile(v, np.linspace(0, 1, nbins + 1)))
    if len(edges) < 2:
        return {'q0': idx}
    b = np.clip(np.digitize(v, edges[1:-1], right=False), 0, len(edges) - 2)
    return {f'q{i}': idx[b == i] for i in range(len(edges) - 1)}


def _load(p):
    return np.load(p).astype(np.float32)


def analyze_run(run_dir, cache_dir):
    """Long-form rows and quadrant rows for one ``(dataset, regime, arm, seed)``."""
    from probe import signals as sig_mod

    run_dir = Path(run_dir)
    steps_f = run_dir / 'steps.jsonl'
    if not steps_f.exists():
        return [], [], f'{run_dir}: no steps.jsonl'
    steps = [json.loads(l) for l in steps_f.read_text().splitlines() if l.strip()]
    if not steps:
        return [], [], f'{run_dir}: steps.jsonl empty'

    meta = steps[0]
    sig = sig_mod.signals_for_run(run_dir, meta['dataset'], meta['regime'],
                                  meta['seed'], cache_dir)
    labels = sig['labels']
    # Analysis population: ground truth available. Held to val u test in every
    # regime, so few-shot runs stay comparable to standard ones even though their
    # dropped train nodes also receive pseudo-labels (EXPERIMENT.md section 14).
    gt_pool = np.union1d(sig['valid_x'], sig['test_x'])

    rows, quads, notes = [], [], []
    for st in steps:
        kind = 'lm' if st['student'] == 'LM' else 'gnn'
        i = st['em_iter']
        after_f = run_dir / 'logits' / f'iter{i}_{kind}.npy'
        before_f = run_dir / 'logits' / f'iter{i - 1}_{kind}.npy'
        teacher_f = run_dir / 'teacher' / f"step{st['step_index']}_{st['em_phase']}.npy"
        pl_f = run_dir / 'plnodes' / f"step{st['step_index']}_{st['em_phase']}.npy"
        missing = [str(f.name) for f in (after_f, before_f, teacher_f, pl_f) if not f.exists()]
        if missing:
            notes.append(f"{run_dir} step{st['step_index']}: missing {missing}")
            continue

        before, after, teacher = _load(before_f), _load(after_f), _load(teacher_f)
        pl_nodes = np.load(pl_f)
        idx = np.intersect1d(pl_nodes, gt_pool)

        # Teacher-side axis. E-step (GNN teaches) -> local homophily; M-step (LM
        # teaches) -> kNN ambiguity. Taken from the step record's own direction
        # field, so this cannot silently invert. The student-side axis is the other
        # signal and is emitted only for context and for the 2x2.
        if st['direction'] == 'gnn->lm':
            teacher_sig, teacher_name = sig['local_homophily'], 'local_homophily'
            student_sig, student_name = sig['knn_ambiguity'], 'knn_ambiguity'
            t_out = 'low'   # teacher out of bias at LOW homophily
            s_out = 'high'
        else:
            teacher_sig, teacher_name = sig['knn_ambiguity'], 'knn_ambiguity'
            student_sig, student_name = sig['local_homophily'], 'local_homophily'
            t_out = 'high'  # teacher out of bias at HIGH ambiguity
            s_out = 'low'

        p_teacher = np.exp(teacher - teacher.max(1, keepdims=True))
        p_teacher /= p_teacher.sum(1, keepdims=True)
        soft_sig = sig_mod.soft_local_homophily(sig['edge_index'], p_teacher)

        base = {k: st[k] for k in ('arm', 'seed', 'dataset', 'regime', 'em_iter',
                                   'step_index', 'em_phase', 'direction',
                                   'teacher', 'student', 'pl_weight')}
        base['label_regime'] = base.pop('regime')
        base['iteration'] = base.pop('em_iter')
        base['step'] = base.pop('em_phase')
        base['n_analysed'] = len(idx)

        for axis_role, values, sname in (('teacher', teacher_sig, teacher_name),
                                         ('student', student_sig, student_name),
                                         ('teacher_soft', soft_sig, 'glance_soft_homophily')):
            for scheme, bins in (('median', median_bins(values, idx)),
                                 ('q5', quantile_bins(values, idx))):
                for bname, bidx in bins.items():
                    s = bin_stats(before, after, teacher, labels, bidx)
                    if s is None:
                        continue
                    rows.append({**base, 'axis': axis_role, 'signal': sname,
                                 'bin_scheme': scheme, 'bin': bname,
                                 'teacher_out_of_bias_bin': (
                                     bname == t_out if scheme == 'median' else ''),
                                 **s})

        # 2x2: teacher out-of-bias x student out-of-bias, median split on each.
        tb, sb = median_bins(teacher_sig, idx), median_bins(student_sig, idx)
        if tb and sb:
            for tk in ('low', 'high'):
                for sk in ('low', 'high'):
                    cell = np.intersect1d(tb[tk], sb[sk])
                    s = bin_stats(before, after, teacher, labels, cell)
                    if s is None:
                        continue
                    quads.append({**base, 'teacher_signal': teacher_name,
                                  'student_signal': student_name,
                                  'teacher_bin': tk, 'student_bin': sk,
                                  'teacher_out_of_bias': tk == t_out,
                                  'student_out_of_bias': sk == s_out,
                                  'is_predicted_cell': (tk == t_out and sk != s_out),
                                  **s})
    return rows, quads, notes


def find_runs(probe_dir):
    """Every ``<dataset>/<regime>/<arm>/seed<n>`` directory holding a step log."""
    return sorted(p.parent for p in Path(probe_dir).glob('*/*/*/seed*/steps.jsonl'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cache-dir', default=None)
    a = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(a.cache_dir) if a.cache_dir else Path(a.probe_dir) / '_signals'

    import pandas as pd
    all_rows, all_quads, all_notes = [], [], []
    runs = find_runs(a.probe_dir)
    print(f'{len(runs)} run(s) found')
    for r in runs:
        rows, quads, notes = analyze_run(r, cache)
        all_rows += rows
        all_quads += quads
        all_notes += ([notes] if isinstance(notes, str) else list(notes))
        print(f'  {r.relative_to(a.probe_dir)}: {len(rows)} rows, {len(quads)} cells')

    pd.DataFrame(all_rows).to_csv(out / 'ncs_long.csv', index=False)
    pd.DataFrame(all_quads).to_csv(out / 'quadrants.csv', index=False)
    if all_notes:
        (out / 'analysis_notes.txt').write_text('\n'.join(str(n) for n in all_notes) + '\n')
        print(f'{len(all_notes)} note(s) -> analysis_notes.txt')
    print(f'wrote {out}/ncs_long.csv ({len(all_rows)} rows), quadrants.csv ({len(all_quads)} rows)')


if __name__ == '__main__':
    main()
