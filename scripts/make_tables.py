#!/usr/bin/env python
"""Refresh the beta / alpha result tables from the probe archive.

    scripts/make_tables.py <paper-dir>          # rewrite the data rows in place
    scripts/make_tables.py <paper-dir> --report  # print coverage, touch nothing

MINIMAL TOUCH. Only lines that begin a known dataset row (``  cora& ...``) are
rewritten. Captions, column specs, \\hline placement, row order and any hand edits
survive, because the sweep keeps landing and these tables get regenerated many times
-- a generator that rewrote whole files would silently revert every caption fix.

WHAT GOES IN A CELL, and why each rule exists:

* Accuracy is computed here, not read: nothing in the archive stores it.
  ``logits/iter<i>_gnn.npy`` is the GNN after the M-step of iteration i -- the LM->GNN
  transfer that beta weights -- and ``iter<i>_lm.npy`` is the LM after the E-step,
  which alpha weights. "Final" is the highest iteration present. The beta table
  therefore reports GNN accuracy and the alpha table reports LM accuracy: each names
  the *student* of the transfer it is about.

* A cell counts only when its ``steps.jsonl`` holds all four distillation events. An
  interrupted run leaves a valid-looking directory whose logits stop early.

* Selection deltas are PAIRED: the mean over seeds where both the arm and its
  size-matched control completed, never a difference of two means over different seed
  sets. GLEM's GNN training is not deterministic run-to-run (EXPERIMENT.md A19), so an
  unpaired delta folds that noise straight into the reported effect.

* WebKB rows use the RevGAT configs. The beta=0.05 column is `published` only where
  the config ships gnn_pl_weight=0.05; the four *_gcn recipes ship 0.7, so they cannot
  populate a beta column at all. RevGAT is also the backbone every other row uses.
"""
import argparse
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# table row label -> archive dataset directory
ROWS = {'arxiv': 'arxiv_TA', 'cora': 'cora_TAG', 'pubmed': 'pubmed_TAG',
        'wikics': 'wikics_TAG', 'citeseer': 'citeseer_TAG',
        'citeseer60': 'citeseer60_TAG',
        'bookhis': 'bookhis_TAG', 'bookchild': 'bookchild_TAG',
        'sportsfit': 'sportsfit_TAG',
        'cornell': 'cornell_TAG+revgat', 'texas': 'texas_TAG+revgat',
        'washington': 'washington_TAG+revgat', 'wisconsin': 'wisconsin_TAG+revgat'}

# (student, weight symbol, selection arms, exposure arms, size-matched control)
SPEC = {
    'glem-beta.tex': dict(
        kind='gnn',
        selection=['conf_gate80', 'sig_gate80_gnn', 'sig_gate80_lm'],
        exposure=['published', 'b30', 'beta_high'],
        control='b05_rand80'),
    # alpha=0.8 IS the published recipe on every RevGAT config, so `published` fills
    # the third exposure slot. Nothing has ever run alpha=0.05 or alpha=0.3 -- no arm
    # pins --lm_pl_weight except the alpha0 controls, and those zero beta at the same
    # time, so they are a joint control rather than a pure alpha=0 point. The
    # selection columns need an E-step size-matched control (`random80:gnn`) that does
    # not exist: b05_rand80 is `random80:lm`, which gates the M-step.
    'glem-alpha.tex': dict(
        kind='lm',
        selection=[None, None, None],
        exposure=[None, None, 'published'],
        control=None),
}


def load_archive(probe_dir):
    """{(dataset, arm, student): {seed: test_acc}} over complete cells only."""
    by = defaultdict(dict)
    for steps in sorted(Path(probe_dir).glob('*/standard/*/seed*/steps.jsonl')):
        if sum(1 for _ in open(steps)) != 4:
            continue
        run = steps.parent
        ds, arm = run.parents[2].name, run.parents[0].name
        seed = int(run.name.replace('seed', ''))
        sp = np.load(run / 'splits.npz')
        labels, test_x = sp['labels'], sp['test_x'].reshape(-1)
        for kind in ('gnn', 'lm'):
            its = [int(m.group(1)) for f in (run / 'logits').glob(f'iter*_{kind}.npy')
                   if (m := re.match(rf'iter(-?\d+)_{kind}\.npy', f.name))]
            if not its:
                continue
            lg = np.load(run / 'logits' / f'iter{max(its)}_{kind}.npy').astype(np.float32)
            if test_x.size:
                by[(ds, arm, kind)][seed] = float(
                    (lg[test_x].argmax(1) == labels[test_x]).mean())
    return by


def fmt(by, ds, arm, kind, control):
    if arm is None:
        return ''
    a = by.get((ds, arm, kind))
    if not a:
        return ''
    if control is None:                       # exposure cell: plain accuracy
        return f'{100 * st.mean(a.values()):.1f}' + (r'\dg' if len(a) == 1 else '')
    c = by.get((ds, control, kind))
    if not c:
        return ''
    seeds = sorted(set(a) & set(c))           # paired, never unpaired
    if not seeds:
        return ''
    d = 100 * st.mean(a[s] - c[s] for s in seeds)
    return f'{d:+.1f}' + (r'\dg' if len(seeds) == 1 else '')


def rewrite(path, by, spec, report):
    kind, ctl = spec['kind'], spec['control']
    out, touched = [], 0
    for line in path.read_text().splitlines(keepends=True):
        m = re.match(r'^(\s*)([A-Za-z][A-Za-z0-9]*)&', line)
        if not m or m.group(2) not in ROWS:
            out.append(line)
            continue
        indent, name = m.groups()
        ds = ROWS[name]
        cells = ([fmt(by, ds, a, kind, ctl) for a in spec['selection']] +
                 [fmt(by, ds, a, kind, None) for a in spec['exposure']])
        out.append(f'{indent}{name}& ' + '& '.join(cells) + '\\\\\n')
        touched += 1
    if report:
        return touched
    path.write_text(''.join(out))
    return touched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paper_dir', help='dir holding sections/tables/glem-*.tex')
    ap.add_argument('--probe-dir', default=str(ROOT / 'temp' / 'probe_output'))
    ap.add_argument('--report', action='store_true', help='print coverage only')
    a = ap.parse_args()

    by = load_archive(a.probe_dir)
    cells = len({(d, ar, s) for (d, ar, k), v in by.items() for s in v if k == 'gnn'})
    print(f'{cells} complete cell(s) in {a.probe_dir}')

    tables = Path(a.paper_dir) / 'sections' / 'tables'
    for fname, spec in SPEC.items():
        p = tables / fname
        if not p.exists():
            print(f'  !! {p} not found, skipped')
            continue
        n = rewrite(p, by, spec, a.report)
        verb = 'would rewrite' if a.report else 'rewrote'
        print(f'  {verb} {n} row(s) in {fname}')

    if a.report:
        print('\n=== per (dataset, arm) seed coverage, GNN student ===')
        for name, ds in ROWS.items():
            arms = {ar: sorted(v) for (d, ar, k), v in by.items()
                    if d == ds and k == 'gnn' for _ in [0]}
            if arms:
                print(f'  {name:<12} ' + '  '.join(
                    f'{ar}{s}' for ar, s in sorted(arms.items())))
    return 0


if __name__ == '__main__':
    sys.exit(main())
