#!/usr/bin/env python
"""Build the cached signal arrays the gated arms read, before any training starts.

    scripts/prepare_signals.py arxiv cora wikics      # named configs
    scripts/prepare_signals.py --all                  # every config in the matrix
    scripts/prepare_signals.py --check --all          # report only, build nothing

WHY THIS EXISTS. ``src/probe/gating.py`` reads two arrays per dataset and raises
FileNotFoundError *inside training* when either is absent:

    <key>_edge_index.npy               GLANCE soft homophily, gates the E-step
    <key>_standard_s0_ambiguity.npy    kNN ambiguity, gates the M-step

On arxiv that failure lands roughly three hours into a thirteen-hour run. Worse, the
edge_index array was previously written only by a cell in
``notebooks/glem_gate_selection.ipynb``, so a fresh checkout could not produce it at
all, and ``wikics`` was missing it on the original machine while the matrix happily
assigned it three homophily-gated arms.

NO COMPLETED RUN IS NEEDED. The train split GLEM uses is exactly TAGDataset's
``train_mask`` -- ``utils/data/preprocess_tag.load_tag_dgl`` derives its ``split_idx``
from those masks -- so every signal can be built straight from the dataset. Where an
archived run does exist, its ``splits.npz`` is cross-checked against the mask and a
mismatch is fatal: a silently different split would put the ambiguity axis out of
step with the runs it gates.

SEED. Only ``s0`` is written. ``gating.py`` reads seed 0's ambiguity for every seed,
which is sound under the standard regime because ambiguity draws its neighbours from
the train split and that does not vary with the run seed there. The per-seed files
``analyze.py`` writes later are byte-identical copies.
"""
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def dataset_str_for(config: str) -> str:
    """DATASET_STR as the config's own shell would set it."""
    text = (ROOT / 'configs' / 'glem' / f'{config}.sh').read_text()
    m = re.findall(r'^\s*DATASET_STR=([^\s#]+)', text, flags=re.M)
    if not m:
        raise SystemExit(f'{config}.sh does not set DATASET_STR')
    return m[-1].strip('"\'')


def configs_from_matrix():
    """ALL_CONFIGS, read from scripts/matrix.sh so the two cannot drift."""
    text = (ROOT / 'scripts' / 'matrix.sh').read_text()
    m = re.search(r'ALL_CONFIGS=\((.*?)\)', text, flags=re.S)
    if not m:
        raise SystemExit('matrix.sh does not define ALL_CONFIGS')
    return m.group(1).split()


def arms_needing_signals(config: str):
    """(needs_hom, needs_amb) for a config, from matrix.sh's ARMS and ARM_NEEDS."""
    text = (ROOT / 'scripts' / 'matrix.sh').read_text()
    arms = re.search(rf'^\s*\[{re.escape(config)}\]="([^"]*)"', text, flags=re.M)
    arms = set(arms.group(1).split()) if arms else set()
    needs = dict(re.findall(r'\[(\w+)\]="([a-z ]+)"',
                            re.search(r'ARM_NEEDS=\((.*?)\n\)', text, re.S).group(1)))
    hom = any('hom' in needs.get(a, '') for a in arms)
    amb = any('amb' in needs.get(a, '') for a in arms)
    return hom, amb


def verify_split(dataset: str, train_idx: np.ndarray, probe_dir: Path, key: str):
    """Fatal if an archived run disagrees with the mask this script split on."""
    hits = sorted(probe_dir.glob(f'{dataset}*/standard/*/seed*/splits.npz'))
    if not hits:
        return 'no archived run to cross-check against'
    archived = np.load(hits[0])['train_x'].reshape(-1)
    if archived.shape != train_idx.shape or not np.array_equal(
            np.sort(archived), np.sort(train_idx)):
        raise SystemExit(
            f'!! {key}: the train split from TAGDataset ({train_idx.size} nodes) does '
            f'not match the archived run {hits[0].parent} ({archived.size} nodes).\n'
            f'   The ambiguity axis would be out of step with the runs it gates. '
            f'Refusing to write a signal that disagrees with the archive.')
    return f'split cross-checked against {hits[0].parent.name}'


def prepare(config: str, probe_dir: Path, check_only: bool, force: bool):
    dataset = dataset_str_for(config)
    key = dataset.split('_')[0]
    need_hom, need_amb = arms_needing_signals(config)
    sig = probe_dir / '_signals'

    edge_f = sig / f'{key}_edge_index.npy'
    hom_f = sig / f'{key}_homophily.npy'
    emb_f = sig / f'{key}_sbert.npy'
    amb_f = sig / f'{key}_standard_s0_ambiguity.npy'

    wanted = []
    if need_hom:
        wanted.append(edge_f)
    if need_amb:
        wanted.append(amb_f)
    if not wanted:
        print(f'  {config:<15} ({key}): no gated arms in the matrix, nothing needed')
        return True

    missing = [f for f in wanted if not (f.exists() and f.stat().st_size > 0)]
    if not missing and not force:
        print(f'  {config:<15} ({key}): present')
        return True
    if check_only:
        print(f'  {config:<15} ({key}): MISSING {", ".join(f.name for f in missing)}')
        return False

    print(f'  {config:<15} ({key}): building {", ".join(f.name for f in missing)}')
    sig.mkdir(parents=True, exist_ok=True)
    from probe.signals import (load_tag_data, local_homophily, sbert_embed,
                               knn_ambiguity)

    data = load_tag_data(dataset)
    edge_index = np.asarray(data.edge_index)
    labels = data.y.view(-1).numpy()
    n_nodes = int(labels.shape[0])
    num_classes = int(labels.max()) + 1
    train_idx = np.nonzero(data.train_mask.numpy().reshape(-1))[0]
    print(f'      {n_nodes} nodes, {num_classes} classes, '
          f'{train_idx.size} train ({train_idx.size / n_nodes:.1%})')
    print(f'      {verify_split(dataset, train_idx, probe_dir, key)}')

    if need_hom and (force or not edge_f.exists()):
        np.save(edge_f, edge_index)
        print(f'      wrote {edge_f.name}')

    if need_amb:
        # homophily and the embeddings are inputs to ambiguity's cache, and are
        # themselves what analyze.py would otherwise recompute per run.
        if force or not hom_f.exists():
            np.save(hom_f, local_homophily(edge_index, labels, n_nodes))
            print(f'      wrote {hom_f.name}')
        if force or not emb_f.exists():
            emb = sbert_embed(list(data.raw_texts))
            np.save(emb_f, emb)
            print(f'      wrote {emb_f.name}')
        else:
            emb = np.load(emb_f)
        if force or not amb_f.exists():
            np.save(amb_f, knn_ambiguity(emb, labels, train_idx, num_classes))
            print(f'      wrote {amb_f.name}')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('configs', nargs='*')
    ap.add_argument('--all', action='store_true', help='every config in matrix.sh')
    ap.add_argument('--check', action='store_true',
                    help='report what is missing and exit non-zero; build nothing')
    ap.add_argument('--force', action='store_true', help='rebuild even if present')
    ap.add_argument('--probe-dir', default=os.environ.get(
        'GLEM_PROBE_DIR', str(ROOT / 'temp' / 'probe_output')))
    a = ap.parse_args()

    configs = configs_from_matrix() if a.all else a.configs
    if not configs:
        raise SystemExit('name at least one config, or pass --all')

    probe_dir = Path(a.probe_dir)
    print(f'signals under {probe_dir / "_signals"}')
    ok = True
    for c in configs:
        ok &= prepare(c, probe_dir, a.check, a.force)
    if not ok:
        print('\nmissing signals; build them by re-running without --check')
        return 1
    print('\nall required signals present')
    return 0


if __name__ == '__main__':
    sys.exit(main())
