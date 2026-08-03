"""Per-step logits archive and step provenance log.

GLEM's teacher and student communicate *only* through fp16 memmap files on disk
(GNN pred written by ``gnn_utils.save_and_report_gnn_result``, LM pred written by
``LMs/infLM.py``, each read back as the other's ``pseudo_label_file``). Two
consequences shape this module:

1. **No new forward passes are needed.** A student's pre-step logits are the file
   its own side wrote at the previous step, and a teacher's logits are the file
   the step reads. Archiving every write reconstructs every before/after/teacher
   triple exactly, from the real snapshots rather than from a proxy.

2. **Those files are overwritten in place.** For ``em_iter >= 0`` the pred paths
   carry no iteration index (``GLEM_utils.EmIterInfo``, the ``gnn_folder`` /
   ``lm_folder`` assignments), so each M-step clobbers the previous GNN pred and
   each E-step clobbers the previous LM pred. Before/after pairs are destroyed as
   a run proceeds. This module is what makes the experiment possible at all, and
   it does it by copying to an iteration-keyed path at write time rather than by
   restructuring ``EmIterInfo``.

Layout under ``<probe dir>/<dataset>/<regime>/<arm>/seed<seed>/``::

    logits/iter<i>_gnn.npy        # GNN pred after the M-step of iteration i
    logits/iter<i>_lm.npy         # LM  pred after the E-step of iteration i
    logits/iter-1_{gnn,lm}.npy    # the pretrained (gold-only) baselines
    teacher/step<n>_<phase>.npy   # the teacher logits that step actually consumed
    plnodes/step<n>_<phase>.npy   # node ids that actually received a pseudo-label
    steps.jsonl                   # one append-only record per distillation event

Nothing here is read back during training; the analysis driver consumes it after
the fact.
"""
import json
import os

import numpy as np

from probe import context


def _as_f16(x) -> np.ndarray:
    """Materialize a torch tensor / memmap / ndarray as an in-memory fp16 array."""
    if hasattr(x, 'detach'):  # torch.Tensor
        x = x.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(x), dtype=np.float16)


def _read_pred(path, n_labels) -> np.ndarray:
    """Load a GLEM pred memmap, inferring ``n_nodes`` from the file size.

    The pred files are raw fp16 with no header, so the row count comes from
    ``filesize / (2 * n_labels)``. Reading it this way avoids having to reconstruct
    a ``SeqGraph`` just to learn ``n_nodes``.
    """
    n_nodes = os.path.getsize(path) // (2 * int(n_labels))
    return np.array(np.memmap(path, dtype=np.float16, mode='r',
                              shape=(int(n_nodes), int(n_labels))))


def step_index(em_order, em_phase, em_iter) -> int:
    """Position of a step in the run's execution order; -1 for pretraining.

    ``GLEM_trainer.glem_train`` runs the two steps of each iteration in an order
    fixed by ``em_order``, so the index is derivable rather than needing a shared
    counter across subprocesses. Recorded on every row so the analysis never has
    to re-derive which model taught which -- the mistake EXPERIMENT.md section 4
    is written to prevent.
    """
    if em_iter is None or int(em_iter) < 0:
        return -1
    i = int(em_iter)
    first = 'GNN' if em_order == 'GNN-first' else 'LM'
    return 2 * i + (0 if em_phase == first else 1)


def _n_labels(cf):
    return int(cf.data.n_labels)


def _run_dir(cf):
    return context.run_dir(cf.dataset, cf.seed)


def archive_pred(cf, pred, kind) -> None:
    """Archive freshly-written logits under an iteration-keyed path.

    Call immediately after the live pred memmap is written, with the same array.
    ``kind`` is ``'gnn'`` or ``'lm'``.
    """
    if not context.enabled() or not context.is_main_rank(cf):
        return
    out = _run_dir(cf) / 'logits' / f'iter{int(cf.em_iter)}_{kind}.npy'
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, _as_f16(pred))


def archive_pred_file(cf, src, kind, em_iter) -> None:
    """Archive logits by reading them back from ``src``.

    Needed for the pretrained baselines: ``GLEM_trainer._pre_train_{lm,gnn}`` skip
    the work entirely when a checkpoint already exists in ``temp/``, so the
    ``archive_pred`` hook inside the training process never fires. Without this the
    ``iter-1`` snapshots -- the "before" side of the very first distillation event
    -- would be missing on any run that reused a cached pretrain.
    """
    if not context.enabled() or not context.is_main_rank(cf):
        return
    out = _run_dir(cf) / 'logits' / f'iter{int(em_iter)}_{kind}.npy'
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return
    np.save(out, _read_pred(src, _n_labels(cf)))


def archive_splits(cf, g_info) -> None:
    """Archive the split and labels this run actually used.

    The kNN ambiguity signal draws its neighbours from the *labelled training
    nodes*, so under the few-shot regimes it depends on the exact k-shot draw. The
    analysis reads the split from here rather than recomputing it, which removes
    any chance of scoring a run against a split it never saw.
    """
    if not context.enabled() or not context.is_main_rank(cf):
        return
    out = _run_dir(cf) / 'splits.npz'
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return
    np.savez_compressed(
        out,
        labels=np.asarray(g_info.labels).reshape(-1),
        train_x=np.asarray(g_info.splits['train_x']).reshape(-1),
        valid_x=np.asarray(g_info.splits['valid_x']).reshape(-1),
        test_x=np.asarray(g_info.splits['test_x']).reshape(-1),
        is_gold=np.asarray(g_info.is_gold).reshape(-1),
        n_nodes=np.array(int(g_info.n_nodes)),
    )


def record_step(cf, pl_node_ids, teacher_file=None, extra=None) -> None:
    """Log one distillation event: its provenance, its α/β, and who it taught.

    ``pl_node_ids`` are the nodes that *actually received a pseudo-label* in this
    step (EXPERIMENT.md section 7). This cannot be reconstructed afterwards -- the
    GNN resamples every epoch and the LM takes a per-iteration window -- so it is
    logged here or not at all.

    ``teacher_file`` is copied, not just referenced. The referenced path is
    overwritten by a later step (see module docstring), and section 4 requires
    teacher accuracy to come from the snapshot the step actually consumed.
    """
    if not context.enabled() or not context.is_main_rank(cf):
        return
    d = _run_dir(cf)
    em_order = cf.emi.cf.em_order
    phase = cf.em_phase
    idx = step_index(em_order, phase, cf.em_iter)
    tag = f'step{idx}_{phase}'

    if pl_node_ids is not None:
        p = d / 'plnodes' / f'{tag}.npy'
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, np.asarray(sorted(set(int(i) for i in pl_node_ids)), dtype=np.int64))

    teacher_snap = None
    if teacher_file and os.path.exists(teacher_file):
        t = d / 'teacher' / f'{tag}.npy'
        t.parent.mkdir(parents=True, exist_ok=True)
        np.save(t, _read_pred(teacher_file, _n_labels(cf)))
        teacher_snap = str(t.relative_to(d))

    rec = {
        'arm': context.arm(),
        'regime': context.regime(),
        'dataset': cf.dataset,
        'seed': int(cf.seed),
        'em_order': em_order,
        'em_phase': phase,
        'em_iter': int(cf.em_iter),
        'step_index': idx,
        # 'direction' is the thing it is easiest to get backwards, so it is
        # written out in full rather than left to the analysis to infer.
        'direction': 'gnn->lm' if phase == 'LM' else 'lm->gnn',
        'teacher': 'GNN' if phase == 'LM' else 'LM',
        'student': 'LM' if phase == 'LM' else 'GNN',
        'pl_weight': float(getattr(cf, 'pl_weight', float('nan'))),
        'pl_ratio': float(getattr(cf, 'pl_ratio', float('nan'))),
        'pl_filter': str(getattr(cf, 'pl_filter', '')),
        'label_input': str(getattr(cf, 'label_input', '')),
        'is_augmented': bool(getattr(cf, 'is_augmented', False)),
        'pseudo_label_file': str(getattr(cf, 'pseudo_label_file', '')),
        'feature_file': str(getattr(cf, 'feature_file', '')),
        'teacher_snapshot': teacher_snap,
        'n_pl_nodes_used': None if pl_node_ids is None else int(len(set(int(i) for i in pl_node_ids))),
        **(extra or {}),
    }
    d.mkdir(parents=True, exist_ok=True)
    with open(d / 'steps.jsonl', 'a') as f:
        f.write(json.dumps(rec, sort_keys=True) + '\n')
