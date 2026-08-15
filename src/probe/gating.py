"""Oracle gating: the ceiling on what any per-node pseudo-label gate could achieve.

GLEM weights every pseudo-label with one global scalar (α for the LM step, β for the
GNN step), so a node the teacher gets right and a node it gets wrong contribute
equally. The experiment in EXPERIMENT.md asked whether that uniformity *harms* an
identifiable population; it does not, because the teacher stays better than the
student even where it is weakest. This module asks the complementary question:

    if you could remove the teacher's wrong labels *perfectly*, how much accuracy
    would you gain?

That is an upper bound, not a method -- it uses ground truth, so it cannot be
deployed. It bounds every realisable gate, including confidence filtering
(``pl_filter``) and any per-node α predicted from exogenous signals.

Two modes, and the second is not optional:

``oracle``
    Keep only pseudo-label nodes where the teacher's argmax equals the gold label.

``random``
    Keep a uniformly random subset of the **same size**. Without this the oracle
    confounds "removed wrong labels" with "trained on less pseudo-data" -- on arxiv
    the oracle drops ~23% of the pseudo-label set, and shrinking a training set
    changes the result on its own. The interpretable comparison is oracle vs random,
    never oracle vs published.

    The matching is **within-run, not across-run**. Each arm sizes its cut from its
    own teacher's error count, so the two agree at the first step (identical
    pretrained teacher) and drift apart afterwards as the trajectories diverge -- on
    cornell, 44/44 at step 1 but 56 vs 47 by step 2. Forcing identical counts would
    mean piping one run's numbers into the other, which is fragile and arguably
    wrong: the honest control is "drop as many labels as this run's teacher gets
    wrong, chosen without regard to which ones", and that is what this does.

``signal<k>`` (e.g. ``signal80``)
    Keep the top ``k%`` of pseudo-label nodes by an **exogenous** signal, chosen by
    direction: GLANCE soft homophily when the GNN teaches, inverted kNN ambiguity
    when the LM teaches. Deployable -- neither needs the node's own label.

    Motivation, measured on arxiv (A18): at the oracle keep-rate, a confidence gate
    excludes 51% of the teacher's errors but **0%** of the *confidently wrong* ones,
    which is exactly why ``conf_gate`` bought nothing. The exogenous signals exclude
    a similar share of all errors (42-49%) but **11.6% / 17.7%** of the confidently
    wrong population that confidence is structurally blind to.

Both modes apply to whichever side is currently the student, since ``SeqGraph.init``
runs in both the LM and GNN processes.
"""
import numpy as np

from probe import context


def _exogenous_score(pl_nodes, pseudo_logits, cf):
    """Per-node score for the signal gate; higher = teacher more likely RIGHT.

    Direction is read from ``cf.em_phase`` rather than re-derived: the LM being the
    student means the **GNN** is teaching, so the axis is local homophily. Getting
    this inverted would gate on the student's weakness instead of the teacher's,
    which is the error EXPERIMENT.md section 4 exists to prevent.

    Fails loudly if the cached signal arrays are absent. A silent fallback would
    degrade the gate invisibly *inside training*, where it could not be caught by
    inspecting the results afterwards.
    """
    import os
    from pathlib import Path

    key = str(cf.dataset).split('_')[0]
    sig_dir = Path(os.environ[context.ENV_DIR]) / '_signals'
    teaching = 'GNN' if cf.em_phase == 'LM' else 'LM'

    p = np.asarray(pseudo_logits, dtype=np.float32)
    p = np.exp(p - p.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)

    if teaching == 'GNN':
        # GNN teacher fails at LOW homophily -> higher GLANCE = more trustworthy.
        edge_f = sig_dir / f'{key}_edge_index.npy'
        if not edge_f.exists():
            raise FileNotFoundError(
                f'signal gate needs {edge_f}; generate it by running '
                f'src/probe/analyze.py on this dataset first')
        from probe.signals import soft_local_homophily
        score = soft_local_homophily(np.load(edge_f), p)
        name = 'glance_soft_homophily'
    else:
        # LM teacher fails at HIGH ambiguity -> negate so higher = more trustworthy.
        amb_f = sig_dir / f'{key}_standard_s0_ambiguity.npy'
        if not amb_f.exists():
            raise FileNotFoundError(
                f'signal gate needs {amb_f}; generate it by running '
                f'src/probe/analyze.py on this dataset first')
        score = -np.load(amb_f)
        name = 'neg_knn_ambiguity'

    score = np.asarray(score, dtype=np.float64)
    score = np.nan_to_num(score, nan=np.nanmedian(score))   # isolated nodes
    return score[pl_nodes], name, teaching


def apply_gate(pl_nodes, pseudo_logits, labels, cf):
    """Filter ``pl_nodes`` per the active gate. Returns ``(kept_nodes, info)``.

    No-op returning the input unchanged unless ``GLEM_PROBE_GATE`` is set, so an
    ungated run is byte-identical to one from before this module existed.

    Also rewrites ``cf.emi.n_pl_nodes`` to the kept count. That is not optional:
    ``EmIterInfo.inf_node_ranges`` derives the LM's per-iteration window from
    ``n_pl_nodes``, which is fixed at config time from the *ungated* set. Leaving it
    stale makes the window overrun the shortened array, wrap into the second copy
    made by ``np.tile`` in ``get_inf_aug_train_ids``, and trip its duplicate
    assertion. GLEM's own ``pl_filter`` scales the same field for the same reason
    (``GLEM_utils.EmIterInfo.__init__``), so this mirrors existing behaviour rather
    than inventing it.

    Deterministic on every rank: the oracle mask is a pure function of the teacher's
    logits, and the random subset is drawn from a generator seeded by the run seed.
    That matters because ``SeqGraph.init`` runs on all ranks under ``torchrun`` and
    a per-rank disagreement about the training set would corrupt DDP silently.
    """
    mode = context.gate()
    if not context.enabled() or not mode:
        return pl_nodes, {}

    pl_nodes = np.asarray(pl_nodes)
    if len(pl_nodes) == 0:
        return pl_nodes, {'gate': mode, 'n_before': 0, 'n_kept': 0}

    teacher_pred = np.asarray(pseudo_logits[pl_nodes], dtype=np.float32).argmax(1)
    correct = teacher_pred == np.asarray(labels)[pl_nodes]
    n_keep = int(correct.sum())

    if mode == 'oracle':
        kept = pl_nodes[correct]
    elif mode == 'random':
        # Same count as the oracle would keep, chosen without regard to correctness.
        rng = np.random.default_rng(int(cf.seed))
        kept = np.sort(rng.choice(pl_nodes, size=n_keep, replace=False))
    elif mode.startswith('signal'):
        # ``signal<k>``        gate every step   -- fix BOTH teachers
        # ``signal<k>:gnn``    gate only when the GNN teaches -- fix the GNN teacher
        # ``signal<k>:lm``     gate only when the LM teaches  -- fix the LM teacher
        #
        # The three arms decompose the effect by teacher. `cf.em_phase` names the
        # STUDENT, so the teacher is the other model: em_phase == 'LM' means the GNN
        # is teaching. Reading it this way rather than re-deriving keeps the
        # direction from silently inverting (EXPERIMENT.md section 4).
        spec = mode[len('signal'):]
        frac_s, _, which = spec.partition(':')
        which = (which or 'both').lower()
        teaching = 'GNN' if cf.em_phase == 'LM' else 'LM'

        if which != 'both' and which != teaching.lower():
            # This step's teacher is not the one this arm fixes: leave it exactly as
            # `published` would run it. Returning early also leaves emi.n_pl_nodes
            # untouched, which is required -- shrinking it for an ungated step would
            # desynchronise the LM's per-iteration window from the full pl set.
            print(f'[probe] gate={mode}: {teaching} is teaching, not gated '
                  f'(this arm fixes {which.upper()} only)')
            return pl_nodes, {'gate': mode, 'gate_active': False,
                              'gate_teaching': teaching,
                              'n_before': int(len(pl_nodes)),
                              'n_kept': int(len(pl_nodes))}

        frac = int(frac_s) / 100.0
        score, sig_name, teaching = _exogenous_score(pl_nodes, pseudo_logits, cf)
        n_keep = max(1, int(round(frac * len(pl_nodes))))
        kept = np.sort(pl_nodes[np.argsort(-score, kind='mergesort')[:n_keep]])
    else:
        raise ValueError(
            f'unknown GLEM_PROBE_GATE={mode!r}; expected oracle|random|signal<k>')

    emi = getattr(cf, 'emi', None)
    if emi is not None:
        emi.n_pl_nodes = int(len(kept))

    info = {'gate': mode, 'n_before': int(len(pl_nodes)), 'n_kept': int(len(kept)),
            'teacher_acc_on_pl': float(correct.mean())}
    if mode.startswith('signal'):
        info['gate_active'] = True
        # teacher_acc_on_kept is diagnostic only -- it uses gold labels, so it is
        # recorded for the analysis, never consulted by the gate itself.
        info.update({'gate_signal': sig_name, 'gate_teaching': teaching,
                     'teacher_acc_on_kept': float(
                         (np.asarray(pseudo_logits[kept], dtype=np.float32).argmax(1)
                          == np.asarray(labels)[kept]).mean())})
    print(f'[probe] gate={mode}: pseudo-label nodes {info["n_before"]} -> '
          f'{info["n_kept"]} (teacher accuracy on them {info["teacher_acc_on_pl"]:.4f})')
    return kept, info
