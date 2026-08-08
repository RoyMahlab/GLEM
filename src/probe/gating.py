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

Both modes apply to whichever side is currently the student, since ``SeqGraph.init``
runs in both the LM and GNN processes.
"""
import numpy as np

from probe import context


def apply_gate(pl_nodes, pseudo_logits, labels, seed):
    """Filter ``pl_nodes`` per the active gate. Returns ``(kept_nodes, info)``.

    No-op returning the input unchanged unless ``GLEM_PROBE_GATE`` is set, so an
    ungated run is byte-identical to one from before this module existed.

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
        rng = np.random.default_rng(int(seed))
        kept = np.sort(rng.choice(pl_nodes, size=n_keep, replace=False))
    else:
        raise ValueError(f'unknown GLEM_PROBE_GATE={mode!r}; expected oracle|random')

    info = {'gate': mode, 'n_before': int(len(pl_nodes)), 'n_kept': int(len(kept)),
            'teacher_acc_on_pl': float(correct.mean())}
    print(f'[probe] gate={mode}: pseudo-label nodes {info["n_before"]} -> '
          f'{info["n_kept"]} (teacher accuracy on them {info["teacher_acc_on_pl"]:.4f})')
    return kept, info
