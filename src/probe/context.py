"""Run identity for a probed GLEM run, carried through the environment.

A GLEM run is one point in ``(dataset, label_regime, arm, seed)``. The dataset and
seed already live on every ``cf``; the regime and arm are what this module adds,
because they are properties of the *experiment* rather than of GLEM.

All state is read from the environment on each call rather than cached, so a
parent process that mutates ``os.environ`` between runs (see
``probe.run_experiment``) does not need to reload anything.
"""
import os
from pathlib import Path

ENV_DIR = 'GLEM_PROBE_DIR'
ENV_ARM = 'GLEM_PROBE_ARM'
ENV_REGIME = 'GLEM_PROBE_REGIME'
ENV_VARIANT = 'GLEM_PROBE_VARIANT'
ENV_GATE = 'GLEM_PROBE_GATE'

#: Arms defined by EXPERIMENT.md section 8. ``published`` is Arm 1; the two
#: ``alpha0_*`` arms are Arm 2a / 2b, differing only in ``gnn_label_input``.
#: ``published_li_F`` is the exploratory fourth cell of that 2x2 (amendment A12):
#: published pseudo-label weights, feature concatenation removed. It is NOT a
#: control -- ``report.py`` treats only the ``alpha0_*`` arms as such.
ARMS = ('published', 'alpha0_li_T', 'alpha0_li_F', 'published_li_F',
        'oracle', 'oracle_random',
        'conf_gate60', 'conf_gate80', 'conf_gate90',
        'sig_gate80', 'sig_gate90',
        'sig_gate80_gnn', 'sig_gate80_lm',
        'sig_gate90_gnn', 'sig_gate90_lm')


def enabled() -> bool:
    """True when instrumentation is active. Every hook short-circuits on False."""
    return bool(os.environ.get(ENV_DIR, ''))


def root() -> Path:
    return Path(os.environ[ENV_DIR])


def arm() -> str:
    return os.environ.get(ENV_ARM, 'published')


def regime() -> str:
    """``standard`` or ``fewshot<k>`` (k labels per class)."""
    return os.environ.get(ENV_REGIME, 'standard')


def fewshot_k():
    """``k`` for a ``fewshot<k>`` regime, else None."""
    r = regime()
    return int(r[len('fewshot'):]) if r.startswith('fewshot') else None


def path_suffix(seed) -> str:
    """Suffix that keys GLEM's artefact directories by label regime.

    Returns ``''`` for the standard split -- so a probed standard run reuses the
    pretrained LM/GNN checkpoints already cached in ``temp/`` and stays
    path-identical to an unprobed one -- and ``_fs<k>_s<seed>`` for few-shot.

    The seed is in the few-shot suffix but not the standard one, and that
    asymmetry is deliberate. GLEM's pretrain paths
    (``EmIterInfo``, ``MNT_TEMP_DIR/prt_{lm,gnn}/<dataset>/...``) carry neither a
    regime nor a seed, so all seeds share one pretrained checkpoint. Under the
    standard split that is GLEM's own published behaviour and is preserved. Under
    few-shot the *training data itself* differs per seed (``probe.fewshot`` draws
    k labels per class with the run seed), so sharing a checkpoint across seeds
    would train on labels the run is supposed not to have.
    """
    k = fewshot_k()
    return '' if k is None else f'_fs{k}_s{seed}'


def gate() -> str:
    """Active pseudo-label gate: ``oracle``, ``random``, or ``''`` for none.

    See ``probe.gating``. Empty by default, so ungated runs are unaffected.
    """
    return os.environ.get(ENV_GATE, '').strip()


def variant() -> str:
    """Optional suffix distinguishing configurations that share a dataset name.

    The archive is keyed by ``(dataset, regime, arm, seed)``, which has no room
    for the GNN backbone -- yet ``configs/glem/cornell.sh`` (RevGAT) and
    ``cornell_gcn.sh`` (GCN) both declare ``DATASET_STR="cornell_TAG"``. Without a
    discriminator the second would land on the first's archive and
    ``reset_run_dir`` would delete it.

    Empty by default, so every existing archive path is unchanged; the sweep sets
    it only for configs that would otherwise collide.
    """
    v = os.environ.get(ENV_VARIANT, '').strip()
    return f'+{v}' if v else ''


def run_dir(dataset, seed) -> Path:
    """Directory holding every artefact this run's probe writes."""
    return root() / f'{dataset}{variant()}' / regime() / arm() / f'seed{seed}'


def is_main_rank(cf) -> bool:
    """True on the rank that owns disk writes.

    Steps are launched under ``torchrun`` with one process per GPU
    (``run_command_parallel``), and all ranks reach the same hooks. Only
    ``local_rank <= 0`` may write, or ranks race on the same paths.
    """
    return getattr(cf, 'local_rank', -1) <= 0
