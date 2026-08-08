"""Measurement instrument for GLEM's uniform pseudo-labeling (see EXPERIMENT.md).

This package **only observes**. Every entry point is a no-op unless the
environment variable ``GLEM_PROBE_DIR`` is set, so an un-instrumented GLEM run is
byte-identical to one from before this package existed. The two exceptions are
deliberate and documented in EXPERIMENT.md section 14:

* ``fewshot`` rewrites the train split when ``GLEM_PROBE_REGIME=fewshot<k>``,
  which is the Arm 3 label-regime sweep and cannot be done by observation alone;
* ``context.path_suffix`` keys the pretrain/EM artefact directories by regime and
  seed, without which a few-shot run would silently reuse a fully-supervised
  pretrained LM (see ``fewshot`` docstring).

Why the environment rather than CLI arguments: GLEM's EM loop is a subprocess
orchestrator (``models/GLEM/GLEM_trainer.py``) that shells out per step with
``os.system``, so the environment is inherited by every child for free, whereas a
new CLI flag would have to be threaded through three independent config classes
and their ``para_prefix`` path-string machinery -- which would also change the
artefact paths of un-instrumented runs.
"""
from probe import context, fewshot, gating, snapshots  # noqa: F401
from probe.context import enabled  # noqa: F401
from probe.fewshot import apply_fewshot  # noqa: F401
from probe.gating import apply_gate  # noqa: F401
from probe.snapshots import (  # noqa: F401
    archive_pred,
    archive_pred_file,
    archive_splits,
    record_step,
    reset_run_dir,
)
