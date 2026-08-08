#!/usr/bin/env bash
# Run one probed GLEM cell: (dataset, arm, label_regime, seed).
#
#   scripts/probe_run.sh <config> <arm> <regime> <seed> [extra args...]
#
#   config  basename in configs/glem/ without .sh   e.g. cornell_gcn, arxiv
#   arm     published | alpha0_li_T | alpha0_li_F   (EXPERIMENT.md section 8)
#   regime  standard | fewshot3 | fewshot5 | fewshot10
#   seed    integer
#
# The published hyperparameters come from configs/glem/<config>.sh verbatim; this
# script only appends the seed and, for the control arms, the zeroed pseudo-label
# weights. Later argparse occurrences win, so the appended flags override the
# config's without editing it.
set -euo pipefail

CONFIG=${1:?config name, e.g. cornell_gcn}
ARM=${2:?arm}
REGIME=${3:?regime}
SEED=${4:?seed}
shift 4

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# Defaults under temp/, which .gitignore already covers, alongside GLEM's own
# artefacts. Override with GLEM_PROBE_DIR to keep the archive elsewhere.
PROBE_DIR=${GLEM_PROBE_DIR:-$ROOT/temp/probe_output}
PY=${PY:-$ROOT/.venv/bin/python}

# shellcheck source=/dev/null
source "$ROOT/configs/glem/$CONFIG.sh"

case "$ARM" in
  published)
    ARM_ARGS="" ;;
  alpha0_li_T)
    # Arm 2a: pseudo-label CE term removed, everything else at published values.
    ARM_ARGS="--lm_pl_weight=0 --gnn_pl_weight=0" ;;
  alpha0_li_F)
    # Arm 2b: additionally stops the teacher's y_hat entering GNN *input features*
    # via gnn_label_input=T (utils/data/datasets.py::node_feature). On configs that
    # already set gnn_label_input=F this is identical to 2a -- do not run both.
    ARM_ARGS="--lm_pl_weight=0 --gnn_pl_weight=0 --gnn_label_input=F" ;;
  published_li_F)
    # A12: published alpha/beta, but the teacher's y_hat no longer concatenated onto
    # the GNN's input features. Isolates the feature channel alone; only differs from
    # 'published' on configs that set gnn_label_input=T (the four WebKB ones).
    ARM_ARGS="--gnn_label_input=F" ;;
  *)
    echo "unknown arm: $ARM" >&2; exit 2 ;;
esac

export GLEM_PROBE_DIR="$PROBE_DIR"
export GLEM_PROBE_ARM="$ARM"
export GLEM_PROBE_REGIME="$REGIME"
# Distinguishes configs sharing a DATASET_STR (cornell.sh RevGAT vs
# cornell_gcn.sh GCN both use cornell_TAG). Empty unless the caller sets it,
# so existing archive paths are untouched.
export GLEM_PROBE_VARIANT="${GLEM_PROBE_VARIANT:-}"

echo "=== probe: $DATASET_STR arm=$ARM regime=$REGIME seed=$SEED -> $PROBE_DIR"
# shellcheck disable=SC2086
exec "$PY" "$ROOT/src/models/GLEM/trainGLEM.py" $GLEM_ARGS --seed="$SEED" $ARM_ARGS "$@"
