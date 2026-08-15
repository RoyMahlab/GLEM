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
  oracle)
    # Published hyperparameters, but the pseudo-label set is restricted to nodes
    # the teacher gets RIGHT. Uses ground truth, so it is an upper bound on any
    # per-node gate, not a deployable method. See src/probe/gating.py.
    ARM_ARGS=""; export GLEM_PROBE_GATE=oracle ;;
  oracle_random)
    # Size-matched control: drops the same NUMBER of pseudo-labels at random.
    # Without it, oracle-vs-published confounds 'removed wrong labels' with
    # 'trained on less pseudo-data'. Compare oracle against THIS, not published.
    ARM_ARGS=""; export GLEM_PROBE_GATE=random ;;
  conf_gate60|conf_gate80|conf_gate90)
    # Confidence gate at a fixed keep-rate, using GLEM's OWN pl_filter mechanism:
    # softmax(...).max(1).topk(k) keeps the k most-confident pseudo-label nodes
    # (utils/data/datasets.py). arxiv/cora/citeseer/pubmed leave pl_filter unset, so
    # they run with no gating at all; 0.8 is the value GLEM ships for its GCN recipe.
    # Swept rather than fixed, because a single operating point cannot distinguish
    # 'confidence gating does not help' from 'this keep-rate is wrong'.
    ARM_ARGS="--pl_filter=0.${ARM#conf_gate}" ;;
  sig_gate80|sig_gate90)
    # Exogenous-signal gate (A18): keep the top k% by GLANCE soft homophily when
    # the GNN teaches, by inverted kNN ambiguity when the LM teaches. Keep-rates
    # deliberately MATCH conf_gate80/90 so the difference isolates the signal with
    # shrinkage held fixed. Needs the cached signal arrays -- run analyze.py first.
    ARM_ARGS=""; export GLEM_PROBE_GATE="signal${ARM#sig_gate}" ;;
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
# Set by the oracle arms above; empty for every other arm.
export GLEM_PROBE_GATE="${GLEM_PROBE_GATE:-}"

echo "=== probe: $DATASET_STR arm=$ARM regime=$REGIME seed=$SEED -> $PROBE_DIR"
# shellcheck disable=SC2086
exec "$PY" "$ROOT/src/models/GLEM/trainGLEM.py" $GLEM_ARGS --seed="$SEED" $ARM_ARGS "$@"
