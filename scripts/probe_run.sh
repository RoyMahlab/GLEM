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
  published_li_T)
    # A19: published alpha/beta, but the teacher's y_hat IS concatenated onto the GNN's
    # input features. The mirror of published_li_F, and the only way to reach the
    # feature channel on configs that ship gnn_label_input=F (every upstream config).
    # node_feature() runs only at the M-step, so this can affect lm->gnn alone --
    # gnn->lm is the built-in placebo.
    ARM_ARGS="--gnn_label_input=T" ;;
  alpha0_li_T_only)
    # A19 attribution arm: pseudo-labels reach the student ONLY through input
    # features, never through the loss. Note the existing alpha0_li_T does NOT force
    # T -- it inherits the config, which is F on arxiv -- so it is not this arm.
    ARM_ARGS="--lm_pl_weight=0 --gnn_pl_weight=0 --gnn_label_input=T" ;;
  teacher_consistent)
    # A21: li=T, but the label feature carries the TEACHER's prediction on train nodes
    # too instead of the gold one-hot. Closes the train/inference reliability gap in
    # that channel (arxiv: 1.000 vs 0.755 as shipped -> 0.750 vs 0.755). Gold labels
    # stay in the loss; only the readable shortcut is removed.
    ARM_ARGS="--gnn_label_input=T"; export GLEM_PROBE_LABELFEAT=teacher_consistent ;;
  mask_pseudo)
    # A21: li=T with the teacher's entries zeroed, gold kept on train nodes. WIDENS
    # the reliability gap (1.000 vs nothing), so it is the least likely of the three
    # to help -- registered to test that prediction, not because it is expected to win.
    ARM_ARGS="--gnn_label_input=T"; export GLEM_PROBE_LABELFEAT=mask_pseudo ;;
  mask_train)
    # A21: li=T with the gold entry zeroed on train nodes (UniMP-style self-label
    # masking), teacher's prediction kept elsewhere.
    ARM_ARGS="--gnn_label_input=T"; export GLEM_PROBE_LABELFEAT=mask_train ;;
  unimp_mask)
    # A23: li=T with gold kept on a random half of train nodes and everything else
    # zeroed. Matched channel (gold-or-empty in training and at inference) that still
    # carries real labels. The one arm that could plausibly beat `published`.
    ARM_ARGS="--gnn_label_input=T"; export GLEM_PROBE_LABELFEAT=unimp_mask ;;
  beta_high)
    # A23: the LOSS channel at alpha-level weight. Section 13.2 argues the loss channel
    # is harmless because beta=0.05 down-weights a teacher that is worse than its
    # student; nothing has ever varied beta to check. gnn_label_input untouched, so on
    # arxiv this is li=F and the feature channel is absent.
    ARM_ARGS="--gnn_pl_weight=0.8" ;;
  beta_high_sig80)
    # A27: beta=0.8 AND the M-step ambiguity gate at 80% keep. Comparator is `published`
    # (beta=0.05, ungated), not beta_high -- the question is whether selective admission
    # beats GLEM's uniform down-weighting, not whether it repairs damage we introduced.
    ARM_ARGS="--gnn_pl_weight=0.8"; export GLEM_PROBE_GATE="signal80:lm" ;;
  beta_high_rand80)
    # A27 control: beta=0.8 with the SAME 80% keep-rate, chosen at random. Without it,
    # beta_high_sig80 confounds which nodes were kept with how many.
    ARM_ARGS="--gnn_pl_weight=0.8"; export GLEM_PROBE_GATE="random80:lm" ;;
  b05_rand80)
    # A28 grid (beta=0.05, random). beta stays at the published 0.05; the M-step
    # pseudo-label set is cut to a random 80%. Rate-matched control for b05 gates.
    ARM_ARGS=""; export GLEM_PROBE_GATE="random80:lm" ;;
  b30)
    # A28 grid (beta=0.3, ungated). The middle exposure point; without it the beta
    # axis is a two-point comparison rather than a dose-response.
    ARM_ARGS="--gnn_pl_weight=0.3" ;;
  b30_conf80)
    ARM_ARGS="--gnn_pl_weight=0.3 --pl_filter=0.8" ;;
  b30_sig80gnn)
    ARM_ARGS="--gnn_pl_weight=0.3"; export GLEM_PROBE_GATE="signal80:gnn" ;;
  b30_sig80lm)
    ARM_ARGS="--gnn_pl_weight=0.3"; export GLEM_PROBE_GATE="signal80:lm" ;;
  b30_rand80)
    ARM_ARGS="--gnn_pl_weight=0.3"; export GLEM_PROBE_GATE="random80:lm" ;;
  b80_conf80)
    ARM_ARGS="--gnn_pl_weight=0.8 --pl_filter=0.8" ;;
  b80_sig80gnn)
    # beta=0.8 with the E-step homophily gate. Note beta weights the M-step, so this
    # cell tests whether cleaning the GNN teacher matters at high M-step exposure --
    # an indirect interaction through the EM loop, not a direct crossing.
    ARM_ARGS="--gnn_pl_weight=0.8"; export GLEM_PROBE_GATE="signal80:gnn" ;;
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
    # Exogenous-signal gate (A18), BOTH teachers: keep the top k% by GLANCE soft
    # homophily when the GNN teaches, by inverted kNN ambiguity when the LM teaches.
    # Keep-rates deliberately MATCH conf_gate80/90 so the difference isolates the
    # signal with shrinkage held fixed. Needs the cached signal arrays from
    # analyze.py, and fails loudly rather than silently degrading if they are absent.
    ARM_ARGS=""; export GLEM_PROBE_GATE="signal${ARM#sig_gate}" ;;
  sig_gate80_gnn|sig_gate90_gnn)
    # Fix the GNN teacher ONLY: gate the E-step (GNN teaches the LM), leave the
    # M-step exactly as published. Isolates how much of the combined arm's effect
    # comes from cleaning the GNN's pseudo-labels.
    _k="${ARM#sig_gate}"; ARM_ARGS=""; export GLEM_PROBE_GATE="signal${_k%_gnn}:gnn" ;;
  sig_gate80_lm|sig_gate90_lm)
    # Fix the LM teacher ONLY: gate the M-step (LM teaches the GNN), leave the
    # E-step exactly as published.
    _k="${ARM#sig_gate}"; ARM_ARGS=""; export GLEM_PROBE_GATE="signal${_k%_lm}:lm" ;;
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
# Set by the A21 arms above; empty for every other arm.
export GLEM_PROBE_LABELFEAT="${GLEM_PROBE_LABELFEAT:-}"

echo "=== probe: $DATASET_STR arm=$ARM regime=$REGIME seed=$SEED -> $PROBE_DIR"
# shellcheck disable=SC2086
exec "$PY" "$ROOT/src/models/GLEM/trainGLEM.py" $GLEM_ARGS --seed="$SEED" $ARM_ARGS "$@"
