#!/usr/bin/env bash
# Sweep the standard-split grid of EXPERIMENT.md section 8 sequentially.
#
#   scripts/probe_sweep.sh            # the cheap datasets (arxiv held back)
#   scripts/probe_sweep.sh arxiv      # named configs only
#
# Sequential on purpose. Runs of the same dataset share GLEM's pretrain
# checkpoint paths (MNT_TEMP_DIR/prt_{lm,gnn}/<dataset>/..., which carry neither
# arm nor seed), so two concurrent first-runs of one dataset would both see a
# missing checkpoint and race to write it. Their per-EM artefacts do not collide --
# glem_cfg_str includes both seed and pl_weight -- but the pretrain does. The GNN
# trainer also pins itself to cuda:0 regardless of --gpus (gnn_utils._exp_init),
# so a second stream would not get its own GPU anyway.
#
# Every run's exit status, wall time and log path is appended to
# logs/probe/manifest.tsv; a failure is recorded and the sweep continues.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SEEDS=${SEEDS:-"0 1 2"}
LOG_DIR="$ROOT/logs/probe"
MANIFEST="$LOG_DIR/manifest.tsv"
mkdir -p "$LOG_DIR"

# config -> arms. The two control arms differ only when gnn_label_input=T routes
# the teacher's y_hat into GNN input features; the RevGAT configs already set F,
# so 2a and 2b coincide there and only one control is run (section 8).
declare -A ARMS=(
  [cornell_gcn]="published alpha0_li_T alpha0_li_F"
  [texas_gcn]="published alpha0_li_T alpha0_li_F"
  [washington_gcn]="published alpha0_li_T alpha0_li_F"
  [wisconsin_gcn]="published alpha0_li_T alpha0_li_F"
  [cora]="published alpha0_li_T"
  [citeseer]="published alpha0_li_T"
  [pubmed]="published alpha0_li_T"
  [arxiv]="published alpha0_li_T"
)
CHEAP=(cornell_gcn texas_gcn washington_gcn wisconsin_gcn cora citeseer pubmed)

CONFIGS=("$@")
[ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=("${CHEAP[@]}")

[ -s "$MANIFEST" ] || printf 'config\tarm\tseed\texit\tseconds\tlog\n' > "$MANIFEST"

total=0
for cfg in "${CONFIGS[@]}"; do
  for arm in ${ARMS[$cfg]}; do
    for seed in $SEEDS; do total=$((total + 1)); done
  done
done
echo "=== $total run(s) queued: ${CONFIGS[*]}"

i=0
for cfg in "${CONFIGS[@]}"; do
  for arm in ${ARMS[$cfg]}; do
    for seed in $SEEDS; do
      i=$((i + 1))
      log="$LOG_DIR/${cfg}__${arm}__seed${seed}.log"
      echo "--- [$i/$total] $cfg $arm seed=$seed -> $log"
      t0=$SECONDS
      "$ROOT/scripts/probe_run.sh" "$cfg" "$arm" standard "$seed" --gpus=0 \
        > "$log" 2>&1
      rc=$?
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$cfg" "$arm" "$seed" "$rc" "$((SECONDS - t0))" "$log" >> "$MANIFEST"
      [ "$rc" -ne 0 ] && echo "    !! exit $rc (continuing) -- see $log"
    done
  done
done
echo "=== sweep done; $(( $(wc -l < "$MANIFEST") - 1 )) run(s) in $MANIFEST"
awk -F'\t' 'NR>1 && $4!=0 {n++} END {printf "failures: %d\n", n+0}' "$MANIFEST"
