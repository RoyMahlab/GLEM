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
SEEDS=${SEEDS:-"0"}
# Restrict to a subset of arms, e.g. to run an arm added later without repeating
# cells already on disk:  ONLY_ARMS="published_li_F" scripts/probe_sweep.sh cornell_gcn
# Configs that do not define the named arm are skipped.
ONLY_ARMS=${ONLY_ARMS:-}
# Skip cells whose archive already holds all 4 distillation events, so an
# interrupted sweep resumes instead of repeating completed work.
SKIP_DONE=${SKIP_DONE:-0}
# List the cells that would run, then exit without training.
DRY_RUN=${DRY_RUN:-0}
LOG_DIR="$ROOT/logs/probe"
MANIFEST="$LOG_DIR/manifest.tsv"
mkdir -p "$LOG_DIR"

# config -> arms. The two control arms differ only when gnn_label_input=T routes
# the teacher's y_hat into GNN input features; the RevGAT configs already set F,
# so 2a and 2b coincide there and only one control is run (section 8).
declare -A ARMS=(
  [cornell_gcn]="published alpha0_li_T alpha0_li_F published_li_F"
  [texas_gcn]="published alpha0_li_T alpha0_li_F published_li_F"
  [washington_gcn]="published alpha0_li_T alpha0_li_F published_li_F"
  [wisconsin_gcn]="published alpha0_li_T alpha0_li_F published_li_F"
  [cora]="published alpha0_li_T"
  [citeseer]="published alpha0_li_T"
  [pubmed]="published alpha0_li_T"
  [arxiv]="published alpha0_li_T"
  # RevGAT on the WebKB sets. These configs set gnn_label_input=F, so 2a/2b
  # coincide and published_li_F would equal published -- two arms only, as for
  # the other RevGAT datasets.
  [cornell]="published alpha0_li_T"
  [texas]="published alpha0_li_T"
  [washington]="published alpha0_li_T"
  [wisconsin]="published alpha0_li_T"
)

# Configs whose DATASET_STR collides with another config's. The tag is appended
# to the archive's dataset directory (cornell_TAG+revgat) so the two backbones do
# not overwrite each other. Empty for everything else.
declare -A VARIANT=(
  [cornell]=revgat
  [texas]=revgat
  [washington]=revgat
  [wisconsin]=revgat
)
WEBKB_REVGAT=(cornell texas washington wisconsin)
CHEAP=(cornell_gcn texas_gcn washington_gcn wisconsin_gcn cora citeseer pubmed)

CONFIGS=("$@")
[ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=("${CHEAP[@]}")

[ -s "$MANIFEST" ] || printf 'config\tarm\tseed\texit\tseconds\tlog\n' > "$MANIFEST"

# Arms for a config, after the ONLY_ARMS filter.
arms_for() {
  local cfg=$1 arm out=""
  for arm in ${ARMS[$cfg]}; do
    if [ -z "$ONLY_ARMS" ]; then out="$out $arm"; else
      for want in $ONLY_ARMS; do [ "$arm" = "$want" ] && out="$out $arm"; done
    fi
  done
  echo "$out"
}

# True when this cell's archive already has all 4 distillation events.
cell_done() {
  local ds arm=$2 seed=$3
  # shellcheck disable=SC1090
  ds=$(. "$ROOT/configs/glem/$1.sh" >/dev/null 2>&1; echo "$DATASET_STR")
  local f="$ROOT/temp/probe_output/$ds/standard/$arm/seed$seed/steps.jsonl"
  [ -s "$f" ] && [ "$(grep -c . "$f")" -eq 4 ]
}

total=0
for cfg in "${CONFIGS[@]}"; do
  for arm in $(arms_for "$cfg"); do
    for seed in $SEEDS; do
      [ "$SKIP_DONE" = 1 ] && cell_done "$cfg" "$arm" "$seed" && continue
      total=$((total + 1))
    done
  done
done
echo "=== $total run(s) queued: ${CONFIGS[*]}"

i=0
for cfg in "${CONFIGS[@]}"; do
  for arm in $(arms_for "$cfg"); do
    for seed in $SEEDS; do
      if [ "$SKIP_DONE" = 1 ] && cell_done "$cfg" "$arm" "$seed"; then
        echo "--- skip (already complete): $cfg $arm seed=$seed"
        continue
      fi
      i=$((i + 1))
      log="$LOG_DIR/${cfg}__${arm}__seed${seed}.log"
      echo "--- [$i/$total] $cfg $arm seed=$seed -> $log"
      if [ "$DRY_RUN" = 1 ]; then continue; fi
      t0=$SECONDS
      GLEM_PROBE_VARIANT="${VARIANT[$cfg]:-}" \
        "$ROOT/scripts/probe_run.sh" "$cfg" "$arm" standard "$seed" --gpus=0 \
        > "$log" 2>&1
      rc=$?
      # shellcheck disable=SC2317
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$cfg" "$arm" "$seed" "$rc" "$((SECONDS - t0))" "$log" >> "$MANIFEST"
      [ "$rc" -ne 0 ] && echo "    !! exit $rc (continuing) -- see $log"
    done
  done
done
echo "=== sweep done; $(( $(wc -l < "$MANIFEST") - 1 )) run(s) in $MANIFEST"
awk -F'\t' 'NR>1 && $4!=0 {n++} END {printf "failures: %d\n", n+0}' "$MANIFEST"
