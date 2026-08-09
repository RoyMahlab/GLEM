#!/usr/bin/env bash
# Run one seed per GPU, concurrently. Each seed's arms run sequentially in its lane.
#
#   scripts/parallel_seed_sweep.sh arxiv                    # seeds 0,1 on GPUs 0,1
#   SEEDS="0 1" GPU_LIST="0 1" scripts/parallel_seed_sweep.sh arxiv
#   ARMS="published alpha0_li_T" scripts/parallel_seed_sweep.sh cora
#   DRY_RUN=1 scripts/parallel_seed_sweep.sh arxiv          # print the plan only
#
# WHY SEEDS AND NOT ARMS. Every EM working path is keyed by seed --
# `ModelConfig.f_prefix` is ".../seed{seed}{model_cf_str}", and `res_file`,
# `emi_file` and `glem_cfg_str` all derive from it -- so two seeds never touch the
# same file. Two ARMS of the same seed do collide: `oracle` and `oracle_random`
# differ only by an environment variable, so they produce an identical
# `glem_cfg_str` and would overwrite each other's `temp/glem_{lm,gnn}/...` and
# `em_info` pickle. Hence arms are sequential within a lane, seeds are parallel
# across lanes.
#
# PRETRAIN CACHES ARE SHARED ACROSS SEEDS (`temp/prt_{lm,gnn}/<dataset>/...` carry
# no seed). That is safe only when they already exist, which is the normal case --
# every run then just reads them. For a dataset that has never been pretrained,
# concurrent lanes would race to write them: set WARMUP=1 to run the first cell
# alone first, at the cost of one serialised run.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SEEDS=${SEEDS:-"0 1"}
GPU_LIST=${GPU_LIST:-"0 1"}
ARMS=${ARMS:-"oracle oracle_random"}
DRY_RUN=${DRY_RUN:-0}
WARMUP=${WARMUP:-0}
LOG_DIR="$ROOT/logs/probe"
MANIFEST="$LOG_DIR/manifest.tsv"
mkdir -p "$LOG_DIR"
[ -s "$MANIFEST" ] || printf 'config\tarm\tseed\texit\tseconds\tlog\n' > "$MANIFEST"

# Configs whose DATASET_STR collides with another config's; mirrors probe_sweep.sh.
declare -A VARIANT=([cornell]=revgat [texas]=revgat [washington]=revgat [wisconsin]=revgat)

CONFIGS=("$@")
[ ${#CONFIGS[@]} -eq 0 ] && { echo "usage: $0 <config> [<config>...]" >&2; exit 2; }
for cfg in "${CONFIGS[@]}"; do
  [ -f "$ROOT/configs/glem/$cfg.sh" ] || { echo "!! no such config: $cfg" >&2; exit 2; }
done

read -ra SEED_ARR <<< "$SEEDS"
read -ra GPU_ARR  <<< "$GPU_LIST"
if [ ${#SEED_ARR[@]} -gt ${#GPU_ARR[@]} ]; then
  echo "!! ${#SEED_ARR[@]} seeds but only ${#GPU_ARR[@]} GPU(s); give GPU_LIST one entry per seed" >&2
  exit 2
fi

echo "=== plan: ${#CONFIGS[@]} config(s) x ${#SEED_ARR[@]} seed(s) x $(wc -w <<< "$ARMS") arm(s)"
for i in "${!SEED_ARR[@]}"; do
  echo "  lane: seed ${SEED_ARR[$i]} -> GPU ${GPU_ARR[$i]}   arms (sequential): $ARMS"
done
[ "$DRY_RUN" = 1 ] && { echo "(dry run)"; exit 0; }

# One cell, appending its own manifest row. Short appends to a file opened O_APPEND
# are atomic under PIPE_BUF, so concurrent lanes do not interleave rows.
run_cell() {
  local cfg=$1 arm=$2 seed=$3 gpu=$4
  local log="$LOG_DIR/${cfg}__${arm}__seed${seed}.log"
  local t0=$SECONDS
  GLEM_PROBE_VARIANT="${VARIANT[$cfg]:-}" \
    "$ROOT/scripts/probe_run.sh" "$cfg" "$arm" standard "$seed" --gpus="$gpu" \
    > "$log" 2>&1
  local rc=$?
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$cfg" "$arm" "$seed" "$rc" "$((SECONDS - t0))" "$log" >> "$MANIFEST"
  echo "    [gpu $gpu] $cfg $arm seed=$seed -> exit $rc ($((SECONDS - t0))s)"
}

for cfg in "${CONFIGS[@]}"; do
  if [ "$WARMUP" = 1 ]; then
    # Serialise one cell so any missing pretrain is created before lanes fan out.
    set -- $ARMS
    echo "--- warmup (serial): $cfg $1 seed=${SEED_ARR[0]}"
    run_cell "$cfg" "$1" "${SEED_ARR[0]}" "${GPU_ARR[0]}"
  fi

  echo "--- $cfg: launching ${#SEED_ARR[@]} lane(s)"
  pids=()
  for i in "${!SEED_ARR[@]}"; do
    (
      for arm in $ARMS; do
        if [ "$WARMUP" = 1 ] && [ "$i" = 0 ]; then
          set -- $ARMS
          [ "$arm" = "$1" ] && continue   # already done in warmup
        fi
        run_cell "$cfg" "$arm" "${SEED_ARR[$i]}" "${GPU_ARR[$i]}"
      done
    ) &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "--- $cfg done"
done

echo "=== all done"
awk -F'\t' 'NR>1 && $4!=0 {n++} END {printf "failures in manifest: %d\n", n+0}' "$MANIFEST"
