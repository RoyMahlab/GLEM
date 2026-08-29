#!/usr/bin/env bash
# Run the whole matrix -- every arm, on every dataset, at every seed -- resumably,
# and split across machines.
#
#   scripts/run_all.sh                          # everything this machine owns
#   DRY_RUN=1 scripts/run_all.sh                # print the plan and the cost, run nothing
#   SHARD=1/3 scripts/run_all.sh                # machine 1 of 3
#   SEEDS="0 1 2" GPU_LIST="0 1" scripts/run_all.sh
#   scripts/run_all.sh cora pubmed              # named configs only
#   ONLY_ARMS="b30 b30_conf80" scripts/run_all.sh arxiv
#
# WHAT IT RUNS. scripts/matrix.sh, which is the single source of truth for which
# arms are registered on which configs. Arms are NOT run everywhere: A19/A21's
# label-feature arms are registered for arxiv only, A28's beta grid only for the
# configs that ship beta=0.05, and the WebKB GCN configs carry the 2a/2b control
# split that the RevGAT ones do not. "All arms on all datasets" means the matrix,
# not the cross product -- the cross product would run arms on datasets where
# EXPERIMENT.md says their sign is not interpretable.
#
# RESUMABLE. SKIP_DONE=1 by default: a cell whose archive already holds all four
# distillation events is skipped. Interrupt this script whenever; re-running it
# picks up where it stopped. That is also how a failed cell is retried -- fix the
# cause and re-run, and only the incomplete cells go again.
#
# HOW WORK IS SPLIT ACROSS MACHINES. SHARD=k/N assigns each machine a cost-balanced
# share, using measured per-run times (matrix.sh COST). Balance matters more than it
# looks: arxiv is ~92% of the total, so splitting by dataset would hand one machine
# seven weeks and the others an afternoon. The split is deterministic -- every
# machine computes the same assignment from the same matrix and simply keeps its own
# k -- so no coordination is needed beyond giving each machine a different k.
#
#   SHARD_UNIT=cell   (default) balances individual (config, arm, seed) cells.
#                     Requires each machine to have its OWN checkout: two machines
#                     running different arms of the same (config, seed) would
#                     collide, because arms that differ only by an env var share a
#                     glem_cfg_str and overwrite each other's temp/glem_{lm,gnn}
#                     working dirs and em_info pickle.
#   SHARD_UNIT=seed   keeps every (config, seed) whole on one machine, so it is safe
#                     even if the machines share a filesystem. Coarser, so less
#                     balanced when a heavy dataset has fewer seeds than machines.
#
# WITHIN ONE MACHINE. Seeds run in parallel, one lane per GPU; arms run sequentially
# inside a lane. Every EM working path is keyed by seed (ModelConfig.f_prefix is
# ".../seed{seed}{model_cf_str}"), so two seeds never touch the same file, while two
# arms of one seed do. This is the same rule parallel_seed_sweep.sh documents.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MATRIX_ROOT=$ROOT
# shellcheck source=/dev/null
source "$ROOT/scripts/matrix.sh"

SEEDS=${SEEDS:-"0 1 2"}
GPU_LIST=${GPU_LIST:-"0"}
ONLY_ARMS=${ONLY_ARMS:-}
SKIP_DONE=${SKIP_DONE:-1}
DRY_RUN=${DRY_RUN:-0}
SHARD=${SHARD:-}
SHARD_UNIT=${SHARD_UNIT:-cell}
# auto = serialise the first cell of a config only when its pretrain is absent.
WARMUP=${WARMUP:-auto}
PY=${PY:-$ROOT/.venv/bin/python}
PROBE_DIR=${GLEM_PROBE_DIR:-$ROOT/temp/probe_output}
LOG_DIR="$ROOT/logs/probe"
MANIFEST="$LOG_DIR/manifest.tsv"
PLAN="$LOG_DIR/shard_plan.tsv"
mkdir -p "$LOG_DIR"
[ -s "$MANIFEST" ] || printf 'config\tarm\tseed\texit\tseconds\tlog\n' > "$MANIFEST"

CONFIGS=("$@")
[ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=("${ALL_CONFIGS[@]}")

read -ra SEED_ARR <<< "$SEEDS"
read -ra GPU_ARR <<< "$GPU_LIST"

# ───────────────────────────────────────────────────────────── helpers

# Arms for a config, after the ONLY_ARMS filter.
arms_for() {
  local cfg=$1 arm want out=""
  for arm in ${ARMS[$cfg]:-}; do
    if [ -z "$ONLY_ARMS" ]; then
      out="$out $arm"
    else
      for want in $ONLY_ARMS; do [ "$arm" = "$want" ] && out="$out $arm"; done
    fi
  done
  echo "$out"
}

# True when this cell's archive already holds all 4 distillation events.
cell_done() {
  local ds f
  ds=$(archive_ds_for "$1")
  f="$PROBE_DIR/$ds/standard/$2/seed$3/steps.jsonl"
  [ -s "$f" ] && [ "$(grep -c . "$f")" -eq 4 ]
}

hms() { printf '%dh%02dm' $(($1 / 3600)) $((($1 % 3600) / 60)); }

# ───────────────────────────────────────────────────────── preflight

echo "=== preflight"
fail=0
[ -x "$PY" ] || { echo "!! no python at $PY (set PY=/path/to/python)" >&2; fail=1; }

for cfg in "${CONFIGS[@]}"; do
  if [ -z "${ARMS[$cfg]:-}" ]; then
    echo "!! '$cfg' is not in the matrix (scripts/matrix.sh)" >&2; fail=1; continue
  fi
  if [ ! -f "$ROOT/configs/glem/$cfg.sh" ]; then
    echo "!! configs/glem/$cfg.sh is missing" >&2; fail=1; continue
  fi
  # A28's grid assumes its b=0.05 row IS the published recipe.
  for arm in $(arms_for "$cfg"); do
    case "$arm" in
      b05_*|b30*|b80_*|beta_high*)
        assert_grid_beta "$cfg" || fail=1; break ;;
    esac
  done
done
[ "$fail" = 0 ] || { echo "preflight failed" >&2; exit 1; }

# Cached signal arrays, checked before anything trains rather than three hours in.
if ! "$PY" "$ROOT/scripts/prepare_signals.py" --check "${CONFIGS[@]}" >/dev/null 2>&1; then
  echo "!! cached signals are missing for the gated arms:"
  "$PY" "$ROOT/scripts/prepare_signals.py" --check "${CONFIGS[@]}" 2>&1 | sed 's/^/   /'
  echo "   build them first:  scripts/prepare_signals.py ${CONFIGS[*]}"
  exit 1
fi
echo "  configs: ${#CONFIGS[@]}   seeds: $SEEDS   gpus: $GPU_LIST   signals: present"

# ───────────────────────────────────────────── build the cell list

CELL_CFG=(); CELL_ARM=(); CELL_SEED=(); CELL_COST=(); CELL_UNIT=()
skipped=0
for cfg in "${CONFIGS[@]}"; do
  for arm in $(arms_for "$cfg"); do
    for seed in "${SEED_ARR[@]}"; do
      if [ "$SKIP_DONE" = 1 ] && cell_done "$cfg" "$arm" "$seed"; then
        skipped=$((skipped + 1)); continue
      fi
      CELL_CFG+=("$cfg"); CELL_ARM+=("$arm"); CELL_SEED+=("$seed")
      CELL_COST+=("${COST[$cfg]:-600}")
      if [ "$SHARD_UNIT" = seed ]; then
        CELL_UNIT+=("$cfg:$seed")
      else
        CELL_UNIT+=("$cfg:$arm:$seed")
      fi
    done
  done
done
n_cells=${#CELL_CFG[@]}
if [ "$n_cells" = 0 ]; then
  if [ "$skipped" = 0 ] && [ -n "$ONLY_ARMS" ]; then
    # Distinguishes "the work is finished" from "the filter matched nothing", which
    # look identical from the outside and mean opposite things.
    echo "=== nothing to run: no arm in ONLY_ARMS='$ONLY_ARMS' is registered on" \
         "${CONFIGS[*]} (see scripts/matrix.sh)"
  else
    echo "=== nothing to run: $skipped cell(s) already complete"
  fi
  exit 0
fi

# ───────────────────────────────────────────── shard, cost-balanced
#
# Longest-processing-time-first: sort units by cost descending, assign each to the
# lightest bin so far. Deterministic given the matrix, so every machine derives the
# same assignment independently and keeps only its own.
declare -A MINE
if [ -n "$SHARD" ]; then
  SHARD_K=${SHARD%%/*}; SHARD_N=${SHARD##*/}
  case "$SHARD_K$SHARD_N" in *[!0-9]*) echo "!! SHARD must be k/N" >&2; exit 2 ;; esac
  if [ "$SHARD_K" -lt 1 ] || [ "$SHARD_K" -gt "$SHARD_N" ]; then
    echo "!! SHARD=$SHARD out of range" >&2; exit 2
  fi

  declare -A UNIT_COST
  for i in $(seq 0 $((n_cells - 1))); do
    u=${CELL_UNIT[$i]}
    UNIT_COST[$u]=$(( ${UNIT_COST[$u]:-0} + CELL_COST[$i] ))
  done

  declare -a BIN_LOAD
  for b in $(seq 0 $((SHARD_N - 1))); do BIN_LOAD[$b]=0; done
  declare -A UNIT_BIN
  # Ties broken by unit name so the order is identical on every machine.
  while read -r cost unit; do
    best=0
    for b in $(seq 1 $((SHARD_N - 1))); do
      [ "${BIN_LOAD[$b]}" -lt "${BIN_LOAD[$best]}" ] && best=$b
    done
    UNIT_BIN[$unit]=$best
    BIN_LOAD[$best]=$(( BIN_LOAD[$best] + cost ))
  done < <(for u in "${!UNIT_COST[@]}"; do echo "${UNIT_COST[$u]} $u"; done \
           | sort -k1,1nr -k2,2)

  for i in $(seq 0 $((n_cells - 1))); do
    [ "${UNIT_BIN[${CELL_UNIT[$i]}]}" = "$((SHARD_K - 1))" ] && MINE[$i]=1
  done

  echo "=== shard $SHARD_K/$SHARD_N (unit=$SHARD_UNIT), balance across machines:"
  for b in $(seq 0 $((SHARD_N - 1))); do
    mark=" "; [ "$b" = "$((SHARD_K - 1))" ] && mark="*"
    echo "   $mark machine $((b + 1)): $(hms "${BIN_LOAD[$b]}") of GPU time"
  done
else
  for i in $(seq 0 $((n_cells - 1))); do MINE[$i]=1; done
fi

mine_n=0; mine_cost=0
: > "$PLAN"
for i in $(seq 0 $((n_cells - 1))); do
  [ -n "${MINE[$i]:-}" ] || continue
  mine_n=$((mine_n + 1)); mine_cost=$((mine_cost + CELL_COST[i]))
  printf '%s\t%s\t%s\t%s\n' \
    "${CELL_CFG[$i]}" "${CELL_ARM[$i]}" "${CELL_SEED[$i]}" "${CELL_COST[$i]}" >> "$PLAN"
done

echo "=== this machine: $mine_n cell(s), ~$(hms "$mine_cost") of GPU time" \
     "($skipped already complete, $n_cells outstanding in total)"
echo "    plan written to $PLAN"

if [ "$DRY_RUN" = 1 ]; then
  echo
  printf '%-16s %-18s %6s %10s\n' CONFIG ARM SEED EST
  while IFS=$'\t' read -r c a s k; do
    printf '%-16s %-18s %6s %10s\n' "$c" "$a" "$s" "$(hms "$k")"
  done < "$PLAN"
  echo "(dry run; nothing executed)"
  exit 0
fi

# ───────────────────────────────────────────────────────── execute

run_cell() {
  local cfg=$1 arm=$2 seed=$3 gpu=$4
  local log="$LOG_DIR/${cfg}__${arm}__seed${seed}.log"
  local t0=$SECONDS rc
  GLEM_PROBE_VARIANT="${VARIANT[$cfg]:-}" GLEM_PROBE_DIR="$PROBE_DIR" PY="$PY" \
    "$ROOT/scripts/probe_run.sh" "$cfg" "$arm" standard "$seed" --gpus="$gpu" \
    > "$log" 2>&1
  rc=$?
  # Short appends to an O_APPEND file are atomic under PIPE_BUF, so concurrent
  # lanes do not interleave rows.
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$cfg" "$arm" "$seed" "$rc" "$((SECONDS - t0))" "$log" >> "$MANIFEST"
  if [ "$rc" = 0 ]; then
    echo "    [gpu $gpu] $cfg $arm seed=$seed  ok ($((SECONDS - t0))s)"
  else
    echo "    [gpu $gpu] $cfg $arm seed=$seed  !! exit $rc ($((SECONDS - t0))s) -- $log"
  fi
}

# True when this config has never been pretrained on this machine. Concurrent lanes
# would otherwise race to write temp/prt_{lm,gnn}/<dataset>, which carry no seed.
needs_warmup() {
  local ds
  # shellcheck disable=SC1090
  ds=$(. "$ROOT/configs/glem/$1.sh" >/dev/null 2>&1; echo "$DATASET_STR")
  [ ! -d "$ROOT/temp/prt_lm/$ds" ] || [ ! -d "$ROOT/temp/prt_gnn/$ds" ]
}

started=$SECONDS
for cfg in "${ALL_CONFIGS[@]}"; do
  # Cells of this config that belong to this machine, grouped by seed.
  declare -A LANE=()
  any=0
  for i in $(seq 0 $((n_cells - 1))); do
    [ -n "${MINE[$i]:-}" ] || continue
    [ "${CELL_CFG[$i]}" = "$cfg" ] || continue
    LANE[${CELL_SEED[$i]}]="${LANE[${CELL_SEED[$i]}]:-} ${CELL_ARM[$i]}"
    any=1
  done
  [ "$any" = 1 ] || continue

  warm=0
  case "$WARMUP" in
    1) warm=1 ;;
    auto) needs_warmup "$cfg" && warm=1 ;;
  esac

  echo "--- $cfg: ${#LANE[@]} lane(s)$([ "$warm" = 1 ] && echo ', warmup first')"

  if [ "$warm" = 1 ]; then
    # Serialise one cell so the missing pretrain is created exactly once.
    first_seed=$(for s in "${!LANE[@]}"; do echo "$s"; done | sort -n | head -1)
    first_arm=$(echo "${LANE[$first_seed]}" | awk '{print $1}')
    echo "  warmup (serial): $cfg $first_arm seed=$first_seed"
    run_cell "$cfg" "$first_arm" "$first_seed" "${GPU_ARR[0]}"
    LANE[$first_seed]=$(echo "${LANE[$first_seed]}" | cut -d' ' -f2-)
  fi

  pids=(); li=0
  for seed in $(for s in "${!LANE[@]}"; do echo "$s"; done | sort -n); do
    arms_here=$(echo "${LANE[$seed]}" | xargs)
    [ -n "$arms_here" ] || continue
    gpu=${GPU_ARR[$((li % ${#GPU_ARR[@]}))]}
    li=$((li + 1))
    (
      for arm in $arms_here; do run_cell "$cfg" "$arm" "$seed" "$gpu"; done
    ) &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "--- $cfg done"
done

echo "=== finished in $(hms $((SECONDS - started)))"
awk -F'\t' 'NR>1 && $4!=0 {n++} END {printf "failures in manifest: %d\n", n+0}' "$MANIFEST"
