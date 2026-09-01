#!/usr/bin/env bash
# Pull finished cells from the worker machines into this machine's archive.
#
#   scripts/pull_results.sh user@host2 user@host3 user@host4
#   DRY_RUN=1 scripts/pull_results.sh user@host2        # report only, copy nothing
#   REMOTE_ROOT=/data/GLEM scripts/pull_results.sh host2
#
# WHY NOT A PLAIN rsync OF temp/probe_output. Two reasons, both of which would
# corrupt the analysis quietly rather than loudly:
#
#   1. A worker is still running. Its in-flight cell has a partially written
#      steps.jsonl and a logits/ directory missing its last iteration. Copied in, that
#      cell looks like any other to analyze.py -- it globs steps.jsonl and reads
#      whatever logits are there. Only cells whose steps.jsonl holds all four
#      distillation events are transferred.
#   2. A cell that exists on both sides. Shards are disjoint by construction, so this
#      should never happen; if it does, something is wrong with how the machines were
#      launched (mismatched SEEDS, or one machine run unsharded). Overwriting would
#      hide that. Collisions are reported and skipped unless OVERWRITE=1.
#
# The transfer is per-cell and resumable: interrupt it and run it again.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROBE_DIR=${GLEM_PROBE_DIR:-$ROOT/temp/probe_output}
# Where the repo lives on the workers. Same path as here unless told otherwise.
REMOTE_ROOT=${REMOTE_ROOT:-$ROOT}
DRY_RUN=${DRY_RUN:-0}
OVERWRITE=${OVERWRITE:-0}
SSH_OPTS=${SSH_OPTS:-"-o BatchMode=yes -o ConnectTimeout=10"}

HOSTS=("$@")
if [ ${#HOSTS[@]} -eq 0 ]; then
  echo "usage: $0 <user@host> [<user@host>...]" >&2
  echo "  the machines running SHARD=2/N, 3/N, ... -- not this one" >&2
  exit 2
fi

command -v rsync >/dev/null || { echo "!! rsync not found on this machine" >&2; exit 1; }

# A cell is complete when steps.jsonl holds all four distillation events. Emitted as
# archive-relative directory paths, which is what rsync --files-from wants.
# shellcheck disable=SC2016
REMOTE_LIST_CMD='
  d="$0"
  [ -d "$d/temp/probe_output" ] || { echo "__NO_ARCHIVE__"; exit 0; }
  cd "$d/temp/probe_output" || exit 0
  find . -name steps.jsonl -print 2>/dev/null | while read -r f; do
    n=$(grep -c . "$f" 2>/dev/null)
    [ "$n" = 4 ] && dirname "$f"
  done
'

total_new=0 total_dup=0 total_hosts=0
for host in "${HOSTS[@]}"; do
  echo "=== $host"
  remote_cells=$(ssh $SSH_OPTS "$host" "sh -s '$REMOTE_ROOT'" <<< "$REMOTE_LIST_CMD" 2>/dev/null)
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "  !! cannot reach $host (ssh exit $rc) -- skipping" >&2
    continue
  fi
  if grep -q '__NO_ARCHIVE__' <<< "$remote_cells"; then
    echo "  !! no archive at $REMOTE_ROOT/temp/probe_output -- wrong REMOTE_ROOT?" >&2
    continue
  fi
  remote_cells=$(grep -v '^$' <<< "$remote_cells" | sed 's|^\./||' | sort)
  n_remote=$(grep -c . <<< "$remote_cells")
  [ -z "$remote_cells" ] && n_remote=0

  new=""; dup=""
  while IFS= read -r cell; do
    [ -z "$cell" ] && continue
    local_steps="$PROBE_DIR/$cell/steps.jsonl"
    if [ -s "$local_steps" ] && [ "$(grep -c . "$local_steps")" = 4 ]; then
      dup="$dup$cell"$'\n'
    else
      new="$new$cell"$'\n'
    fi
  done <<< "$remote_cells"

  n_new=$(grep -c . <<< "$new"); [ -z "${new//[$'\n']/}" ] && n_new=0
  n_dup=$(grep -c . <<< "$dup"); [ -z "${dup//[$'\n']/}" ] && n_dup=0
  echo "  $n_remote complete cell(s) there: $n_new new here, $n_dup already complete here"

  if [ "$n_dup" -gt 0 ]; then
    # Disjoint shards mean this should be zero. Non-zero is a launch problem worth
    # seeing, not a merge problem worth silently resolving.
    echo "  !! $n_dup cell(s) are complete on BOTH machines. Shards should be disjoint --"
    echo "     check that every machine used the same SEEDS and a distinct SHARD=k/N."
    grep -c . <<< "$dup" >/dev/null && sed 's|^|       |' <<< "$dup" | head -5
    [ "$n_dup" -gt 5 ] && echo "       ... and $((n_dup - 5)) more"
    if [ "$OVERWRITE" = 1 ]; then
      echo "     OVERWRITE=1: taking the remote copy anyway"
      new="$new$dup"
      n_new=$((n_new + n_dup))
    else
      echo "     keeping the local copy; re-run with OVERWRITE=1 to take theirs"
    fi
  fi
  total_dup=$((total_dup + n_dup))

  if [ "$n_new" = 0 ]; then
    echo "  nothing to pull"
    total_hosts=$((total_hosts + 1))
    continue
  fi

  if [ "$DRY_RUN" = 1 ]; then
    echo "  would pull:"
    sed 's|^|       |' <<< "$new" | head -10
    [ "$n_new" -gt 10 ] && echo "       ... and $((n_new - 10)) more"
    total_new=$((total_new + n_new))
    total_hosts=$((total_hosts + 1))
    continue
  fi

  list=$(mktemp)
  grep -v '^$' <<< "$new" > "$list"
  mkdir -p "$PROBE_DIR"
  # --files-from with directory entries copies each cell whole. -r is required
  # because --files-from disables recursion by default.
  rsync -a -r --info=stats2 --files-from="$list" \
        "$host:$REMOTE_ROOT/temp/probe_output/" "$PROBE_DIR/" \
    && echo "  pulled $n_new cell(s)" || echo "  !! rsync failed for $host" >&2
  rm -f "$list"
  total_new=$((total_new + n_new))
  total_hosts=$((total_hosts + 1))

  # The manifest is append-only timing data, one row per run. Worth having for the
  # cost model in matrix.sh; harmless to duplicate rows, so it is merged separately
  # and deduplicated.
  if ssh $SSH_OPTS "$host" "test -s '$REMOTE_ROOT/logs/probe/manifest.tsv'" 2>/dev/null; then
    tmp=$(mktemp)
    ssh $SSH_OPTS "$host" "cat '$REMOTE_ROOT/logs/probe/manifest.tsv'" > "$tmp" 2>/dev/null
    mkdir -p "$ROOT/logs/probe"
    touch "$ROOT/logs/probe/manifest.tsv"
    # Keep the header once, then the union of data rows.
    { head -1 "$ROOT/logs/probe/manifest.tsv" 2>/dev/null || head -1 "$tmp";
      { tail -n +2 "$ROOT/logs/probe/manifest.tsv"; tail -n +2 "$tmp"; } \
        | grep -v '^$' | sort -u; } > "$ROOT/logs/probe/manifest.merged"
    mv "$ROOT/logs/probe/manifest.merged" "$ROOT/logs/probe/manifest.tsv"
    rm -f "$tmp"
    echo "  manifest merged"
  fi
done

echo
echo "=== $total_hosts host(s), $total_new cell(s) pulled, $total_dup collision(s)"
if [ "$DRY_RUN" = 1 ]; then
  echo "(dry run; nothing copied)"
  exit 0
fi
echo "local archive now holds $(find "$PROBE_DIR" -name steps.jsonl 2>/dev/null | wc -l) cell(s)"
echo "next:  $ROOT/scripts/run_all.sh   # re-plan, the pulled cells now count as done"
echo "       python src/probe/analyze.py --probe-dir $PROBE_DIR --out $ROOT/temp/probe_analysis"
