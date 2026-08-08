#!/usr/bin/env bash
# Perfect-teacher (oracle) sweep, on the RevGAT configs only.
#
#   scripts/perfect_teacher_sweep.sh                 # the 7 cheap datasets
#   scripts/perfect_teacher_sweep.sh arxiv           # named configs only
#   DRY_RUN=1 scripts/perfect_teacher_sweep.sh       # list cells, train nothing
#
# Runs two arms per cell (EXPERIMENT.md amendment A14):
#
#   oracle         pseudo-label set restricted to nodes the teacher gets RIGHT.
#                  Uses ground truth, so it is an upper bound on any per-node gate,
#                  not a deployable method.
#   oracle_random  drops the same NUMBER of pseudo-labels at random. Not optional:
#                  without it, oracle-vs-published confounds "removed wrong labels"
#                  with "trained on less pseudo-data".
#
# **Compare oracle against oracle_random, never against published.**
#
# Every config named here is RevGAT. That is the whole point of the file: the
# `*_gcn.sh` configs are the GCN variants of the four WebKB sets and are deliberately
# excluded, so this sweep is single-backbone. Note the naming is inverted between the
# two -- the config suffix marks GCN (`cornell_gcn.sh`) while the archive suffix marks
# RevGAT (`cornell_TAG+revgat`), the latter set by probe_sweep.sh's VARIANT map.
#
# Delegates to probe_sweep.sh so the manifest, per-run logs, SKIP_DONE and DRY_RUN all
# behave identically to every other sweep.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# RevGAT configs. arxiv is excluded from the default set on cost grounds -- ~35 h per
# run against 2-78 min for the others (see logs/probe/manifest.tsv) -- so it is opt-in
# by naming it explicitly.
REVGAT_CHEAP=(cornell texas washington wisconsin cora citeseer pubmed)

CONFIGS=("$@")
[ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=("${REVGAT_CHEAP[@]}")

# Guard: refuse a GCN config, which would silently make the sweep multi-backbone.
for cfg in "${CONFIGS[@]}"; do
  case "$cfg" in
    *_gcn)
      echo "!! $cfg is a GCN config; this sweep is RevGAT-only." >&2
      echo "   Use scripts/probe_sweep.sh directly if you want the GCN variants." >&2
      exit 2 ;;
  esac
  if [ ! -f "$ROOT/configs/glem/$cfg.sh" ]; then
    echo "!! no such config: configs/glem/$cfg.sh" >&2
    exit 2
  fi
done

echo "=== perfect-teacher sweep (RevGAT): ${CONFIGS[*]}"
SEEDS="${SEEDS:-0 1 2}" ONLY_ARMS="oracle oracle_random" \
  exec "$ROOT/scripts/probe_sweep.sh" "${CONFIGS[@]}"
