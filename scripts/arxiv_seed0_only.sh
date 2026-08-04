#!/usr/bin/env bash
# Reduce the running arxiv sweep to seed 0 of both arms (EXPERIMENT.md A7).
#
# The sweep iterates arm-major (published s0,s1,s2 then alpha0_li_T s0,s1,s2), so
# simply stopping after two runs would give published seeds 0 and 1 rather than
# seed 0 of each arm. Instead: let the in-flight published/seed0 run finish, kill
# the queue, discard any partially-written run dir the kill leaves behind, then run
# the one remaining cell directly.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
MANIFEST=logs/probe/manifest.tsv
ARCHIVE=temp/probe_output/arxiv_TA/standard

echo "[$(date '+%H:%M')] waiting for arxiv published/seed0 to complete..."
until awk -F'\t' '$1=="arxiv" && $2=="published" && $3=="0"' "$MANIFEST" | grep -q .; do
  pgrep -f "probe_sweep.sh arxiv" >/dev/null || { echo "sweep gone before seed0 finished"; break; }
  sleep 60
done
echo "[$(date '+%H:%M')] published/seed0 done:"
awk -F'\t' 'NR==1 || ($1=="arxiv")' "$MANIFEST" | cut -f1-5

pkill -f "probe_sweep.sh arxiv" 2>/dev/null && echo "[$(date '+%H:%M')] sweep queue stopped"
sleep 5
# The kill can land mid-run; a run dir without all 4 distillation events would give
# the analysis partial rows, so drop it and let it be re-run if ever wanted.
for d in "$ARCHIVE"/*/seed*; do
  [ -d "$d" ] || continue
  n=$(grep -c . "$d/steps.jsonl" 2>/dev/null || echo 0)
  if [ "$n" -lt 4 ]; then
    echo "[$(date '+%H:%M')] discarding incomplete archive ($n/4 steps): $d"
    rm -rf "$d"
  fi
done

echo "[$(date '+%H:%M')] running arxiv alpha0_li_T seed0 (the control cell)"
log=logs/probe/arxiv__alpha0_li_T__seed0.log
t0=$SECONDS
./scripts/probe_run.sh arxiv alpha0_li_T standard 0 --gpus=0 > "$log" 2>&1
rc=$?
printf 'arxiv\talpha0_li_T\t0\t%s\t%s\t%s\n' "$rc" "$((SECONDS - t0))" "$log" >> "$MANIFEST"
echo "[$(date '+%H:%M')] control cell exit=$rc"
awk -F'\t' 'NR==1 || ($1=="arxiv")' "$MANIFEST" | cut -f1-5
