#!/usr/bin/env bash
# Train GLEM (DeBERTa-base + RevGAT) on every text-attributed graph, logging to
# wandb (project GLEM-TAG). Each dataset's hyperparameters live in
# configs/glem/<dataset>.sh; this runner sources each and launches trainGLEM.
#
# Runs sequentially (one GPU each). Per-dataset stdout+stderr go to logs/.
# Usage:  bash run_all_glem.sh            # all feasible datasets
#         bash run_all_glem.sh cora pubmed  # only the named ones
set -u

PY=/home/roymahlab/miniconda3/envs/ct/bin/python
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG_DIR="$ROOT/configs/glem"
cd "$ROOT"
mkdir -p logs

# Feasible on 2x24GB GPUs (products is intentionally excluded — see bottom).
DEFAULT_DATASETS=(cornell texas washington wisconsin \
                  cora citeseer pubmed wikics bookchild bookhis sportsfit arxiv)

# Allow overriding the set on the command line.
if [ "$#" -gt 0 ]; then
  DATASETS=("$@")
else
  DATASETS=("${DEFAULT_DATASETS[@]}")
fi

for ds in "${DATASETS[@]}"; do
  cfg="$CFG_DIR/$ds.sh"
  if [ ! -f "$cfg" ]; then
    echo "!! no config for '$ds' ($cfg) — skipping"
    continue
  fi
  echo "==================== GLEM: $ds ===================="
  # shellcheck disable=SC1090
  source "$cfg"
  # shellcheck disable=SC2086  (GLEM_ARGS is an intentional word-split arg list)
  $PY src/models/GLEM/trainGLEM.py $GLEM_ARGS 2>&1 | tee "logs/glem_$ds.log"
  echo "==================== done: $ds ===================="
done

# ---------------------------------------------------------------------------
# ogbn-products is DEFERRED and NOT run by default.
#
# On 2x24GB GPUs, full-graph RevGAT over products' 2.4M nodes / 124M edges will
# OOM, and DeBERTa fine-tuning over 2.4M x 512 tokens would take days. A ready
# config exists at configs/glem/products.sh. To attempt it anyway (you will
# likely need a mini-batch GNN trainer and/or a larger-memory GPU), uncomment:
#
# source "$CFG_DIR/products.sh"
# $PY src/models/GLEM/trainGLEM.py $GLEM_ARGS 2>&1 | tee "logs/glem_products.log"
# ---------------------------------------------------------------------------
