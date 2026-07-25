#!/usr/bin/env bash
# GLEM configuration for 'cora' — LM=DeBERTa-base, GNN=RevGAT, logging to wandb.
# Hyperparameters follow the paper's ogbn-arxiv RevGAT+GLEM recipe
# (OGB/ogbn-arxiv/README.md); only --dataset / --wandb_name / --lm_eval_patience
# are dataset-specific. Sourced by run_all_glem.sh (defines $GLEM_ARGS).
DATASET_STR="cora_TAG"
WANDB_NAME="cora"
GLEM_ARGS="--dataset=${DATASET_STR} \
  --lm_model=Deberta --gnn_model=RevGAT --gnn_ckpt=RevGAT --em_order=LM-first \
  --gnn_early_stop=300 --gnn_epochs=2000 --gnn_input_norm=T --gnn_label_input=F \
  --gnn_pl_ratio=1 --gnn_pl_weight=0.05 \
  --inf_n_epochs=2 --inf_tr_n_nodes=100000 \
  --lm_ce_reduction=mean --lm_cla_dropout=0.4 --lm_epochs=3 --lm_eq_batch_size=30 \
  --lm_eval_patience=1600 --lm_init_ckpt=None --lm_label_smoothing_factor=0 \
  --lm_load_best_model_at_end=T --lm_lr=2e-05 --lm_pl_ratio=1 --lm_pl_weight=0.8 \
  --pseudo_temp=0.2 --seed=0 --gpus=0 \
  --wandb_name=${WANDB_NAME}"
