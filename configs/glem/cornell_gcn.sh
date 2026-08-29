#!/usr/bin/env bash
# GLEM configuration for 'cornell' — LM=DeBERTa-base, GNN=GCN, logging to wandb.
# Hyperparameters follow the paper's ogbn-arxiv GCN+GLEM recipe
# (OGB/ogbn-arxiv/README.md GLEM+GCN). Sourced by run_all_glem.sh ($GLEM_ARGS).
DATASET_STR="cornell_TAG"
WANDB_NAME="cornell"
GLEM_ARGS="--dataset=${DATASET_STR} \
  --lm_model=Deberta --gnn_model=GCN --em_order=GNN-first \
  --gnn_early_stop=300 --gnn_epochs=500 --gnn_input_norm=F --gnn_label_input=T \
  --gnn_pl_ratio=0.2 --gnn_pl_weight=0.7 \
  --inf_n_epochs=2 --inf_tr_n_nodes=100000 \
  --lm_ce_reduction=mean --lm_cla_dropout=0.4 --lm_epochs=1 --lm_eq_batch_size=30 \
  --lm_eval_patience=90 --lm_init_ckpt=PrevEM --lm_label_smoothing_factor=0 \
  --lm_load_best_model_at_end=T --lm_lr=3e-05 --lm_pl_ratio=0.1 --lm_pl_weight=0.5 \
  --pl_filter=0.8 --pseudo_temp=0.2 --seed=0 --gpus=0 \
  --wandb_name=${WANDB_NAME}"
