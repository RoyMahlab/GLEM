#!/usr/bin/env bash
# The run matrix: which arms run on which configs, what each cell costs, and what
# each arm needs before it can run. Sourced by probe_sweep.sh and run_all.sh so the
# two cannot drift -- a sweep that disagreed with the driver about which cells exist
# would silently under- or over-run the grid.
#
# Nothing here executes. Source it, do not run it. The caller must set MATRIX_ROOT
# to the repository root before sourcing.

# ─────────────────────────────────────────────────────── config -> arms
#
# The two control arms differ only when gnn_label_input=T routes the teacher's
# y_hat into GNN input features; the RevGAT configs already set F, so 2a and 2b
# coincide there and only one control is run (EXPERIMENT.md section 8).
#
# A28's beta x selection grid is registered for the beta=0.05 configs ONLY. Its
# b=0.05 row IS `published`, which is only true where the config ships 0.05 --
# the four *_gcn configs ship 0.7, so the grid is not defined there and is not
# listed. assert_grid_beta() below enforces that rather than trusting this comment.
declare -A ARMS=(
  [cornell_gcn]="published alpha0_li_T alpha0_li_F published_li_F oracle oracle_random"
  [texas_gcn]="published alpha0_li_T alpha0_li_F published_li_F oracle oracle_random"
  [washington_gcn]="published alpha0_li_T alpha0_li_F published_li_F oracle oracle_random"
  [wisconsin_gcn]="published alpha0_li_T alpha0_li_F published_li_F oracle oracle_random"
  [cora]="published conf_gate80 sig_gate80_gnn sig_gate80_lm b05_rand80 b30 b30_conf80 b30_sig80gnn b30_sig80lm b30_rand80 beta_high b80_conf80 b80_sig80gnn beta_high_sig80 beta_high_rand80 alpha0_li_T oracle oracle_random"
  # citeseer at its shipped 20-labels-per-class split leaves DeBERTa at 0.208
  # against a 0.167 chance baseline -- not a teacher. citeseer60 is the same graph
  # re-split to 60% train (318/class); see settings.py.
  [citeseer60]="published conf_gate80 sig_gate80_gnn sig_gate80_lm b05_rand80 b30 b30_conf80 b30_sig80gnn b30_sig80lm b30_rand80 beta_high b80_conf80 b80_sig80gnn beta_high_sig80 beta_high_rand80 alpha0_li_T oracle oracle_random"
  [citeseer]="published conf_gate80 sig_gate80_gnn sig_gate80_lm b05_rand80 b30 b30_conf80 b30_sig80gnn b30_sig80lm b30_rand80 beta_high b80_conf80 b80_sig80gnn beta_high_sig80 beta_high_rand80 alpha0_li_T oracle oracle_random"
  [pubmed]="published conf_gate80 sig_gate80_gnn sig_gate80_lm b05_rand80 b30 b30_conf80 b30_sig80gnn b30_sig80lm b30_rand80 beta_high b80_conf80 b80_sig80gnn beta_high_sig80 beta_high_rand80 alpha0_li_T oracle oracle_random"
  # published_li_T / alpha0_li_T_only / the A21 label-feature arms are A19/A21 and
  # are registered for arxiv ONLY.
  [arxiv]="published alpha0_li_T oracle oracle_random conf_gate60 conf_gate80 conf_gate90 sig_gate80 sig_gate80_gnn sig_gate80_lm sig_gate90 published_li_T alpha0_li_T_only teacher_consistent mask_pseudo mask_train unimp_mask beta_high beta_high_sig80 beta_high_rand80 b05_rand80 b30 b30_conf80 b30_sig80gnn b30_sig80lm b30_rand80 b80_conf80 b80_sig80gnn"
  # wikics: the A23 feature-channel replication on a second dataset. RevGAT, li=F.
  [wikics]="published conf_gate80 sig_gate80_gnn sig_gate80_lm b05_rand80 b30 b30_conf80 b30_sig80gnn b30_sig80lm b30_rand80 beta_high b80_conf80 b80_sig80gnn beta_high_sig80 beta_high_rand80 published_li_T teacher_consistent alpha0_li_T"
  [cornell]="published alpha0_li_T oracle oracle_random"
  [texas]="published alpha0_li_T oracle oracle_random"
  [washington]="published alpha0_li_T oracle oracle_random"
  [wisconsin]="published alpha0_li_T oracle oracle_random"
)

# Configs whose DATASET_STR collides with another config's. The tag is appended to
# the archive's dataset directory (cornell_TAG+revgat) so the two backbones do not
# overwrite each other.
declare -A VARIANT=(
  [cornell]=revgat
  [texas]=revgat
  [washington]=revgat
  [wisconsin]=revgat
)

# ─────────────────────────────────────────── arm -> absolute beta, or unset
#
# An arm appears here when its IDENTITY is an absolute --gnn_pl_weight, so the value
# must be pinned rather than inherited from the config. Arms absent from this map are
# defined RELATIVE to the published recipe ("published, plus a gate") and must keep
# inheriting: pinning `conf_gate80` to 0.05 would silently redefine it on the *_gcn
# configs, which ship 0.7.
#
# probe_run.sh reads this map -- it is the single place any beta is stated.
declare -A ARM_BETA=(
  [alpha0_li_T]=0        [alpha0_li_F]=0       [alpha0_li_T_only]=0
  [b05_rand80]=0.05
  [b30]=0.3              [b30_conf80]=0.3      [b30_sig80gnn]=0.3
  [b30_sig80lm]=0.3      [b30_rand80]=0.3
  [beta_high]=0.8        [beta_high_sig80]=0.8 [beta_high_rand80]=0.8
  [b80_conf80]=0.8       [b80_sig80gnn]=0.8
)

# Arms that also pin alpha (--lm_pl_weight). Only the alpha0 controls do; every
# other arm leaves the LM's pseudo-label weight at the published value.
declare -A ARM_ALPHA=(
  [alpha0_li_T]=0 [alpha0_li_F]=0 [alpha0_li_T_only]=0
)

# ──────────────────────────────────── arm -> cached signals it needs before running
#
# `hom` = <key>_edge_index.npy   (GLANCE soft homophily, gates the E-step)
# `amb` = <key>_standard_s0_ambiguity.npy (kNN ambiguity, gates the M-step)
#
# src/probe/gating.py raises FileNotFoundError *inside training* when one is absent,
# so these are checked up front instead: a 13h arxiv cell should not die at its first
# distillation event over a missing 1MB array.
declare -A ARM_NEEDS=(
  [sig_gate80]="hom amb"     [sig_gate90]="hom amb"
  [sig_gate80_gnn]="hom"     [sig_gate90_gnn]="hom"
  [sig_gate80_lm]="amb"      [sig_gate90_lm]="amb"
  [b30_sig80gnn]="hom"       [b80_sig80gnn]="hom"
  [b30_sig80lm]="amb"        [beta_high_sig80]="amb"
)

# ───────────────────────────────────────────────── config -> seconds per run
#
# Measured means over successful runs in logs/probe/manifest.tsv as of 2026-08-29.
# Used only to balance shards; a wrong value costs balance, never correctness.
# citeseer60 has no history yet -- assumed 4x citeseer, it trains on 2.7x the nodes.
declare -A COST=(
  [arxiv]=48369 [pubmed]=4630 [cora]=698 [wikics]=697 [citeseer]=259
  [citeseer60]=1040
  [wisconsin]=138 [washington]=131 [cornell]=117 [texas]=116
  [washington_gcn]=104 [cornell_gcn]=99 [wisconsin_gcn]=97 [texas_gcn]=90
)

# Every config the matrix covers, cheapest first so a dry run reads in the order a
# human would want to start them.
ALL_CONFIGS=(texas_gcn wisconsin_gcn cornell_gcn washington_gcn
             texas cornell washington wisconsin
             citeseer cora wikics citeseer60 pubmed arxiv)

# ───────────────────────────────────────────────────────────── helpers

# The signal-cache key for a config: the leading token of its DATASET_STR, which is
# what src/probe/gating.py derives (`str(cf.dataset).split('_')[0]`). citeseer60
# resolves to `citeseer60`, NOT `citeseer` -- it shares the graph but not the train
# split, so its ambiguity array is genuinely its own.
sig_key_for() {
  local ds
  # shellcheck disable=SC1090
  ds=$(. "$MATRIX_ROOT/configs/glem/$1.sh" >/dev/null 2>&1; echo "$DATASET_STR")
  echo "${ds%%_*}"
}

# The dataset directory used in the probe archive, variant suffix included.
archive_ds_for() {
  local ds tag
  # shellcheck disable=SC1090
  ds=$(. "$MATRIX_ROOT/configs/glem/$1.sh" >/dev/null 2>&1; echo "$DATASET_STR")
  tag="${VARIANT[$1]:-}"
  [ -n "$tag" ] && ds="$ds+$tag"
  echo "$ds"
}

# The --gnn_pl_weight a config ships, read from the config text.
published_beta_for() {
  local v
  v=$(grep -oE -- '--gnn_pl_weight=[0-9.]+' "$MATRIX_ROOT/configs/glem/$1.sh" | tail -1)
  echo "${v#--gnn_pl_weight=}"
}

# A28's grid is only meaningful where its b=0.05 row coincides with `published`.
# Called by run_all.sh's preflight for every config carrying a grid arm.
assert_grid_beta() {
  local cfg=$1 beta
  beta=$(published_beta_for "$cfg")
  if [ "$beta" != "0.05" ]; then
    echo "!! $cfg ships --gnn_pl_weight=$beta, but the A28 grid assumes the b=0.05" >&2
    echo "   row is the published recipe. Either drop the grid arms from this config" >&2
    echo "   in scripts/matrix.sh, or register a separate row for beta=$beta." >&2
    return 1
  fi
  return 0
}
