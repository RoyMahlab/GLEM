# Running the full matrix, on one machine or several

Everything below assumes the repo root as the working directory.

## What "all arms on all datasets" means

`scripts/matrix.sh` is the single source of truth: which arms are registered on
which configs, what each cell costs, and what each arm needs before it can run.
`scripts/probe_sweep.sh` and `scripts/run_all.sh` both read it, so they cannot
disagree about which cells exist.

It is deliberately **not** the cross product of arms and datasets. A19/A21's
label-feature arms are registered for arxiv only; A28's beta grid only for configs
that ship `beta=0.05`; the WebKB GCN configs carry the 2a/2b control split that the
RevGAT configs do not. EXPERIMENT.md §13.7 and A22 record why running an arm outside
its registered set produces a number whose sign is not interpretable — the homophily
gate inverts on heterophilous graphs, for one.

As of 2026-08-29 the matrix is **474 cells** (14 configs × their arms × 3 seeds),
of which 194 are complete and **280 outstanding, about 462 GPU-hours**.

Cost is extremely lopsided. arxiv is ~13.4h per run and 29 arms, so it is about
**92% of the total**. Any plan that does not split arxiv itself will not balance.

## One machine

```bash
DRY_RUN=1 scripts/run_all.sh          # see the plan and the cost first
scripts/run_all.sh                    # run everything outstanding
GPU_LIST="0 1" scripts/run_all.sh     # one seed per GPU, arms sequential per lane
```

It is resumable and interrupt-safe: `SKIP_DONE=1` is the default, and a cell counts
as done only when its archive holds all four distillation events. Stop it whenever;
re-running picks up where it left off. That is also how you retry a failure — fix
the cause and re-run, and only the incomplete cells go again.

## Several machines

Give each machine a different `k` in `SHARD=k/N`. The assignment is a cost-balanced
longest-processing-time packing computed from the matrix, so it is deterministic:
every machine derives the identical split and simply keeps its own share. No
coordination, no central queue.

**Machines may start at different times.** The packing runs over the *whole* matrix,
not over what is still outstanding, so the assignment is a pure function of
`matrix.sh`. A machine joining a day late computes the same split as one that started
first, then skips whatever is already complete. Were the packing computed over only
the outstanding cells, a late joiner would pack a smaller set, land on a different
assignment, and the shards would stop lining up — some cells claimed twice, others by
nobody. Verified at N=4: the four shards cover all 474 cells with no overlap, and each
machine's remaining work is always a strict subset of what it owns.

```bash
# machine 1                # machine 2                # machine 3
SHARD=1/3 scripts/run_all.sh   SHARD=2/3 scripts/run_all.sh   SHARD=3/3 scripts/run_all.sh
```

For the current backlog that is 154h02m / 154h02m / 154h03m — balanced to within a
minute across 462 hours. Each machine prints the whole balance table, so you can see
what the others were given.

### Which shard unit

- `SHARD_UNIT=cell` (default) balances individual `(config, arm, seed)` cells. Best
  balance. **Requires each machine to have its own checkout.**
- `SHARD_UNIT=seed` keeps every `(config, seed)` whole on one machine. Coarser and
  less balanced, but safe even if the machines share a filesystem.

The distinction is not cosmetic. Arms that differ only by an environment variable
(`oracle` vs `oracle_random`, every `sig_*` pair) produce an identical
`glem_cfg_str`, so two of them running at once against the same filesystem overwrite
each other's `temp/glem_{lm,gnn}` working directories and `em_info` pickle. Seeds
never collide — every EM path is keyed by seed — which is why lanes within a machine
are per-seed and why `SHARD_UNIT=seed` is the shared-filesystem-safe option.

### Setting up a second machine

```bash
git clone <repo> && cd GLEM && git checkout <this branch>
# python env: environment.yml, or point PY at an existing interpreter
export PY=/path/to/python

scripts/prepare_signals.py --check --all   # what is missing
scripts/prepare_signals.py --all           # build it (SBERT; arxiv is the slow one)

DRY_RUN=1 SHARD=2/3 scripts/run_all.sh     # confirm the plan
SHARD=2/3 scripts/run_all.sh
```

Datasets and pretrain checkpoints are per-machine and are created on demand.
`WARMUP=auto` (the default) serialises the first cell of any config whose
`temp/prt_{lm,gnn}/<dataset>` is absent, because concurrent lanes would otherwise
race to write a pretrain path that carries no seed. Once the pretrain exists, lanes
fan out normally.

### Collecting the results

Each machine writes its own `temp/probe_output/<dataset>/standard/<arm>/seed<n>/`.
Cells are disjoint by construction, so merging is a union — no conflicts:

```bash
# on each worker, back to the machine holding the analysis
rsync -av temp/probe_output/ <analysis-host>:<repo>/temp/probe_output/
```

Then rebuild the tables:

```bash
python src/probe/analyze.py --probe-dir temp/probe_output --out temp/probe_analysis
```

`_signals/` is a derived cache and is identical everywhere it is built from the same
dataset, so it does not matter which copy wins.

## Cached signals

The gated arms read two arrays per dataset, and `src/probe/gating.py` raises
`FileNotFoundError` **inside training** when either is absent — on arxiv that lands
roughly three hours into a thirteen-hour run:

| file | axis | gates |
|---|---|---|
| `<key>_edge_index.npy` | GLANCE soft homophily | the E-step, where the GNN teaches |
| `<key>_standard_s0_ambiguity.npy` | kNN ambiguity | the M-step, where the LM teaches |

`scripts/prepare_signals.py` builds both from the dataset alone — no completed run
is needed, because GLEM's train split is exactly TAGDataset's `train_mask`
(`utils/data/preprocess_tag.load_tag_dgl` derives its `split_idx` from those masks).
Where an archived run does exist, its `splits.npz` is cross-checked against the mask
and a mismatch is fatal.

Both `run_all.sh`'s preflight and `probe_run.sh` itself check for these before
anything trains, so a missing 1MB array fails in the first second instead.

## Where the pseudo-label weights come from

alpha (`--lm_pl_weight`) and beta (`--gnn_pl_weight`) are stated in exactly one
place: `ARM_ALPHA` / `ARM_BETA` in `scripts/matrix.sh`.

- An arm **listed** there pins an absolute weight. `b30` is beta=0.3 on every config.
- An arm **absent** from it is defined relative to the published recipe and inherits
  the config's value. `conf_gate80` means "published, plus a confidence gate", which
  is beta=0.05 on arxiv and beta=0.7 on the four `*_gcn` configs.

Getting this backwards silently redefines an arm on some datasets, which is how
`b05_rand80` shipped inheriting beta — it would have been a beta=0.7 arm on the
`*_gcn` configs while being named for 0.05. Every run now prints the effective
values it actually used:

```
=== effective: alpha=0.8 beta=0.05 (published) pl_filter=none gnn_label_input=F gate=none labelfeat=none
```

`run_all.sh`'s preflight additionally refuses to run any A28 grid arm on a config
whose published beta is not 0.05, since the grid's b=0.05 row *is* `published`.

## Environment reference

| variable | default | meaning |
|---|---|---|
| `SEEDS` | `0 1 2` | seeds to run |
| `GPU_LIST` | `0` | one lane per GPU; seeds are the lanes |
| `SHARD` | unset | `k/N` — this machine's share |
| `SHARD_UNIT` | `cell` | `cell` (own checkout) or `seed` (shared filesystem) |
| `ONLY_ARMS` | unset | restrict to named arms; cannot add arms the matrix omits |
| `SKIP_DONE` | `1` | skip cells whose archive holds all 4 events |
| `DRY_RUN` | `0` | print the plan, run nothing |
| `WARMUP` | `auto` | serialise the first cell when a pretrain is missing |
| `PY` | `.venv/bin/python` | interpreter |
| `GLEM_PROBE_DIR` | `temp/probe_output` | archive location |
