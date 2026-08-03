# Does GLEM's uniform pseudo-labeling harm an identifiable node population?

**Status: PREREGISTRATION. Written and committed before any instrumentation code
exists and before any measurement has been taken.** Sections 1-12 are the
protocol. Section 13 is empty and will be filled in only after the runs
complete. Section 14 logs any deviation from this protocol, with its reason and
its date.

Preregistered: 2026-08-03.
Repository state at preregistration: branch `tag_datasets_testings`, HEAD `3ceaa2e`.

---

## 1. What this measures, and what it does not

This adds a **measurement instrument** to GLEM (Zhao et al., ICLR 2023). It is
not a change to the method. GLEM runs with its published hyperparameters; its
headline accuracy is reported as a fidelity check (§12). No hyperparameter is
tuned, and no part of GLEM is improved or weakened, in any arm.

The one exception is Arm 2, which sets the pseudo-label loss weight to zero. That
is a **control**, not a variant under evaluation, and it is interpreted only as a
subtrahend: any pattern it reproduces is not attributable to distillation.

## 2. Hypothesis

GLEM alternates an E-step (train the LM on the GNN's pseudo-labels over unlabeled
nodes) and an M-step (train the GNN on the LM's embeddings and pseudo-labels).
Each step weights its pseudo-label term with a single global scalar — α for the LM
step, β for the GNN step (`src/models/GLEM/GLEM_utils.py:131-144`). That encodes
the assumption that **the value of the teacher's signal does not depend on the
node.**

We predict that assumption is false in a specific, directional way. Each model has
an inductive bias, and outside it the model is not merely uncertain but
*confidently wrong*:

- The **GNN** fails on **low-local-homophily** nodes: it aggregates neighbours of
  other classes and emits a peaked, incorrect posterior.
- The **LM** fails on **semantically ambiguous** text: nodes whose neighbours in
  text-embedding space carry mixed labels.

**Prediction.** When a model teaches from inside a region where it is out of its
own inductive bias, its pseudo-labels will *corrupt* student nodes that were
previously correct, and the net effect in those bins will be negative.

## 3. Unit of measurement

The unit is one **distillation event**: a single E-step or a single M-step. For
each event, the student's per-node prediction is compared immediately before and
immediately after that step:

```
correction  : wrong   -> correct
corruption  : correct -> wrong
NCS         : (corrections - corruptions) / n_bin      # net correction score
```

`n_bin` is the number of analysed nodes in the bin (§6), and is reported alongside
every NCS value without exception.

With `inf_n_epochs=2` and `inf_tr_n_nodes=100000` (every config in
`configs/glem/`), `EmIterInfo` yields `total_iters = 2`
(`src/models/GLEM/GLEM_utils.py:26`). Each run therefore produces **4
distillation events**: 2 E-steps and 2 M-steps. Step order differs by config and
is recorded per row: `arxiv`/`cora`/`citeseer`/`pubmed` use `em_order=LM-first`,
the four WebKB configs use `GNN-first`.

## 4. Binning axis: teacher-side, not student-side

Nodes are binned by the axis on which the **TEACHER** is out of its inductive
bias. This is the single most important detail in the protocol.

| GLEM step | teacher | student | bin nodes by | out-of-bias direction |
|---|---|---|---|---|
| E-step (GNN pseudo-labels train the LM) | GNN | LM | **local homophily** | low |
| M-step (LM pseudo-labels train the GNN) | LM | GNN | **kNN semantic ambiguity** | high |

Binning the E-step by ambiguity instead would answer a different question — "does
transfer help where the *student* is weak?" — which is not this hypothesis. The
student-side axis is computed and stored, but only for the 2x2 table below.

**2x2 quadrant table.** Crossing (teacher out-of-bias?) x (student out-of-bias?),
median split on each signal within each dataset. The cell the hypothesis
specifically predicts is **teacher out-of-bias, student fine** — an unreliable
teacher overwriting a node the student already had right.

**Teacher accuracy per bin is a required column, not optional.** Without it,
"corruption is higher in low-homophily bins" has the trivial alternative
explanation that those nodes are hard for everyone. Teacher accuracy is measured
from the actual teacher logits snapshot consumed by that step, never from a proxy
or a later checkpoint.

## 5. Signals

Both signals are computable **without the model being evaluated**, so no
correlation reported here is circular. Both are computed once per dataset and
cached.

1. **Local homophily** (teacher-side axis for the E-step): per node, the fraction
   of its neighbours sharing its label. **NaN for isolated nodes, which are
   excluded from every bin, every quadrant and every count.**

2. **kNN semantic ambiguity** (teacher-side axis for the M-step): embed every
   node's raw text with SBERT `all-MiniLM-L6-v2`; L2-normalize. For each node take
   its **k=15** nearest neighbours by cosine similarity, **drawn only from the
   labelled training nodes**, excluding itself. Compute the entropy of those
   neighbours' true-label distribution, normalized by `log(num_classes)` so it
   lands in [0, 1].

   Under the few-shot regimes (§8) the labelled training set shrinks, so this
   signal is **recomputed per label regime** and cached per
   `(dataset, label_regime)`. It is not reused across regimes.

3. **GLANCE label-free homophily estimate** (secondary axis, reported alongside):
   `h_v = p_v · mean_{u∈N(v)} p_u` over predicted class distributions. Reported
   because true local homophily needs neighbour labels and so is unusable at
   inference time; the question is whether the label-free estimate reproduces the
   pattern the label-derived one shows. Isolated nodes: NaN, excluded.

Bin definitions, fixed now:
- **Primary:** median split within dataset (2 bins per axis) — chosen so that the
  WebKB datasets, whose analysable population is ~100-140 nodes (§7), retain
  usable per-bin counts.
- **Secondary:** 5 quantile bins within dataset, with duplicate quantile edges
  collapsed (local homophily is heavily tied on low-degree graphs, so the realised
  bin count may be < 5 and is reported).
- Any bin with **n < 30** is reported with an explicit flag and is excluded from
  the verdict in §11.

## 6. Metrics reported per bin and per quadrant cell

`n`, corrections, corruptions, NCS, corruption rate, student accuracy before,
student accuracy after, **teacher accuracy in that bin**, and a two-sided exact
McNemar test.

McNemar, stated exactly: discordant pairs are the `corrections + corruptions`
nodes whose correctness changed; the test is a two-sided exact binomial on that
count with p = 0.5. Concordant nodes are excluded, by construction. This
distinguishes real net harm from ordinary churn.

## 7. Population analysed

GLEM pseudo-labels the unlabeled set U. Analysis is restricted to nodes that

(a) **actually received a pseudo-label in that step**, and
(b) **have ground truth available** — transductively, val ∪ test.

Condition (a) is not the same set for the two steps and cannot be reconstructed
after the fact, so the node-id set is logged at the time the step runs:

- The **GNN** resamples its pseudo-label nodes *every epoch*
  (`get_sampled_aug_ids`, `src/utils/data/datasets.py:128-131`), so over hundreds
  of epochs nearly all of `pl_nodes` is touched.
- The **LM** takes a deterministic range slice per EM iteration
  (`get_inf_aug_train_ids`, `src/utils/data/datasets.py:118-126`, driven by
  `EmIterInfo.inf_node_ranges`), so each E-step sees only a window of `pl_nodes`.
- Where `pl_filter` is set (0.8 in the four WebKB configs; unset for arxiv/cora/
  citeseer/pubmed), only the top-80% most-confident pseudo-label nodes are
  eligible, and that set changes per iteration
  (`src/utils/data/datasets.py:89-105`).

Because WebKB test splits are tiny, **all ground-truth nodes (val ∪ test) are
analysed, not test alone.** Approximate analysable population, from the split
masks: cornell ~99, texas ~97, washington ~119, wisconsin ~138 of 191/187/229/265
nodes. Exact per-bin n is recorded in every output row.

## 8. Arms — all three are required

**Arm 1 — GLEM as published.** Unmodified, published hyperparameters, exactly the
`configs/glem/*.sh` recipes. arxiv: DeBERTa-base + RevGAT, LM-first, α = 0.8,
β = 0.05. WebKB: DeBERTa-base + GCN, GNN-first, α = 0.5, β = 0.7, `pl_filter=0.8`.

**Arm 2 — α = β = 0 control.** Same code, same number of epochs per step, same
warm start, same pseudo-label node sampling; pseudo-label weight set to zero so
each model's loss sees gold labels only. This separates "the pseudo-label term
corrupted these nodes" from "any retraining flips nodes." **Any NCS pattern that
appears identically in this arm is not attributable to distillation.**

Two documented caveats on what Arm 2 does and does not remove — both discovered
by reading the loss and feature paths, both recorded here rather than found later:

- **2a / 2b split for the GCN configs.** `gnn_label_input=T` (the four WebKB
  configs) makes `node_feature` concatenate the teacher's `y_hat` onto the GNN's
  **input features**, not just its loss (`src/utils/data/datasets.py:171-180`).
  Setting β = 0 there removes the pseudo-label CE term while the teacher's
  predictions still reach the GNN through the feature channel. So for those
  datasets both variants are run and reported:
  **2a** = β = 0 with `gnn_label_input=T` (isolates exactly the loss term, holds
  everything else at published values), and
  **2b** = β = 0 with `gnn_label_input=F` (removes the teacher from the GNN
  entirely, at the cost of a second deviation from the published recipe).
  The arxiv/cora/citeseer/pubmed configs already set `gnn_label_input=F`, so 2a
  and 2b coincide there and only one control is run.
- **No arm is a true "teacher-free" baseline.** The GNN's input features are
  *always* LM embeddings — that channel cannot be removed without destroying
  GLEM. Arm 2's scope is therefore precisely "**the pseudo-label CE term
  removed**," and it will be described that way in the results, not as "no
  distillation."
- **α = 0 has a known asymmetry in the LM step.** `compute_loss` takes `mle_loss`
  over the gold subset *of each batch*. At α = 0 an all-pseudo batch yields
  `mle_loss` over an empty set → NaN → `deal_nan` → zero loss, so the optimizer
  steps with zero gradient: parameters are untouched but the LR schedule advances
  identically to Arm 1, which is the behaviour we want for "same number of
  epochs." With `ce_reduction=mean`, mixed gold/pseudo batches do give gold nodes
  a slightly different effective weight than in Arm 1. This is a limitation of
  the control, stated in advance, not a result.

**Arm 3 — label regime sweep. WITHDRAWN 2026-08-03, before any measurement; see
amendment A5.** As originally preregistered: both arms at (a) the standard split
and (b) few-shot at 3, 5 and 10 labels per class, holding val/test fixed, on the
expectation that the effect would be **much stronger at, or present only at,
few-shot** — with a large labelled fraction the gold-label CE term anchors the
student and a bad teacher cannot drag it far.

The experiment now runs the **standard split only**. Every verdict in §11 is
therefore a statement about the full-label regime, and A5 records what that
forecloses.

## 9. Datasets

- **ogbn-arxiv** (`arxiv_TA`) — required. Per-node local homophily varies widely
  even though global homophily is moderate, so the teacher-side binning has range.
- **Cora, Citeseer, PubMed** (`cora_TAG`, `citeseer_TAG`, `pubmed_TAG`).
- **WebKB: cornell, texas, washington, wisconsin** — strongly heterophilous, and
  where the GNN-as-teacher failure should be most visible. Small (187-265 nodes);
  all ground-truth nodes analysed, n reported everywhere.

Global and per-node homophily for each dataset will be **computed and reported**
from the loaded graphs rather than quoted from the literature.

Cost note, recorded so the final scope is auditable: arxiv's E-step is a DeBERTa
fine-tune and dominates the compute. With Arm 3 withdrawn (A5) the scope is the
standard split on all eight datasets, at ≥3 seeds, for Arm 1 and Arm 2 (2a and 2b
on the four WebKB/GCN configs, where they differ; a single control elsewhere). Any
configuration listed here that is ultimately not run will be named explicitly in
§13 as not run — not silently omitted.

## 10. Seeds and stability

**≥ 3 seeds per configuration.** NCS is reported with across-seed variance.

**A bin whose NCS sign is not stable across all seeds is not a result** and will
be reported as unstable regardless of any single seed's p-value.

Primary significance test is **per-seed** McNemar. A pooled McNemar on summed
discordant counts across seeds is also reported, with the caveat — stated here in
advance — that pooling reuses the same nodes across seeds, so the pooled p-value
is anti-conservative and is treated as descriptive only.

## 11. Preregistered decision rule

Fixed before any number is looked at. Bins, thresholds, axes and arms will not be
adjusted to move the result.

- **Supported** — NCS is significantly negative (McNemar p < 0.05, sign stable
  across all seeds) in the teacher-out-of-bias bins while positive elsewhere,
  **AND** the pattern is absent in the α = β = 0 control.
- **Weakly supported** — NCS stays positive everywhere but is significantly lower
  in teacher-out-of-bias bins, with teacher accuracy degrading across the axis.
- **Not supported** — NCS is flat across the axis, **or** the same pattern appears
  in the α = β = 0 control.

Bins flagged n < 30 do not contribute to the verdict.

**The weak and null outcomes are useful results.** Whichever occurs is reported
plainly in the summary of §13. A null will not be buried, softened, or
recharacterised.

## 12. GLEM fidelity check

Reproduced headline accuracy is reported against the published figure for the
matching GNN backbone. The paper reports **76.97** test on ogbn-arxiv with RevGAT;
this repo's README reports GLEM+RevGAT `0.7697 ± 0.0019` test / `0.7749 ± 0.0017`
val over 10 runs, and GLEM+GCN `0.7593 ± 0.0019` test. `configs/glem/arxiv.sh`
uses RevGAT, so RevGAT is the fidelity target. Our figure comes from 3 seeds, not
10, and will be reported with that caveat and with its own spread.

If reproduced accuracy is materially below the published number, that is reported
in §13 **before** any harm analysis is interpreted, since a broken reproduction
would make the rest of the measurement uninterpretable.

## 13. Results

*Empty by design. Nothing has been run. To be filled in after the runs complete,
with the verdict from §11 stated plainly and first.*

## 14. Amendment log

Any deviation from §1-12 is recorded here with its date and reason, including
deviations forced by compute limits or by bugs found during instrumentation.
Entries are append-only.

**A1 (2026-08-03) — analysis population under few-shot is still val ∪ test.**
§7 defines the population as pseudo-labeled nodes with ground truth, "in the
transductive setting that is the val ∪ test nodes." Under the few-shot regimes the
nodes dropped from the train split also become unlabeled, so they receive
pseudo-labels *and* have ground truth, which would enlarge the population in
few-shot arms only. They are **excluded**: the population is held to val ∪ test in
every regime so regimes stay comparable. Cost: few-shot NCS is measured on the
same nodes as standard-split NCS, ignoring a population that GLEM does in fact
pseudo-label.

**A2 (2026-08-03) — `eval_steps` is clamped to the run's total optimizer steps.**
`lm_trainer` derives `eval_steps` from `eval_patience`, which is tuned for the
standard split. Under few-shot the train set is orders of magnitude smaller and the
derived value can exceed the run's total optimizer steps, yielding zero
evaluations, no `eval_loss`, and a `KeyError` from `load_best_model_at_end`. Now
clamped to at least one evaluation. This binds only where the run would otherwise
crash; on the standard split the original value is already inside the budget and is
untouched. Affects evaluation cadence and hence which checkpoint
`load_best_model_at_end` selects in few-shot arms — not the loss, not the method.

**A3 (2026-08-03) — SBERT embeddings computed locally rather than taken from the
shipped `sbert_x.pt`.** §5 names `all-MiniLM-L6-v2`. Some datasets ship a 384-dim
`sbert_x.pt`, but of unstated provenance, and the datasets needed here do not all
have one. Embeddings are therefore computed for every dataset the same way
(mean pooling over the attention mask, then L2 normalization, per this model's
`1_Pooling/config.json`). Cross-checked against the shipped cornell tensor:
**mean cosine 1.00000, min 1.00000**, so the two are the same model and pooling and
the choice changes no value — it only makes provenance uniform.

**A5 (2026-08-03) — Arm 3 (few-shot label-regime sweep) withdrawn; standard split
only.** Requested by the experiment owner, to concentrate effort on full training.
Recorded before any measurement was taken, so this is a scope decision and not a
response to results.

What it costs, stated plainly because it bears directly on how a null must be
read. §8 predicted the harm effect would be strongest at, or exclusive to, the
few-shot regimes, on the reasoning that ~54% of arxiv being labelled leaves the
gold-label CE term able to anchor the student against a bad teacher. Dropping the
sweep removes the arm that would have tested that reasoning. Consequently a **Not
supported** verdict (§11) at the standard split **cannot** distinguish between
"uniform pseudo-labeling does not harm this population" and "it does, but the
gold-label anchor at this labelled fraction masks it." Any null will be reported
with that limitation attached rather than as a general negative result about GLEM.

Amendments A1 and A2 are moot in consequence — A1 governed a population that only
differs under few-shot, and A2's clamp only binds on train sets far smaller than
the standard splits. Both are left in place: the code paths are inert on the
standard split, and this log is append-only. The regime suffix in `EmIterInfo`
likewise resolves to the empty string throughout, so all runs reuse GLEM's cached
pretrain checkpoints exactly as an unprobed run would.

**A4 (2026-08-03) — measurement power of the E-step direction is severely limited
on the WebKB datasets.** Found by running the instrument, not by reading the code.
The published GCN recipe sets `lm_pl_ratio=0.1`, and `EmIterInfo.inf_node_ranges`
turns that into `ceil(n_train * 0.1)` pseudo-label nodes per E-step — on cornell,
**10 nodes**, of which ~9 fall in val ∪ test, giving **n ≈ 3-7 per median bin** for
the `gnn->lm` direction. That is the direction where the heterophilous WebKB
datasets were expected to show the GNN-as-teacher failure most clearly. Raising
`lm_pl_ratio` would deviate from the published hyperparameters and is therefore
**not** being done (§1). Consequences, fixed now rather than after seeing results:
the arxiv/cora/citeseer/pubmed configs use `lm_pl_ratio=1` and so cover the entire
unlabeled set per E-step — those datasets carry the `gnn->lm` direction at full
power; the four WebKB datasets' `gnn->lm` rows will be **flagged n < 30 and
excluded from the verdict** per §5, and enter only the pooled across-dataset figure
(deliverable 5). The WebKB `lm->gnn` direction is unaffected (n ≈ 30-34 per bin,
`gnn_pl_ratio=0.2` resampled every epoch).

---

## Appendix A — instrumentation contract

Recorded now so the measurement cannot be quietly redefined later.

**Snapshots.** The student's full logits `[N, C]`, aligned to node ids, are saved
immediately **before** and immediately **after** every E-step and M-step — not
only at the end of training — to a path keyed by
`(arm, seed, dataset, label_regime, iteration, step)`. The teacher's logits *as
consumed by that step* are saved with the same key.

Two facts about the codebase make this cheap and make it faithful:

1. GLEM's EM loop is a **subprocess orchestrator**
   (`src/models/GLEM/GLEM_trainer.py:137-148`): each step is a separate process
   launched via `uf.run_command`, with state passed through a pickled `em_info`
   file. Teacher and student communicate **only** through fp16 memmap files on
   disk — GNN pred written at `src/models/GNNs/gnn_utils.py:138`, LM pred written
   in `src/models/LMs/infLM.py`, and read back as `pseudo_label_file` at
   `src/models/GNNs/gnn_utils.py:48` and `src/models/LMs/lm_utils.py:78`.
   Therefore the student's pre-step logits *are* the file its own side wrote at
   the previous step, and the teacher's logits *are* the file the step reads. No
   new forward passes are needed, and teacher accuracy is measured from the real
   teacher snapshot.
2. **Those files are currently overwritten in place.** For `em_iter >= 0` the pred
   paths carry no iteration index (`src/models/GLEM/GLEM_utils.py:43` and `:53`),
   so each M-step clobbers the previous GNN pred and each E-step clobbers the
   previous LM pred. Before/after pairs are destroyed as a run proceeds. This is
   the one thing that must be fixed for the experiment to be possible at all, and
   it is fixed by copying to an iteration-keyed path at write time — not by
   restructuring `EmIterInfo`.

**Also logged per step:** the node-id set that actually received a pseudo-label
(§7), the resolved α/β actually in force, and the arm identifier — so no row's
provenance depends on inferring it from a directory name.

**Signals** are computed once per `(dataset, label_regime)` and cached to disk.

## Appendix B — deliverables

1. `EXPERIMENT.md` — this preregistration, then §13 results and which of the three
   §11 verdicts holds.
2. Long-form CSV: one row per
   `(arm, seed, dataset, label_regime, iteration, step, direction, axis, bin)`
   with every column in §6.
3. The 2x2 quadrant CSV.
4. Per-dataset figure: correction/corruption bars plus NCS line against the
   teacher-side axis, teacher accuracy on a secondary y-axis, one panel per
   direction.
5. Pooled figure: NCS against within-dataset quantile of the signal, so datasets
   with different signal distributions are comparable.
6. NCS per EM iteration for the worst bin — does corruption accumulate across EM
   iterations, or get repaired later?
7. The §12 fidelity note.

## Appendix C — prior code

`analysis/` (untracked at preregistration) is prior work from a **different**
codebase: it imports `data.data.TAGDataset`, hydra/omegaconf configs,
`cfg.project_root` and an `analysis_output/{gnn,llm}_logits.pt` layout, none of
which exist in this repository. Its before/after model is also solo-vs-final-EM —
a single transfer event, not per-step.

Its numerics match this protocol and will be ported onto this repo's `SeqGraph` /
`graph.info` loader rather than rewritten: `semantic_ambiguity_per_node` (k=15,
train-only neighbours, entropy / log C), `soft_local_homophily` (the GLANCE `h_v`
estimate, NaN on isolated nodes), `node_homophily`, `fast_auroc`, and
`quantile_bins` (which handles the heavy ties in local homophily on low-degree
graphs — relevant for WebKB). The driver is rewritten around per-step snapshots.
