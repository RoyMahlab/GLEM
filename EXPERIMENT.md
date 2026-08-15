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

**A18 (2026-08-13) — new exploratory arms `sig_gate80/90`: an exogenous-signal
gate. Excluded from the §11 verdict.**

A16's confidence gate failed: on arxiv it recovered ~0 of the oracle's +2.82pp (GNN)
and +3.62pp (LM), against the +0.8pp predicted from selection precision. **That
prediction was wrong and the reason is diagnostic.** At the oracle keep-rate a
confidence gate excludes 51% of the teacher's errors but **0.0%** of the
*confidently wrong* ones — it is structurally blind to them, which is the same
blindness §7 measured as AUROC 0.925 → 0.628 across the homophily axis. It strips
the harmless errors and leaves the damaging ones.

These arms test whether an **exogenous** signal reaches the population confidence
cannot. Measured on arxiv at the oracle keep-rate:

| gate | all errors excluded | **confidently-wrong excluded** |
|---|---|---|
| confidence, gnn→lm | 0.516 | **0.000** |
| homophily, gnn→lm | 0.485 | **0.116** |
| confidence, lm→gnn | 0.507 | **0.000** |
| ambiguity, lm→gnn | 0.423 | **0.177** |

Design: keep the top k% by GLANCE soft homophily when the GNN teaches, by inverted
kNN ambiguity when the LM teaches. Direction is read from `cf.em_phase`, never
re-derived. Keep-rates **match `conf_gate80/90` exactly**, so the difference between
the two families isolates the *signal* with shrinkage held constant.

**Decomposed by teacher into three arms**, so the combined effect can be attributed:

| arm | gates the E-step (GNN teaches) | gates the M-step (LM teaches) |
|---|---|---|
| `sig_gate80_gnn` | yes | no — runs exactly as `published` |
| `sig_gate80_lm` | no | yes |
| `sig_gate80` | yes | yes |

An ungated step returns early without touching `emi.n_pl_nodes`; shrinking it for a
step that was not gated would desynchronise the LM's per-iteration window from the
full pseudo-label set, which is the failure mode already recorded for the oracle arm.
`sig_gate80` is the sum of the two single-teacher arms only if the effects are
additive, which is itself worth testing — the EM loop couples the two steps, so a
teacher cleaned at one step changes what the other teacher sees at the next.

Preflight on arxiv, before any training — the gate does raise kept-set teacher
accuracy: GNN-teaching 0.767 → 0.833 (+6.6pp) at 80% keep, +3.3pp at 90%;
LM-teaching 0.755 → 0.805 (+5.0pp) at 80%, +2.6pp at 90%.

**Prediction fixed in advance: +0.3 to +0.5pp over `published`**, from scaling the
oracle's +2.82pp by the 12–18% of the confidently-wrong population these signals
reach. At arxiv's seed sd of 0.0018 that is t ≈ 2–3 at three seeds — detectable but
not comfortable. This is deliberately a more modest claim than A16's, and it is
grounded in the mechanism that explains A16's failure rather than in selection
precision, which is now known not to transfer.

**Scope limit found by the smoke test, and it is not incidental.** On cornell
(RevGAT, global homophily 0.219) the GLANCE gate *lowers* kept-set teacher accuracy
(−0.050, −0.045) while the ambiguity gate raises it (+0.027, +0.037). GLANCE's sign
inverts on heterophilous graphs — the same failure that made `nbr_pl_agree` score
0.35–0.42 AUROC on WebKB. The arms are therefore registered for **arxiv only**.
Applying them to a heterophilous dataset would gate in exactly the wrong direction,
and nothing in the current implementation detects that: the sign would have to be
chosen per (graph, direction), which global homophily alone cannot do.

Not §11 controls — they carry published α/β, and `report.py` matches `published`
exactly while treating only the `alpha0_*` arms as controls, so they are
structurally invisible to the verdict.

**A17 (2026-08-13) — §11's control disqualifier: an implementation gap fixed, and
an ambiguity in the preregistered text resolved on the record.**

Two distinct problems, found while arxiv's E-step became scoreable at three seeds.

*The implementation gap.* §11 states the disqualifier generally — "Not supported —
NCS is flat across the axis, **or** the same pattern appears in the α=β=0 control."
`report.py` consulted the control only inside the **Supported** branch; the
**Weakly supported** branch never checked it. That is a straightforward mismatch
against the preregistered text and is fixed: the weak clause now applies the same
disqualifier.

*The ambiguity.* §11 never defined what "the same pattern appears in the control"
means quantitatively, and on arxiv the readings diverge:

| arm | per-seed gap | mean | sd |
|---|---|---|---|
| published | −0.0089 / −0.0121 / −0.0114 | **−0.0108** | **0.0017** |
| α=β=0 control | +0.0001 / +0.0103 / −0.0169 | −0.0022 | **0.0137** |

Matching only the **sign of the mean** disqualifies the result. Requiring the
control to reproduce the pattern **stably** does not, because the control's gap
flips sign twice.

Resolved in favour of the stability reading, with the reasoning stated because the
choice was made after seeing the data. A disqualifier that fires on an unstable,
sign-flipping control would reject almost any true effect — the control's mean is
negative only because three values scattered about zero happen to average that way —
and §10 already establishes across-seed sign stability as this study's standard for
whether an effect is real. Applying a weaker standard to the control than to the
published arm would be incoherent.

The strict sign-only reading is disclosed rather than buried: under it, arxiv
`gnn->lm` would be **not supported** instead of **weakly supported**, and the study
would contain no positive verdict. Readers preferring that convention should read it
that way.

*A third defect fixed in passing.* The existing `{arm}_sign_stable` column measures
stability of NCS **within the out-of-bias bin**, not of the **gap**. On arxiv the
control's bin NCS is stably negative (−0.0229 / −0.0163 / −0.0387) while its gap
flips — so the disqualifier had no correct quantity available to test. Per-seed gap
stability is now computed as `{arm}_gap_sign_stable`, with `{arm}_gap_sd` alongside.

*Effect on the verdicts:* none besides making the rule match its text. Counts remain
13 not supported, 10 no test, **1 weakly supported** (arxiv `gnn->lm`).

**A16 (2026-08-09) — new exploratory arms `conf_gate60/80/90`: a *deployable*
confidence gate, swept over keep-rate. Excluded from the §11 verdict.**

A14's oracle established headroom on arxiv — +3.24pp (GNN) and +4.35pp (LM) over the
size-matched control, at seed sd 0.0018 and 0.0006. These arms ask how much of it a
gate that cannot see labels recovers.

No new gating code: they use GLEM's **own** `pl_filter`, which is already a per-node
confidence gate (`softmax(...).max(1).topk(k)` in `utils/data/datasets.py`). The four
RevGAT configs and arxiv leave it **unset**, so arxiv has never been run with any
filter; `0.8` is the value GLEM ships for its GCN recipe.

Justification for confidence rather than a learned mix: measured on arxiv, a
val-fitted logistic combination of seven deployable features scores AUROC **0.784**
against confidence alone at **0.784** for the E-step, and 0.796 vs 0.771 for the
M-step. Confidence is therefore at or near the achievable ceiling on this dataset, so
the extra machinery would buy ~0.00–0.03 AUROC and add a second thing that could be
wrong. Confidence also lifts selection precision 0.766 → 0.851, closing ~36% of the
random→oracle gap.

**Swept, not fixed**, because one operating point cannot distinguish "confidence
gating does not help" from "this keep-rate is wrong". The three arms bracket the
oracle's own keep-rate of 0.767 and the val-estimated teacher accuracy (0.768 E-step,
0.755 M-step).

Interpretation fixed in advance. The existing arms bracket the answer: dropping 23% of
pseudo-labels *at random* costs **−0.42pp** (0.7699 → 0.7657), and dropping exactly
the wrong 23% gains **+2.82pp**. A confidence gate must clear the shrinkage cost
before showing any gain, so ~36% of the precision gap predicts roughly **+0.8pp over
published** — t ≈ 6 at two seeds. A result near zero at every keep-rate means
confidence gating cannot exploit the headroom the oracle proves exists, which would
redirect the work toward signals confidence cannot capture.

Two limits: `pl_filter` applies **one** scalar to both directions, so it cannot take
the 0.768 / 0.755 split the two teachers want; and these arms carry published α/β, so
they are not §11 controls — `report.py` matches `published` exactly and treats only
the `alpha0_*` arms as controls, leaving them structurally invisible to the verdict.

**A15 (2026-08-09) — CORRECTED. An earlier version of this entry claimed the
oracle arm was invalidated by "transductive label leakage". That claim was wrong
and is withdrawn; what follows replaces it.**

*Why the original claim was wrong.* GLEM is transductive: the unlabeled set it
pseudo-labels **is** val ∪ test, and training a student on pseudo-labels for
evaluation nodes is the method, not contamination. The `published` arm does exactly
this, and so would any deployable gate. All three arms are trained and evaluated the
same way, so oracle-vs-`oracle_random` is a fair comparison and its gain is a real
answer to "what would perfect gating buy in this setting?". Calling the mechanism
that makes gating work an artifact was a category error on my part.

*What is actually true, and is the ordinary oracle caveat.* The oracle **selects**
using gold labels — selection quality AUROC 1.0 — while a deployable gate must
*predict* teacher correctness. Measured selection precision, i.e. the fraction of
kept nodes the teacher is right on, at the same keep-count the oracle uses:

| dataset | direction | random (base) | homophily | confidence | both | oracle |
|---|---|---|---|---|---|---|
| cora | gnn→lm | 0.896 | 0.942 | 0.938 | 0.949 | 1.000 |
| pubmed | gnn→lm | 0.947 | 0.956 | 0.968 | 0.969 | 1.000 |
| citeseer | gnn→lm | 0.575 | 0.696 | 0.725 | 0.739 | 1.000 |
| cornell | lm→gnn | 0.680 | 0.824 | 0.897 | 0.926 | 1.000 |
| wisconsin | lm→gnn | 0.717 | 0.818 | 0.859 | 0.899 | 1.000 |

A realisable gate closes roughly **40–75% of the random→oracle precision gap** —
substantial, not negligible, and best when confidence and the exogenous signal are
combined. So the oracle's measured gain (pubmed +0.9pp seed-stable, cora +3.9pp,
WebKB +7.7 to +10.6pp on tiny test sets) is an upper bound that a real gate would
partially, not wholly, capture.

*What the kept/not-kept decomposition does and does not show.* The split is still
worth reporting, but it means something narrower than originally stated: the benefit
is **local to the nodes the gate acted on** and does not spill over to nodes it
excluded (residual ≈ 0 or slightly negative on the three datasets with usable n;
1–9 nodes on WebKB, i.e. noise). That is expected for transductive gating rather
than evidence against it.

The not-kept comparison remains the study's cleanest harm test, and its reading is
unchanged: on nodes where the teacher is **wrong**, the oracle supplies no
pseudo-label while the control supplies a wrong one, and the oracle does not win
there. Wrong pseudo-labels are not what damages those nodes.

*Net effect on the study's direction.* This is a **positive** result for per-node
gating, not a null: perfect gating is worth +0.9 to +3.9 points on the datasets with
meaningful test sets, and exogenous signals plus confidence recover a large share of
the selection precision needed to chase it. The natural next experiment is a
realisable gate keyed to `z(homophily) + z(confidence)`, measured against
`published` and against this oracle ceiling.

**A14 (2026-08-08) — new exploratory arms `oracle` and `oracle_random`: the ceiling
on per-node gating. Excluded from the §11 verdict.**

§11 asked whether uniform α *harms* an identifiable population. It does not — the
teacher stays better than the student even where it is weakest (§7 analysis). These
arms answer the complementary question the null raises: **if the teacher's wrong
labels could be removed perfectly, how much accuracy would that buy?**

`oracle` restricts the pseudo-label set to nodes where the teacher's argmax equals
the gold label. It uses ground truth, so it is an upper bound rather than a method —
it bounds every realisable gate, including GLEM's own `pl_filter` confidence
filtering and any per-node α predicted from exogenous signals.

`oracle_random` drops the same *number* of pseudo-labels uniformly at random. It is
not optional: the oracle removes ~23% of the pseudo-label set on arxiv, so
oracle-vs-published would confound "removed wrong labels" with "trained on less
pseudo-data". **The interpretable comparison is oracle vs oracle_random.** Matching
is within-run, not across-run — each arm sizes its cut from its own teacher's error
count, so the two agree at the first step and drift afterwards as trajectories
diverge (cornell: 44/44 at step 1, 56 vs 47 by step 2). Forcing identical counts
would require piping one run's numbers into the other.

Neither arm is a control in §11's sense — both carry published pseudo-label weights,
and `report.py` selects the published arm by exact string match with
`CONTROL_ARMS = ('alpha0_li_T', 'alpha0_li_F')`, so both are structurally invisible
to the verdict. Post-hoc, added after seeing the null, and reported as such.

Interpretation fixed in advance: a large oracle gain over `oracle_random` means
per-node gating has headroom and the exogenous signals are worth building on
(homophily predicts GNN teacher error at AUROC 0.845 on arxiv's E-step, against
0.785 for the teacher's own confidence). A negligible gain means uniform α is
near-optimal — which would be a substantive finding in its own right, and one the
§7 relative-margin analysis already predicts.

**A13 (2026-08-08) — WebKB re-run on RevGAT as a parallel set, not a replacement.**
The study was split by backbone: cora/citeseer/pubmed/arxiv on RevGAT, the four
WebKB sets on GCN, because those were the configs in `configs/glem/` at the outset.
That is clean *within* a dataset — every arm shares its dataset's backbone, so no
§11 verdict is affected — but it confounds cross-dataset comparison with six other
hyperparameters (`em_order`, α, β, `pl_filter`, `lm_pl_ratio`, `gnn_label_input`).

The four WebKB configs have now also been run under RevGAT (`cornell.sh` etc.,
2 arms × 3 seeds). **This resolves A4.** The RevGAT recipe uses `lm_pl_ratio=1`
rather than 0.1, so the E-step population rises from 6–11 nodes to **71–101**, and
the `gnn->lm` direction clears §5's n ≥ 30 floor on heterophilous datasets for the
first time — the direction where the GNN-as-teacher failure was most predicted and
which A4/A6 had left untestable everywhere except arxiv.

These runs are archived under `<dataset>+revgat` (see `probe.context.variant`) and
are therefore treated by `report.py` as **distinct datasets**. That is deliberate:
they are a parallel set enabling a same-graph GCN-vs-RevGAT contrast, not a
replacement for the GCN runs, which remain the basis of the existing WebKB verdicts
and of the A12 feature-channel ablation (impossible under RevGAT, whose configs set
`gnn_label_input=F`). Any headline count must therefore state which backbone it
refers to, or it double-counts four datasets.

**A12 (2026-08-08) — new exploratory arm `published_li_F`: the feature-concatenation
channel removed *alone*. Excluded from the §11 verdict.**

§8 defined three arms, which leave one cell of a 2×2 unfilled. The LM teacher reaches
the GNN student by two removable routes — the pseudo-label CE term (weighted by β) and
the concatenation of the teacher's `y_hat` onto the GNN's **input features**
(`datasets.py::node_feature`, gated by `gnn_label_input`):

| | `label_input=T` | `label_input=F` |
|---|---|---|
| **published β** | `published` (Arm 1) | **`published_li_F`** (this amendment) |
| **β = 0** | `alpha0_li_T` (2a) | `alpha0_li_F` (2b) |

Existing arms confound the two routes: `alpha0_li_F` removes both at once, so a
difference between it and `published` cannot be attributed to either. The new arm keeps
the published α and β and removes only the feature concatenation, which — with the three
existing arms — makes the design factorial and lets each channel's contribution be read
off separately.

Scope: meaningful only on the four WebKB configs, the only ones setting
`gnn_label_input=T`. On arxiv/cora/citeseer/pubmed the setting is already `F`, so this
arm would be identical to `published` and is not run. It affects only the **M-step**;
`gnn->lm` events are unchanged, since `label_input` is a GNN-side setting.

**Not a control, and not part of any verdict.** It carries published pseudo-label
weights, so it cannot serve §11's disqualifier clause — `report.py`'s `CONTROL_ARMS`
remains `('alpha0_li_T', 'alpha0_li_F')` and its published-arm selection is an exact
match on `'published'`, so this arm is structurally invisible to the verdict. It is an
exploratory ablation added after the fact, at the experiment owner's request, and is
reported as such.

**A11 (2026-08-07) — exploratory, post-hoc: mean-split sensitivity analysis.
Excluded from the §11 verdict.** Written before the results were computed.

§5 made the **median** the primary split. Local homophily turns out to be heavily
left-skewed on the citation graphs (skew −1.73 cora, −1.50 pubmed, −0.96 citeseer)
with a large tie mass at exactly 1.0. The median lands *on* that tie mass, and §5's
tie rule (`v <= median` → low) then sweeps the whole population into one bin — A6.
A **mean** split sits off the tie mass and stays usable:

| dataset | median split | mean split |
|---|---|---|
| cora | 1084 / **0** (degenerate) | 334 / 750 |
| citeseer | 116 / **0** (degenerate) | 43 / 73 |
| pubmed | 7887 / **0** (degenerate) | 2182 / 5705 |
| arxiv | 39216 / 39186 | 34474 / 43928 |

For kNN ambiguity the two are near-identical (skew 0.16–0.25 on the large datasets;
identical counts on arxiv and pubmed), so this affects the E-step direction only.

**This analysis does not and cannot change any verdict.** §11 states verbatim: *"Do
not adjust bins, thresholds, or arms to move the result."* Switching the primary
statistic after observing that the median degenerated — in a way that restores three
datasets to testability — is the exact researcher degree of freedom preregistration
removes. The legitimate remedy for A6 was the quantile fallback §5 had already
preregistered for this discreteness, and that remains what the verdict rests on.

It is run and reported because *"what would a different, equally defensible
preregistered choice have shown?"* is a fair question about the robustness of a null,
and because burying the answer would be worse than reporting it under a label. Rows
are emitted with `bin_scheme='mean'`; `report.py` filters on `bin_scheme=='median'`,
so the verdict is protected structurally rather than by discipline alone.

The design lesson for any future preregistration: a median split is the **worst**
choice for a signal with a large tie mass at an extreme. §5 saw the discreteness
clearly enough to preregister a quantile fallback, yet still made the median primary.

**A10 (2026-08-07) — arxiv control completed; it does NOT reproduce the E-step
pattern. Two bookkeeping corrections.**

*Result.* With `arxiv alpha0_li_T seed0` finished (58.2 h), the `gnn->lm` comparison
is:

| arm | NCS, in-bias (high hom.) | NCS, out-of-bias (low hom.) | gap |
|---|---|---|---|
| published | **+0.0142** | **+0.0054** | **−0.0089** |
| α=β=0 control | −0.0230 | −0.0229 | **+0.0001** |

The control is **flat across the axis** (gap ≈ 1e-4) while the published arm is
differentiated. Corruption per corruptible node is 5.79× out-of-bias in the published
arm against 2.77× in the control. This is the **first and only** cell in the experiment
where §11's disqualifier does *not* fire — on cora and pubmed the control matched or
exceeded the published arm, and here it does not.

On substance this satisfies **Weakly supported**: NCS positive throughout, significantly
lower out-of-bias (McNemar p=4.9e-6), teacher accuracy degrading across the axis
(0.957 → 0.581), and the pattern absent from the control. It remains scored **no test**
because A7's single seed makes §10's across-seed stability vacuous. That gate is not
waived retroactively.

The `lm->gnn` direction on arxiv behaves like the other seven datasets: published gap
+0.0031 (wrong sign for the hypothesis), and the corruption-concentration ratio is
4.86× published against 5.08× control — disqualifier fires.

*Correction 1 — A9's limitation does not apply to the E-step.* A9 stated the control is
a clean single-variable ablation only at WebKB's iteration-0 M-step. That is right for
the `lm->gnn` direction, where the GNN's input features are LM embeddings. It is **wrong
for `gnn->lm`**: the LM student's inputs are its own text tokens, and the only channel
from the GNN teacher is the pseudo-label file. At α=0 the LM's training is therefore
entirely independent of the GNN, so the E-step control is a clean ablation of the
pseudo-label term **on every dataset**, arxiv included. This strengthens the result
above rather than qualifying it.

*Correction 2 — a damaged `published/seed1` archive was discarded.* The `pkill` used to
stop the arxiv sweep killed the queue's bash loop but not its already-exec'd child, so
that run continued for two further days *concurrently with the control*, sharing GPU 0 —
which is the previously unexplained 3.5× slowdown (4.52 s/it against ~1.3 s/it), now
accounted for. The cleanup step had already deleted that run's directory underneath it
at 20:15, so its `iter-1` baselines were lost and only later files were recreated. It
could have contributed iteration 1 alone; counting it as a second seed would have
manufactured a stability test it cannot support, and would have let arxiv escape A7's
gate on damaged data. Discarded. Concurrency affected wall-clock only, not results:
the runs write to disjoint archive and `glem_cfg_str` paths.

**A9 (2026-08-04) — the α=β=0 control is a minimal one-variable ablation only at
the iteration-0 M-step of the GNN-first configs.** §8 already recorded that no arm
is teacher-free, because the GNN's input features are always the LM's embeddings.
This entry records a second, subtler point that the recorded `feature_file`
provenance makes explicit and that qualifies every cross-arm comparison.

What the runs actually consumed, read from the step records:

| config | M-step | features come from | clean ablation? |
|---|---|---|---|
| WebKB (GNN-first) | iteration 0 | the **pretrained** LM, in every arm | **yes** — β is the only difference |
| WebKB (GNN-first) | iteration 1 | this run's LM | no |
| arxiv (LM-first) | every iteration | this run's LM | no |

Because the control's LM is itself trained with α=0, its embeddings differ from the
published arm's. So wherever the GNN's features come from "this run's LM", the two
arms differ in the GNN's **inputs** as well as in its loss, and the control is
"pseudo-label CE term removed **plus** the consequent representation drift" rather
than a clean single-variable ablation. Under `em_order=LM-first` this applies from
iteration 0, since the E-step runs first.

This does **not** rescue the hypothesis, and the asymmetry is why: §11's clause asks
whether the pattern survives with the pseudo-label loss switched off. A pattern that
appears with that term off is not caused by that term, and additional differences in
the control cannot manufacture the hypothesis's predicted pattern — they can only
add noise. What it does undercut is the **quantitative** cross-arm comparison in A8
(e.g. cora 5.83 published vs 5.22 control), since those two numbers come from runs
whose GNN inputs differ, not only whose losses differ.

The one exactly-clean comparison available is therefore the **iteration-0 M-step of
the WebKB configs**, where features are the pretrained LM's in all three arms. Any
strengthened claim about attribution should rest on that cell, with its n stated —
and its n is small, so the honest reading may be that no exactly-clean, adequately
powered attribution test exists in this design. Recorded so that limitation is
visible rather than inferred.

**A8 (2026-08-04) — exploratory, post-hoc: corruption normalized by corruptible
nodes. Did not change the §11 verdict; the control disqualified it.** Recorded
explicitly as post-hoc because it was computed *after* seeing NCS come out null,
which is exactly the move preregistration exists to constrain. It does not enter
the verdict and does not replace NCS.

Motivation was a genuine confound in NCS, not a search for a positive. NCS divides
by bin size, but corruption is bounded by how many nodes the student had right to
begin with, and student accuracy before the step is **6-19 points lower** in the
teacher-out-of-bias bin on every dataset (cornell -19.5, citeseer -18.7, cora -15.4,
washington -14.2, wisconsin -10.2, pubmed -6.5). So a flat NCS could have been a
headroom artifact rather than an absence of harm. The 2×2 predicted cell was
preregistered to isolate exactly this, but is unavailable here: its student-side
axis is local homophily, degenerate on cora/citeseer/pubmed per A6, and n<30 on the
WebKB datasets.

Conditioning on corruptible nodes (`corruptions / (acc_before x n)`), the published
arm does show elevated corruption where the teacher is out of bias — ratios of
1.26-5.83 on five of six datasets, ~5.8x on cora and ~5.6x on pubmed. **The α=β=0
control reproduces it at the same magnitude or larger**: cora 5.22, pubmed 6.30,
texas 8.95, citeseer 1.15, cornell 1.58, washington 1.76. So the fragility of these
nodes is real but is **not attributable to the pseudo-label term** — it is what
retraining on gold labels alone does to low-margin nodes. This is §11's
disqualifying clause firing on a second metric, and it strengthens the
**Not supported** verdict rather than qualifying it.

Note also that wisconsin, the dataset closest to supporting the hypothesis under
NCS, reverses on this metric (0.41 published vs 0.80/1.16 control) — consistent
with its NCS gap being noise, as its p=0.45 already indicated.

**A7 (2026-08-04) — arxiv reduced to seed 0 of both arms, below §10's ≥3 seeds.**
Cost decision by the experiment owner after the first arxiv run measured at ~13 h
wall (11 h 08 min to reach its fourth distillation event), which put the full 2
arms × 3 seeds at roughly 3 days. The driver is `lm_pl_ratio=1`: each E-step
fine-tunes DeBERTa over all 90.9k gold plus all 78,402 unlabeled nodes, twice per
run, each followed by a 169,343-node inference pass.

Consequence, which is a hard limit and not a caveat: with one seed per arm, §10's
sign-stability requirement **cannot be evaluated on arxiv at all**, so no arxiv row
can reach a **Supported** or **Weakly supported** verdict under §11. arxiv is
therefore run for a narrower purpose — to establish whether the `gnn->lm` direction
is *measurable* on a dataset with both real per-node homophily spread and full
unlabeled coverage, after A4 and A6 left it untestable on all seven other datasets.
Seeds 1-2 remain available to run later if that question comes back positive.

**A6 (2026-08-04) — the median split on local homophily is degenerate on the
homophilous datasets; the E-step direction falls back to the preregistered
quantile scheme there.** Written before inspecting the quantile-binned outcome.

Measured signal distributions: cora, citeseer and pubmed have **median local
homophily = 1.000**, with 64-66% of nodes at exactly 1.0. §5's primary scheme
assigns `v <= median` to the low bin and `v > median` to the high bin, so on those
datasets the high bin is **empty** and the low bin holds the entire analysed
population. The four WebKB datasets have median local homophily = 0.000 (q75 = 0.00
on cornell and texas), giving a valid but heavily unbalanced split.

This is not a null result, it is a **missing test**: a comparison with one empty
cell measures nothing, so there is nothing for the §11 rule to be applied to. It
also invalidated the first pooled table computed from these runs, in which the
apparent low-vs-high teacher-accuracy contrast was in fact cora/citeseer/pubmed
(all in "low") versus WebKB (supplying every "high" row) — a dataset contrast
wearing an axis label.

Remedy, and why it is not a post-hoc choice of a favourable test: §5 already
preregistered **5 quantile bins with duplicate edges collapsed** as the secondary
scheme, precisely because "local homophily is very discrete on low-degree graphs".
For the `gnn->lm` direction that secondary scheme becomes the reported one on any
dataset where the median split yields an empty bin. The median split remains
primary for `lm->gnn`, where the kNN ambiguity signal is well spread (per-dataset
medians 0.32-0.73, quartiles well separated) and both bins are populated.

`probe.analyze.median_bins` is additionally being made tie-aware so it reports an
empty-bin degeneracy explicitly instead of silently returning a one-sided split.
The affected rows are being recomputed from the existing archive; no retraining is
needed, since the archive stores logits rather than derived bins.

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
