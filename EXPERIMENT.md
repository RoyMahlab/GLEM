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

### 13.1 The preregistered verdict: **Not supported**

Of 24 (dataset, direction) cells, **13 are Not supported, 10 admit no test, and 1 is
Weakly supported.** Uniform α does not produce net harm on the population §4
identifies. Across 8 datasets and 3 seeds, students in the out-of-bias bin gain more
than they lose.

The single Weakly supported cell is **arxiv `gnn→lm`**: NCS is positive throughout but
lower out-of-bias (+0.0062 against +0.0170 in-bias), a gap of **−0.0108 ± 0.0017**,
exact McNemar p = 6.2 × 10⁻¹⁹, while teacher accuracy falls from 0.957 to 0.580 across
the axis. It is *Weakly* rather than Supported because NCS never turns negative, and
because the α=β=0 control's gap has the same sign with an unstable magnitude (A17).

The 10 no-test cells are not silence about the hypothesis; they are the cost of the
datasets available. Seven fail because the median split on local homophily is
degenerate (A6) or every bin is under n=30 (A4); the remainder because a bin is absent.

### 13.2 Why the null holds, and why the first explanation was wrong

An early reading — that the teacher stays more accurate than the student even out of
its bias — is **true only for `gnn→lm`**, and there only barely on arxiv (0.580
against 0.572, a 0.9pp edge). In the `lm→gnn` direction the teacher is *worse* than the
student in **12 of 12 cells**, by 41pp on cornell and 20pp on citeseer, and NCS is
still non-negative in 5 of those 12. That explanation does not survive.

What does explain it is the weighting GLEM already ships. α runs 0.50–0.80 for
`gnn→lm`, where the teacher is usually the stronger model; β runs 0.05–0.70 for
`lm→gnn`, where it is always the weaker one. **GLEM's asymmetric α/β already performs,
between steps, the down-weighting a per-node gate was meant to perform within one.**
That is why there is little left for node selection to recover.

The harm §4 predicted is nonetheless real and measurable: students adopt the teacher's
*specific* wrong label at 1.7–5.8× the retraining baseline. It does not net out,
because the nodes where the teacher fails are nodes that are hard for every model —
on arxiv **88% of the GNN teacher's errors are also the LM teacher's errors**, and
P(LM right | GNN wrong) = 0.12. The out-of-bias bins identify hard nodes, not one
modality's blind spot, and removing them removes as much signal as noise.

### 13.3 The feature channel: **Supported** (A19/A20)

The hypothesis holds in a channel §4 did not measure. When the teacher's `y_hat` is
concatenated onto the GNN's **input features** (`gnn_label_input=T`) rather than only
weighted into its loss, arxiv `lm→gnn` out-of-bias NCS turns negative on every clause
of A19's rule:

| clause | observed |
|---|---|
| placebo `gnn→lm`, iteration 0 (A20) | **bit-identical** to `published` (+0.0087; equal per seed) |
| out-of-bias NCS | **−0.0073** (−0.0037 / −0.0078 / −0.0104) |
| exact McNemar p | **5.1 × 10⁻³²** |
| gap against `published` | **−0.0108**, sign stable (−0.0079 / −0.0119 / −0.0127) |

And unlike §13.1, the harm is **concentrated where the teacher is weak** — the
out-of-bias minus in-bias gap is −0.0056 for `published_li_T` against +0.0030 for
`published`, with teacher accuracy 0.629 out-of-bias against 0.877 in-bias. The damage
tracks teacher wrongness rather than the mechanism, which is what licenses attributing
it to the labels being wrong.

It is not confined to NCS. Final GNN test accuracy, 3 seeds:

| | `li=F` | `li=T` | feature effect |
|---|---|---|---|
| loss on (α=.8, β=.05) | 0.7677 | 0.7556 | **−1.21pp, t = −6.93** |
| loss off (α=β=0) | 0.7561 | 0.7336 | **−2.25pp, t = −19.15** |

Reading the margins: the loss channel **helps** (+1.16pp), the feature channel
**hurts**, and removing the loss channel nearly doubles the feature channel's damage.
β = 0.05 partially protects against the very labels it delivers.

**Scope, stated as narrowly as the design permits.** This is one direction by
architecture — no LM code path consumes pseudo-labels as input, so the channel does not
exist in the E-step. It is `li=T`, which **no upstream GLEM config sets**, so it does
not bear on the 76.97 reproduced in §13.6. The claim is about *cross-model pseudo-label
reuse*, adjacent to but not identical with the gold-label masked reuse of UniMP and
"Bag of Tricks".

### 13.4 Why the feature channel harms: the reliability mismatch (A21)

§13.3 established the cost but not the cause. A21 registered three arms that alter
*only* what the label-feature vector contains, and the one targeting the train/inference
reliability gap removes essentially all of the damage.

| arm | GNN test acc | out-of-bias NCS | recovery of the li=T damage |
|---|---|---|---|
| `published` (li=F) | 0.7677 | +0.0035 | — (the target) |
| `published_li_T` | 0.7556 | −0.0073 | — (the damage) |
| **`teacher_consistent`** | **0.7680** | **+0.0032** | **102% / 97%** |
| `mask_pseudo` (2 seeds) | 0.7635 | +0.0007 | 53% / 65% |
| `mask_train` | 0.7606 | −0.0032 | 41% / 38% |

`teacher_consistent` changes nothing except deleting the gold overwrite, which closes
the channel's reliability gap from 24.5pp to −0.5pp. Its recovery against
`published_li_T` is sign-stable on all three seeds (+0.0078 / +0.0149 / +0.0146), and
it restores both measures simultaneously: accuracy to within 0.03pp of `published`, and
out-of-bias NCS from −0.0073 back to +0.0032 against `published`'s +0.0035.

**The half-measures are what make the attribution airtight.** Masking either half of the
vector alone recovers only 38–65%. It is specifically *matching the reliability* of the
channel between training and inference that repairs it, not removing information from
it. Three arms, one mechanism, and only the arm aimed at that mechanism works.

Against A21's registered outcomes this is **Mechanism identified**, and explicitly
**not** *Method*: `teacher_consistent` sits +0.03pp above `published` with an unstable
sign across seeds, i.e. indistinguishable from the shipped default. A21's prediction —
"recovers most of the 1.21pp but does not exceed `published`" — is what happened.

**What this licenses, and it is the paper's central claim.** Cross-model label reuse is
harmful *because it is unmasked*, not because the labels come from another model. The
GNN is trained on a label channel that is 100% reliable and evaluated on one that is
75.5% reliable; equalising the two removes the entire cost. UniMP's masked label
prediction and the label reuse of "Bag of Tricks" each achieve matched reliability by
different means, which is why the published forms of this technique are safe and GLEM's
is not. The finding is a condition under which a standard technique is sound, not a
defect in one configuration flag.

**Scope condition, added after A23 arm C (A24).** The account above is **conditional on
the labelled fraction**, not general. On wikics — 5% train split against arxiv's 53.7% —
the same harm appears and is *four times larger* (−4.59pp), but `teacher_consistent`
recovers only **41%** of it, because there is almost no gold in the channel to overwrite.
The unreliability is the same quantity in both cases; only on arxiv is it concentrated
in a discontinuity that equalising can remove. The general statement is that **harm
scales with the unreliable fraction of the channel**, and the mismatch repair works in
proportion to how much of that channel is gold. See A24.

**And on arxiv it closes the feature channel as a source of gain.** `teacher_consistent`
returns to `published` and no further. Once the mismatch is removed there is no residual harm
for a per-node gate to target and no accuracy above `li=F` to be had, so the contingent
feature-gate follow-up registered in A19 is withdrawn as unmotivated (A22). The
practical recommendation remains `gnn_label_input=F`, which is what every shipped GLEM
recipe already sets — the contribution is knowing *why*.

### 13.5 Headroom that exists and cannot be reached

**Perfect gating pays.** Against a size-matched random control the oracle gains
**+3.37pp (GNN) / +4.37pp (LM)** on arxiv (t = 22.5 / 115.9), and is positive on **17
of 18 dataset × model cells**, significant on arxiv, pubmed (+0.94 / +0.98pp) and cora.
Headroom is not an arxiv artifact.

**No deployable gate reaches it.** Confidence filtering (`pl_filter`, A16) recovers ≈0
against the +0.8pp predicted, on arxiv and on pubmed. The reason is diagnostic and is
the cleanest negative result here: at the oracle keep-rate a confidence gate excludes
51% of the teacher's errors but **0.0%** of the *confidently wrong* ones. It strips the
harmless errors and leaves the damaging ones. Exogenous signals (A18) do reach
11.6–17.7% of that population and raise kept-set teacher accuracy by 5.0–6.4pp, yet
convert it to only +0.23pp (GNN, t = 2.21) / +0.32pp (LM, t = 0.45) against the
size-matched control — under a tenth of the oracle's gain. Every loss-channel gate also
pays a shrinkage tax: dropping ~24% of pseudo-labels at random costs 0.20pp (GNN) /
1.00pp (LM), which is why gates that beat random still lose to `published`.

The precision→accuracy relationship is sharply sublinear: 75.7%→82.1% teacher precision
buys +0.23pp, while 82.1%→100% buys the remaining +3.1pp.

### 13.6 Fidelity (§12)

`published`, arxiv, seed 0: **test 0.76997**, against the paper's 0.7697 ± 0.0019 for
RevGAT+GLEM. Val 0.77593 against 0.7749 ± 0.0017. The instrument does not perturb the
method it measures.

### 13.7 Not established

- **Whether the label-feature channel can be made to *help*.** On arxiv its harm is
  fully removable but recovers nothing above `li=F`, so the A19 gate was withdrawn
  (A22). A24 narrows that to **high-labelled-fraction settings**: on wikics 59% of the
  harm survives the repair, so a gate has something to target there. Neither implemented.
- **The fraction-scaling law.** A24 predicts harm ∝ (teacher share) × (1 − teacher
  accuracy); two points match to within 20%. `bookhis` (60% train split, both pretrains
  on disk) is the cheapest third point and is not registered.
- **`mask_pseudo` at 3 seeds.** Two of three landed; the third is outstanding. It does
  not affect the §13.4 verdict, which rests on `teacher_consistent`.
- **The low-label regime.** Arm 3 was withdrawn (A5). §8 predicted harm is strongest
  where the gold CE term is too weak to anchor the student, and every null in §13.1 is
  at ~54% labelled. **The largest untested lever.**
- **Whether `sig_gate80` beats `published`.** Its LM effect rests on 2 seeds;
  `published` LM varies 1.8pp across 3. Unresolved, not negative.
- **Large heterophilous TAGs.** WebKB is 18–26 test nodes; arxiv is homophilous.
- **Attribution on the M-step.** The α=β=0 control is a clean single-variable ablation
  only for the E-step; on the M-step the GNN still consumes LM embeddings (A9/A10).

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

**A25 (2026-08-21) — the homophily gate on GNN-as-Judge's disagreement set. Run in a
separate repository against that method's own code; registered here because the
hypothesis, the signal and the decision rule are this study's.**

GNN-as-Judge (few-shot semi-supervised learning on TAGs) splits unlabeled nodes by
LLM/GNN agreement. On the **disagreement** set it applies ORPO preference tuning with
the **GNN's** prediction as preferred and the LLM's as dispreferred — uniformly, with no
per-node weighting. That is the condition §13.2 identifies as the one GLEM does *not*
satisfy: GLEM discounts the weak direction with β = 0.05, and this framework does not
discount at all.

**Motivating analysis, and it is post-hoc.** Computed from GLEM's own teacher snapshots
on arxiv — *not* from GNN-as-Judge — by splitting its disagreement rule along local
homophily. Three seeds, `published` arm:

| signal | disagreement subset | share | GNN right | LM right | value of preferring the GNN |
|---|---|---|---|---|---|
| oracle (gold-label homophily) | low | 78% | 0.362–0.384 | 0.343–0.360 | **+0.004 – +0.042** |
| oracle | high | 22% | 0.747–0.774 | 0.148–0.181 | **+0.57 – +0.63** |
| **GLANCE (label-free)** | low | **87%** | 0.418–0.435 | 0.316–0.332 | **+0.090 – +0.119** |
| **GLANCE** | high | 13% | 0.661–0.691 | 0.197–0.248 | **+0.41 – +0.49** |

The oracle row is a **ceiling, not a result**: `probe.signals.local_homophily` uses gold
labels for every node and is not deployable. The gate below uses GLANCE soft homophily
(`soft_local_homophily`), which needs no labels, and on which the separation is ~4.5×
rather than ~30×.

Also recorded, as an empirical check on that paper's Theorem 2 rather than a criticism
of its proof: on arxiv the agreement set is **91% of nodes** — so it barely selects —
gains **+2.9pp** over the GNN alone, and **20.4% of it is both models wrong with the
same label**. 88% of the GNN's errors are also the LM's (§13.2), and correlated errors
are the regime in which agreement-based selection is weakest.

**Arms.** All three run on GNN-as-Judge's own code, its own datasets, its own
hyperparameters. Nothing from this repository is transplanted except the signal.

| arm | disagreement set treated how |
|---|---|
| `published` | as that paper ships it — ORPO on every selected disagreement node |
| `judge_gate` | ORPO only on nodes with GLANCE soft homophily **above the median** of the disagreement set; the rest are excluded from ORPO |
| `judge_gate_random` | **size-matched**: exclude the same *number* of disagreement nodes, chosen uniformly at random |

**The random control is not optional, and is the thing most likely to be skipped.** The
gate removes ~87% of disagreement nodes, and shrinking a training set changes the result
on its own — on GLEM, dropping ~24% of pseudo-labels at random cost 0.20pp (GNN) and
0.68pp (LM) *before any selection effect*. A14 exists because of exactly this. The
interpretable comparison is **`judge_gate` against `judge_gate_random`**; `judge_gate`
against `published` measures selection *and* shrinkage together and must be reported
separately, never instead.

**Signal definition, fixed so the two repositories compute the same thing.** GLANCE
`h_v = p_v · mean_{u∈N(v)} p_u` where `p` is the softmax of the **GNN's** logits over
the same graph, following `probe.signals.soft_local_homophily`; isolated nodes yield NaN
and are imputed with the median before ranking, as `probe.gating` already does; the
threshold is the median **over the disagreement set**, recomputed per run, never
transplanted from arxiv.

**Prediction, fixed before any GNN-as-Judge result is seen.** `judge_gate` beats
`judge_gate_random` by **+0.5 to +2.0pp** on the LLM's test accuracy, and beats
`published` by **0 to +1.5pp**. The second interval starts at zero deliberately: on the
low-homophily subset the GNN is still marginally better than the LLM (+0.09 by GLANCE),
so excluding those nodes discards a small amount of real signal along with a large
amount of noise, and which dominates is genuinely open. Few-shot baselines are lower
than GLEM's, so effects should be larger in absolute terms than anything in §13.

**Decision rule.**

- **Supported** — `judge_gate` > `judge_gate_random` with sign-stable per-seed
  differences over ≥3 seeds, **and** `judge_gate` ≥ `published`. The uniformity
  assumption costs measurable accuracy in a framework that does not discount its weak
  teacher, and an exogenous signal recovers it.
- **Weakly supported** — beats the random control sign-stably but not `published`. The
  signal selects, but not enough to pay for the shrinkage. Report as a selection result,
  not a method.
- **Not supported** — fails against the random control. Then the disagreement split is
  not exploitable by this signal, and §13.1's null extends to a framework with no
  protective weighting at all — which would make the null considerably stronger than it
  currently is.

**What a null here would mean, stated in advance so it cannot be reframed later.**
§13.2 attributes the §13.1 null to GLEM's asymmetric α/β. If harm is *also* absent where
no such protection exists, that attribution is wrong and §13.2 must be rewritten. This
arm can therefore falsify the explanation this study currently rests on, and that is why
it is worth running.

**A24 (2026-08-21) — A23 arm C: the feature-channel harm replicates on wikics, its
mechanism does not, and §13.4's account is corrected from *arxiv-specific* to
*split-conditional*.**

A23 required both clauses to hold. **Clause 1 holds, clause 2 fails, so arm C is Not
replicated** — and the failure locates a second component of the mechanism that arxiv
could not have exposed.

| | arxiv | wikics |
|---|---|---|
| `published` GNN | 0.7677 | 0.7861 |
| `published_li_T` GNN | 0.7556 (**−1.21pp**) | 0.7402 (**−4.59pp**, t = −4.84) |
| `teacher_consistent` recovery | **102%** | **41%** |

Clause 1 replicates and then some: every one of six seed-differences is negative and the
harm is nearly **four times larger** than on arxiv. Clause 2 fails: the repair that
recovered everything on arxiv recovers 41% here.

**The diagnostic, and it is not a property of the datasets but of their splits.**

| | gold share of the label vector | teacher accuracy (train / eval) | share `teacher_consistent` alters |
|---|---|---|---|
| arxiv | 53.7% | 0.750 / 0.755 | 53.7% |
| wikics | **5.0%** | 0.666 / 0.625 | **5.0%** |

wikics has a 5% train split. There is almost no gold in the channel to overwrite, so
deleting the overwrite changes 5% of the vector; the remaining 95% is teacher prediction
at 0.625 accuracy. On arxiv the unreliability is concentrated in a *discontinuity*
(perfect gold beside a 0.755 teacher) and removing the discontinuity removes the harm.
On wikics there is no discontinuity worth removing — the channel is simply noisy.

**§13.4's claim is therefore corrected.** It reads as though the reliability mismatch is
*the* mechanism. It is one of two, and which dominates is set by the **labelled
fraction**, not by the dataset:

> Harm in the feature channel scales with the *unreliable fraction* of the channel.
> Where gold occupies a large share, that unreliability appears as a train/inference
> discontinuity and equalising the two removes it. Where gold is a small share, there is
> no discontinuity to equalise and the harm is irreducible noise.

Quantitatively, with the obvious caveat that two points determine a line:

```
expected wrong entries = (teacher share) x (1 - teacher accuracy)
  arxiv    0.463 x 0.245 = 0.113   ->  observed harm 1.21pp
  wikics   0.950 x 0.375 = 0.356   ->  observed harm 4.59pp
  ratio             3.14           ->  ratio         3.79
```

The ratio is *predicted by the mechanism* rather than fitted to the outcome, which is
why it is recorded despite n = 2. A third point would test it; `bookhis` (41.5k nodes,
60% train split, both pretrains already on disk, ~2–3h per run) is the cheapest
available and is **not** currently registered.

**Credit where due:** the correction from "arxiv-specific" to "split-conditional" was
the user's, not the analysis's. The first reading treated a scope *condition* as a
*limitation*, which is weaker and less testable.

**Consequence for A21/A22.** A22 withdrew the contingent per-node feature gate on the
grounds that `teacher_consistent` leaves no residual harm. That holds **on arxiv only**.
On wikics 59% of the harm survives the repair, so a gate has something to target there
after all. The withdrawal is narrowed to high-labelled-fraction settings rather than
reversed; nothing is implemented either way.

**A23 (2026-08-20) — three arms addressing what §13.4 leaves open: `unimp_mask`,
`beta_high`, and the feature-channel result replicated on wikics.**

§13.4 identified the mechanism but leaves the study with three weaknesses, and each of
these arms targets exactly one. Predictions and decision rules are fixed here, before
any of them runs.

---

**Arm A — `unimp_mask`. Can the channel be made to *help*?**

§13.4 closes the feature channel as a source of harm and simultaneously as a source of
gain: `teacher_consistent` reaches `published` and stops. But it gets there by
*discarding* the gold labels, which is not what the literature does. UniMP keeps them
and masks a subset; the channel still carries real label information.

`unimp_mask` keeps gold on a random **half** of train nodes and zeroes everything else —
the other half of train, and every unlabelled node. The channel is then gold-or-empty in
training and gold-or-empty at inference, i.e. **matched**, while still carrying real
labels rather than none.

*Honest scope: this is not UniMP.* UniMP redraws its mask every step and predicts only
the masked nodes; GLEM builds `self.features` once per M-step
(`gnn_trainer.__init__`) and its loss targets are fixed independently, so neither is
reachable without modifying the trainer. The mask here is **static**, drawn once from
the run seed. It is channel-matched, which is the property under test, but the paper
must say "the mechanism UniMP's masking exists to prevent", never "we apply UniMP".

**Prediction: `unimp_mask` lands between `published` and +0.5pp above it (0.7677 to
0.7727 GNN).** Label reuse is on the ogbn-arxiv leaderboard because it pays, and this
is the first arm that both matches the channel and retains label information. Failing to
exceed `published` would mean the channel is worth nothing on arxiv under *any*
treatment, which is a stronger negative than §13.4 currently supports.

**Decision rule.** *Method* — exceeds `published` with sign-stable per-seed differences
over 3 seeds. *Channel worthless* — at or below `published`. Nothing in between.

---

**Arm B — `beta_high`. Does the loss channel harm when it is not down-weighted?**

§13.2 explains the entire §13.1 null by GLEM's asymmetric weights: α = 0.50–0.80 where
the teacher is usually stronger, β = 0.05–0.70 where it is always weaker. **That
explanation has never been tested.** β has not been varied in any arm; the claim rests
on a correlation between two things that were never manipulated.

`beta_high` sets `--gnn_pl_weight=0.8`, matching α, and changes nothing else.
`gnn_label_input` is untouched, so on arxiv this is `li=F` and the feature channel is
absent — the loss channel alone, at feature-channel-like exposure.

**Prediction: out-of-bias NCS in `lm→gnn` turns negative, with the out-of-bias minus
in-bias gap negative and sign-stable, and final GNN accuracy falls by 1–3pp.** On arxiv
the LM teacher is 0.629 against a 0.648 student out-of-bias; weighting a worse teacher
sixteen times more heavily should transmit its errors.

**Decision rule.** *§13.2 confirmed* — out-of-bias NCS < 0 with a negative, sign-stable
bin gap. Harm is then a function of **exposure**, and §13.3's feature-channel result and
§13.1's loss-channel null become one principle rather than two findings: the same wrong
label is harmless or harmful according to how strongly the student is made to attend to
it. *§13.2 refuted* — NCS stays positive at β = 0.8. The loss channel is then robust for
some reason other than its weight, and §13.2 must be rewritten.

This is the arm whose result I would otherwise be able to explain either way, which is
why the rule is written down first.

---

**Arm C — wikics. The feature-channel result on a second dataset.**

§13.3 and §13.4 rest on **one dataset**, against a §13.1 null spanning eight. That is
the study's largest asymmetry and the cheapest to fix: wikics is the only unused TAG set
with a non-degenerate homophily axis (median 0.746 against cora/citeseer/pubmed's
1.000), both pretrains are already on disk, and runs are ~1h against arxiv's 11.4h.

Arms: `published`, `published_li_T`, `teacher_consistent`, `alpha0_li_T`, 3 seeds — 12
runs, ≈12 GPU-hours, a third of one arxiv arm.

This **extends A19's scope**, which registered the feature-channel arms for arxiv only.
The extension is deliberate and is recorded here rather than taken silently.

**Prediction: `published_li_T` < `published` on wikics with a sign-stable per-seed
difference, and `teacher_consistent` recovers a majority of it.** Magnitude is not
predicted — wikics has 10 classes against arxiv's 40 and a 5% train split against 54%,
so the label-feature vector is a different shape and a different fraction of it is gold.

**Decision rule.** *Replicated* — both clauses hold. *Not replicated* — either fails.
A null here does not overturn §13.3, which stands on its own placebo-controlled arxiv
evidence, but it does confine the claim to arxiv and that must be stated in §13.3 rather
than buried.

---

**Not §11 controls.** All three carry either a non-`published` `gnn_label_input` or a
non-`published` β, so `report.py` — which matches `published` exactly — leaves the
existing verdicts untouched.

**A22 (2026-08-20) — A21's result, one sub-prediction missed, and the contingent
feature gate withdrawn.**

A21's primary prediction held: `teacher_consistent` recovered most of the 1.21pp and
did not exceed `published` (§13.4). Two things it got wrong or left open are recorded
here rather than absorbed silently.

**1. `mask_pseudo` was registered as the expected weakest of the three. It was not.**
A21 argued it would be worst because it *widens* the reliability gap to 1.000 against
nothing. Measured, it recovers 53% of the accuracy damage and 65% of the NCS damage,
against `mask_train`'s 41% and 38% — second of three, not third. (Both figures are
seed-paired over the two seeds it has; an unpaired 2-seed-against-3-seed comparison
inflates them to 65% / 74%.) The margin is small
and `mask_pseudo` currently has two seeds against `mask_train`'s three, so this is a
missed ordering rather than a reversed mechanism; the arm that mattered,
`teacher_consistent`, was correctly predicted. Recorded because a preregistration that
only reports the predictions it got right is not one.

**2. The contingent per-node feature gate (A19) is withdrawn as unmotivated.** A19
registered it conditional on the feature channel showing harm, which §13.3 established.
But §13.4 then showed the harm is *fully* attributable to the reliability mismatch and
*fully* removed by correcting it: `teacher_consistent` reaches 0.7680 against
`published`'s 0.7677. There is no residual harm left for a per-node gate to target, and
no accuracy above `li=F` available from the channel at all. Running it would be
measuring a remedy for a fault that no longer exists. The withdrawal is a consequence of
the result, not a change of mind about the design, and the gate remains unimplemented —
`y_hat` was never modified, only the feature-side call path (A21).

What stays open is narrower and is recorded in §13.7: whether a *masked* label channel
could beat `li=F` on a dataset where label reuse is known to pay. arxiv is not that
dataset — here the channel is worth exactly nothing once repaired.

**A21 (2026-08-19) — three new arms on the label-feature channel:
`teacher_consistent`, `mask_pseudo`, `mask_train`. The comparator is `published`,
not `published_li_T`. Excluded from the §11 verdict.**

§13.3 established that routing the teacher's predictions through the GNN's input
features costs 1.21pp. It did not establish **why**, and the difference matters: the
answer determines whether the finding is about a configuration flag or about a
condition under which a widely-used technique is safe.

**The mechanism these arms test.** `y_hat` fills the label-feature vector with the
teacher's softmaxed prediction and then **overwrites it with the gold one-hot on train
nodes**. Nothing masks it — the only `mask` in the codebase is the tokenizer's
attention mask. So the channel the GNN learns from and the channel it meets at
inference have very different reliability. Measured on arxiv, LM teacher accuracy:

| | in training (train nodes) | at inference (val ∪ test) | gap |
|---|---|---|---|
| as shipped (`li=T`) | **1.000** (gold) | 0.755 | **24.5pp** |
| `teacher_consistent` | 0.750 | 0.755 | **−0.5pp** |

The teacher is 0.750 on train nodes and 0.755 on evaluation nodes — near-identical,
because the LM is a 3-epoch fine-tune that does not memorise its train split. So
removing the gold overwrite closes the gap almost exactly, changing nothing else.

That predicts the pattern §13.3 measured: harm concentrated where the teacher is least
accurate (0.629 out-of-bias against 0.877 in-bias), because that is where the channel
deviates most from what the model learned to trust.

**Arms.** Registered for **arxiv only**.

| arm | train-node entry | val/test entry | isolates |
|---|---|---|---|
| `teacher_consistent` | teacher's prediction | teacher's prediction | the reliability gap alone |
| `mask_train` | zeroed (UniMP-style) | teacher's prediction | the readable-shortcut alone |
| `mask_pseudo` | gold | zeroed | the unreliable half alone |

`teacher_consistent` is the arm that answers the reviewer objection this amendment
exists for, and it runs first. **`mask_pseudo` is expected to be the weakest of the
three** and is registered to test that expectation rather than because it is promising:
it *widens* the reliability gap to 1.000 against nothing. An earlier informal
recommendation to run it first is **withdrawn** — it was made before the table above
was computed.

Note that none of the three removes supervision. Gold labels remain in the GNN's loss
throughout; only their presence as a *readable input feature* changes. This is the
distinction UniMP's masked label prediction rests on.

**Implementation, recorded because the obvious version would be wrong.** `y_hat` is
also the **loss target** — `node_labels`, `gnn_trainer.pseudo_labels` and `get_tokens`
all call it. Transforming `y_hat` itself would change the objective, not the feature
channel, and would silently make these arms a different experiment. The transform is
therefore applied in `SeqGraph._label_feature`, reached only from `node_feature`, and
`y_hat` gains one additive keyword (`overwrite_gold=True`) whose default reproduces its
previous body exactly. With `GLEM_PROBE_LABELFEAT` unset the code path is unchanged.

**Decision rule, fixed before any of these arms is run.**

The comparator is **`published` (0.7677 GNN, 3 seeds)** — *not* `published_li_T`
(0.7556). Beating `published_li_T` would only mean undoing damage introduced by a flag
that GLEM's own recipes set to `F`, which is available for free. Stated explicitly here
because it is the difference between a contribution and a repair.

- **Mechanism identified** — `teacher_consistent` recovers a majority of the 1.21pp gap
  against `published` (i.e. lands at or above ≈0.764 GNN), with sign-stable per-seed
  differences over 3 seeds. The harm is then attributable to the reliability mismatch,
  and the safe-usage condition is stated rather than guessed.
- **Mechanism rejected** — `teacher_consistent` stays at or below `published_li_T`. The
  harm is cross-model label input as such, independent of the mismatch.
- **Method** — any arm exceeds `published` with sign-stable per-seed differences. Only
  this outcome licenses a recommendation rather than a warning.

**Prediction fixed in advance: `teacher_consistent` recovers most of the 1.21pp but does
not exceed `published`.** Closing a 24.5pp reliability gap should remove the mismatch
penalty, while replacing gold with a 75%-accurate signal in the feature channel gives
up real information that `li=F` never had to give up. A result above `published` would
be a genuine surprise, and is registered as such.

**A limitation found while smoke-testing these arms, and it applies to every result in
§13.** GLEM's GNN training is **not deterministic run-to-run**: two runs of the
`published` arm at the same seed, same code, same GPU produce different GNN logits
(`iter0_lm`, the one step with no GNN involvement, is identical). Every "seed sd"
reported in §13 therefore conflates seed variance with run-to-run variance. This is
conservative for the paired comparisons — the noise is included, so effects that clear
it are real — but it means no effect smaller than run-to-run noise is detectable at all,
and that a repeated run is not expected to reproduce a previous one exactly. The
magnitude of that noise on arxiv is unmeasured; quantifying it would cost one duplicate
run (11.4h) and has not been spent.

**A20 (2026-08-19) — A19's placebo clause needed scoping to iteration 0, and an
analysis error found and corrected before reporting. A19's text is left unedited.**

A19 was committed (`14e304a`) before any `li=T` data existed, and that timestamp is
the only thing making it a preregistration. It is therefore **not** edited here; the
two corrections it needs are recorded as this separate entry.

**1. The placebo clause was under-specified, and unscoped it would have voided a real
result.** A19 requires that `gnn→lm` be "unchanged", on the grounds that
`node_feature()` runs only at the M-step. That is true per step but false per *run*:
the EM loop couples the two directions, so at iteration 1 the E-step's teacher **is**
the GNN that the iteration-0 M-step just trained with `li=T`. Movement at iteration 1
is the mechanism propagating, not the manipulation leaking.

Measured: at iteration 0 the `gnn→lm` out-of-bias NCS is **bit-identical** between
`published` and `published_li_T` (+0.0087 both, and equal per seed at +0.0098 /
+0.0109 / +0.0055), because that step's teacher is the shared pretrained GNN and α is
unchanged. At iteration 1 it shifts by −0.0098. Read unscoped, that shift would have
failed the placebo and discarded the result; read correctly, iteration 0 is the only
step at which the placebo is identifiable at all.

**The clause is therefore scoped to iteration 0** — the reading the mechanism dictates
and the only one under which the test is well-posed. This is a resolution of an
ambiguity in A19's own wording, in the same category as A17, and it is disclosed
because it changes the verdict from "not supported" to "supported".

**2. An analysis error, caught before reporting.** The first pass over `ncs_long.csv`
filtered on `bin_scheme` but not on `signal`, averaging NCS across all three cached
axes. For `lm→gnn` two of those axes are homophily-based, where the out-of-bias bin has
the teacher *more* accurate, not less; the contamination inverted binned teacher
accuracy (0.824 out-of-bias against 0.687 in-bias — backwards) and corrupted the NCS
values. It was caught by that inversion, and every A19 number in §13 is computed on the
§4 axis for the direction in question — `local_homophily` when the GNN teaches,
`knn_ambiguity` when the LM teaches — matching `report.py`. Recorded because the log is
meant to be the honest record, not only of protocol changes but of what nearly went
into it.

**A19 (2026-08-16) — new exploratory arms `published_li_T` / `alpha0_li_T_only`:
the pseudo-label *feature* channel. Post-hoc hypothesis, decision rule fixed before
the arxiv run. Excluded from the §11 verdict.**

**This hypothesis is post-hoc and is labelled as such.** It was found by slicing
§13's existing null by `gnn_label_input` *after* those results were known — the
practice §1 exists to constrain. It is recorded here so that the arxiv test is
preregistered even though the hypothesis that motivated it is not.

The slice — `published` arm, out-of-bias bin, `lm→gnn` direction, mean over seeds:

| `gnn_label_input` | mean out-of-bias NCS | cells negative |
|---|---|---|
| `F` — pseudo-label enters the **loss** only | +0.0047 | 2 / 8 |
| `T` — loss **and** input features | **−0.0253** | **4 / 4** |

**Three confounds, all present, none separable in the existing data:**

1. every `T` cell is WebKB at n = 23–37 per bin — the regime A4 flagged as
   underpowered and A13 only partly resolved;
2. every `T` cell uses GCN and every `F` cell RevGAT — the backbone varies with the
   channel;
3. the four `T` configs (`configs/glem/*_gcn.sh`) were written for this study and are
   untracked. **No upstream GLEM config sets `T`.**

The slice is therefore a lead, not a finding, and it cannot be strengthened by adding
more datasets of the same kind.

**Mechanism, stated before the test.** `y_hat` is
`softmax(pseudo_labels / temperature)` with gold one-hots overwriting it on train
nodes, concatenated onto the GNN's input features
(`src/utils/data/datasets.py::node_feature`). A wrong pseudo-label in the **loss** is
one term scaled by β — 0.05 for the M-step on arxiv. A wrong pseudo-label in the
**features** is read at inference time for that node and enters its representation
with no such scaling. §13's null is explained by GLEM's asymmetric α/β down-weighting
the direction in which the teacher is weak (α = 0.50–0.80 for `gnn→lm`, β = 0.05–0.70
for `lm→gnn`, where the teacher is worse than the student in 12 of 12 cells). The
feature channel has no equivalent protection. That asymmetry is the reason to expect a
different answer here, and it is the only reason.

**Not a GLEM idiosyncrasy.** Concatenating predicted labels onto input features is
*label reuse* — Wang et al., *Bag of Tricks for Node Classification with GNNs* (2021);
Shi et al., *UniMP: Masked Label Prediction* (IJCAI 2021) — standard practice on
ogbn-arxiv, from which this study's primary dataset is drawn. `label_input = 'T'` is
GLEM's own framework default (`src/models/GNNs/gnn_utils.py:32`); every shipped config
overrides it to `F`. If the effect holds it is a statement about a widely-used
technique, not about a configuration flag.

**Design.** Registered for **arxiv only**.

| arm | α, β | `gnn_label_input` | role |
|---|---|---|---|
| `published` (exists, 3 seeds) | published | F | reference — feature channel **off** |
| `published_li_T` | published | **T** | treatment — both channels active |
| `alpha0_li_T_only` | 0, 0 | **T** | attribution — feature channel **alone** |

`published_li_T` against the existing `published` holds dataset, backbone, α, β,
splits and seeds fixed; `gnn_label_input` is the single manipulated variable, at
n ≈ 38,800 per bin against the lead's 23–37.

Note that the existing `alpha0_li_T` does **not** force `T` — it inherits the config,
which is `F` on arxiv — so it is not the control for this arm. `alpha0_li_T_only`
forces it. The existing arm's semantics are left unchanged so that runs already on
disk stay correctly labelled.

**Built-in placebo.** `node_feature` is called only at the GNN (M-) step, so this arm
can only affect `lm→gnn`. The `gnn→lm` direction is an internal negative control: if
it also shifts, something other than the feature channel changed and the result is
void.

**Decision rule, fixed before any `li=T` arxiv data is examined.** §11's criteria,
unchanged, applied to the `lm→gnn` out-of-bias bin of `published_li_T`:

- **Supported** — NCS < 0, two-sided exact McNemar p < 0.01, sign stable across ≥ 3
  seeds, the gap against `published` negative with stable sign, and `gnn→lm`
  unchanged.
- **Weakly supported** — NCS ≥ 0 but the gap against `published` is negative with
  stable sign, and `gnn→lm` unchanged.
- **Not supported** — otherwise, including any case where the `gnn→lm` placebo moves.

If supported, `alpha0_li_T_only` then attributes the effect: reproducing it with α=β=0
shows the feature channel suffices on its own; failing to reproduce it shows the two
channels interact.

**Prediction fixed in advance: the `lm→gnn` out-of-bias NCS moves from `published`'s
+0.0016 to between −0.008 and 0.000 — a gap of −0.002 to −0.010.** The WebKB lead's
−0.025 is not the prediction; effect magnitudes on arxiv run an order of magnitude
smaller than on the 19–26-node sets, and the lead's magnitude is inflated by the same
quantisation A4 records. At the observed per-seed NCS sd of ~0.001–0.004 a gap of
−0.005 is resolvable at three seeds.

**Scope of the existing gates, recorded because it is not obvious from their names.**
`probe.gating.apply_gate` rewrites `self.pl_nodes` only, and `pl_nodes` feeds
`get_inf_aug_train_ids` — that is, which nodes contribute the pseudo-label CE term.
The feature channel reads `self.ndata['pseudo_labels']` directly through `y_hat`,
which never consults `pl_nodes`. **Every gate built for A14/A16/A18 — `oracle`,
`oracle_random`, `conf_gate*`, `sig_gate*` — therefore filters the loss channel
alone.** No result already on disk is affected, because every dataset those arms ran
on ships `gnn_label_input=F` and so has no feature channel at all. But it means a gate
combined with `li=T` would be **silently incomplete**: a node removed from `pl_nodes`
still carries the teacher's wrong label in its input features. No arm currently
combines the two, and none should be added without closing this gap first.

**Contingent follow-up, registered now and deliberately not implemented.** If
`published_li_T` returns a negative out-of-bias NCS, the matching remedy is to gate
the *feature* channel — mask `y_hat` to zero (or 1/C) for nodes an exogenous signal
distrusts, applied identically at training and inference, with node selection reusing
the cached `_signals/` arrays. This is registered here so that the idea is timestamped
before the result is known; it is **not** wired, because it requires editing `y_hat`,
which is on the hot path of every GNN step, and it should not be touched until there
is a demonstrated effect to remedy.

Its predicted advantage over the A16/A18 loss gates is structural rather than
empirical, and is stated in advance: every loss gate paid a shrinkage tax —
`oracle_random` shows that dropping ~24% of pseudo-labels at random costs 0.20pp (GNN)
and 1.00pp (LM), which is why `conf_gate80` sits at +0.73pp against random yet −0.28pp
against `published`. A feature mask removes no node from training and no gradient from
the objective; it removes only the label hint. There is therefore no shrinkage tax to
overcome. If a feature gate also fails, that absence of a tax removes the last
available explanation for the A16/A18 nulls.

**Not §11 controls.** Both arms carry a `gnn_label_input` value that `published` does
not, so `report.py` — which matches `published` exactly and treats only the `alpha0_*`
arms as controls — leaves the existing verdicts untouched.

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

## 15. Transferability: what these findings say about other frameworks

§13's results are about GLEM. Their value depends on whether the *mechanism* transfers,
and the mechanism makes a falsifiable prediction about any framework that distils
between a language model and a GNN.

**The principle, stated as generally as the evidence supports.**

> Pseudo-label harm is a function of **exposure**, not of an identifiable node
> population. Exposure has two axes: *which channel* delivers the teacher's label — a
> loss term scaled by a weight, or an input feature scaled by nothing — and *how heavily*
> the student is made to attend to it. A framework is safe on an axis exactly to the
> extent it discounts a teacher that is weaker than its student on that axis.

GLEM is protected on the loss axis and unprotected on the feature axis, which is why
§13.1 is null and §13.3 is not. §13.2 identifies the protection: α = 0.50–0.80 in the
direction where the teacher is usually stronger, β = 0.05 where it is always weaker.

**This predicts where harm should appear elsewhere,** and the prediction is testable
without re-running anything: a framework that commits to one teacher on contested nodes,
with no per-node or per-direction discounting, should show the harm §4 predicted and
§13.1 failed to find.

**GNN-as-Judge is such a framework.** It partitions unlabeled nodes by LLM/GNN agreement
and, on the disagreement set, applies ORPO preference tuning with the GNN's prediction
preferred and the LLM's dispreferred — uniformly. There is no β.

Measured on GLEM's own teacher snapshots as a proxy (arxiv, `published`, 3 seeds; **not**
on GNN-as-Judge itself), splitting its disagreement rule along label-free GLANCE soft
homophily:

| disagreement subset | share | GNN right | LM right | value of preferring the GNN |
|---|---|---|---|---|
| high soft homophily | 13% | 0.661–0.691 | 0.197–0.248 | **+0.41 – +0.49** |
| low soft homophily | **87%** | 0.418–0.435 | 0.316–0.332 | **+0.09 – +0.12** |

The rule is worth four to five times more on the minority of nodes where the GNN is
inside its inductive bias than on the majority where it is not — and on that majority
the GNN it defers to is wrong ~58% of the time. Under a gold-label homophily split the
separation is ~30× rather than ~4.5×, but that signal is not deployable and is reported
only as a ceiling.

Their Theorem 2 — that the agreement set is strictly more accurate than either model
alone — holds directionally here (0.796 against 0.767 and 0.755) but is narrower than it
appears: the agreement set is **91% of nodes**, so it selects very little, and **20.4% of
it is both models wrong with the same label**. §13.2 measured 88% of the GNN's errors to
be the LM's errors as well; correlated errors are precisely the regime in which
agreement-based selection buys least.

**Three limits on the above, none of which the numbers can hide.**

1. The models are GLEM's DeBERTa and RevGAT, not GNN-as-Judge's LLM and GNN. The
   *structural* argument transfers; the magnitudes do not.
2. GLEM here is ~54% labelled; GNN-as-Judge is few-shot. §8 predicted harm is strongest
   where the gold CE term is too weak to anchor the student, so few-shot should if
   anything *increase* the exposure — but that direction is a prediction, not a result.
3. This analysis is post-hoc. The gate it motivates is preregistered separately, with a
   size-matched random control, in **A25**, and is run against that method's own code.

**What would falsify the principle.** If A25's gate fails against its random control,
then harm is absent even where no protective weighting exists, §13.2's attribution of
the §13.1 null to GLEM's asymmetric α/β is wrong, and §13.2 must be rewritten. The
principle in this section is stated so that it can lose.

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
