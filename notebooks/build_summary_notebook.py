"""Build notebooks/glem_project_summary.ipynb — the whole arc, idea to result."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip('\n')))
co = lambda s: C.append(nbf.v4.new_code_cell(s.strip('\n')))

md(r"""
# Does uniform pseudo-labeling harm an identifiable node population in GLEM?

**Project summary — the idea, what was built, what was run, and what came back.**

Companion notebooks: [`glem_harm_results.ipynb`](glem_harm_results.ipynb) (the
preregistered measurement in detail), [`glem_results.ipynb`](glem_results.ipynb)
(per-step exploration), [`glem_gate_selection.ipynb`](glem_gate_selection.ipynb)
(choosing a deployable gate). Protocol and full amendment log:
[`EXPERIMENT.md`](../EXPERIMENT.md).

---

## The idea

GLEM (Zhao et al., ICLR 2023) alternates an **E-step** — train the LM on the GNN's
pseudo-labels — with an **M-step** — train the GNN on the LM's embeddings and
pseudo-labels. Each step weights its pseudo-label term by a single global scalar:
α for the LM step, β for the GNN step. That encodes an assumption:

> *the value of the teacher's signal does not depend on the node.*

The hypothesis was that this is false in a **predictable, directional** way. Outside
its inductive bias a model is not merely uncertain but **confidently wrong** — the
GNN on **low-homophily** nodes, the LM on **semantically ambiguous** text. So a
model teaching from inside that region should *corrupt* student nodes that were
previously correct, and the net effect there should be **negative**.

Two exogenous signals make that testable without circularity: local homophily and
kNN semantic ambiguity, neither of which depends on the model being evaluated.
""")

md(r"""
## How the work actually unfolded

| phase | question | answer |
|---|---|---|
| **1. Instrument** | can before/after/teacher be recovered per step? | yes — but GLEM overwrites its own logits, so an archive had to be built first |
| **2. Harm measurement** | is NCS negative where the teacher is out of bias? | **no**, on 13 of 14 scorable cells |
| **3. Why not?** | what breaks the prediction? | the teacher stays *better than the student* even where it is weakest |
| **4. Mechanism** | does the student adopt the teacher's wrong labels? | **yes, 1.7–5.8×** — the mechanism is real, it just does not net out |
| **5. Ceiling** | would perfect gating pay? | **yes, +2.8 / +3.6pp** on arxiv |
| **6. Confidence gate** | does the field's standard fix capture it? | **no, ≈0** — and we can say precisely why |
| **7. Signal gate** | do exogenous signals reach what confidence cannot? | *running* |

The pivot at phase 3 is the substance of the project: the original prediction was
wrong, and being wrong turned out to be more informative than being right, because
the reason is measurable.
""")

co(r"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd()
while not (ROOT / 'src' / 'probe').is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
ANALYSIS = ROOT / 'temp' / 'probe_analysis'
PROBE = ROOT / 'temp' / 'probe_output'
pd.set_option('display.width', 220)

ncs = pd.read_csv(ANALYSIS / 'ncs_long.csv')
verdicts = pd.read_csv(ANALYSIS / 'verdicts.csv')

runs = sorted(PROBE.glob('*/standard/*/seed*/steps.jsonl'))
inv = pd.DataFrame([{'dataset': r.parts[-5], 'arm': r.parts[-3],
                     'seed': int(r.parts[-2][4:])} for r in runs])
print(f'{len(inv)} completed runs | {inv.dataset.nunique()} dataset variants '
      f'| {inv.arm.nunique()} arms')
print()
print('runs per arm:')
print(inv.groupby('arm').size().sort_values(ascending=False).to_string())
""")

md(r"""
## Phase 1 — the instrument, and the bug that made it necessary

GLEM's teacher and student communicate **only** through fp16 memmaps on disk: the
GNN writes its predictions, the LM reads them as `pseudo_label_file`, and vice
versa. That is convenient — every before/after/teacher triple is already on disk, so
no extra forward passes are needed.

But for `em_iter >= 0` **the paths carry no iteration index**, so each M-step
clobbers the previous GNN prediction and each E-step the previous LM one. Before/after
pairs are destroyed as a run proceeds. Nothing downstream was measurable until that
was fixed.

`src/probe/` archives every write under an iteration-keyed path, plus the teacher
snapshot each step actually consumed, the node ids that actually received a
pseudo-label (unreconstructable afterwards — the GNN resamples every epoch), and the
split. Every hook is a no-op unless `GLEM_PROBE_DIR` is set, so an unprobed run is
byte-identical to stock GLEM.

**Fidelity check first**, because a broken reproduction would void everything after it.
""")

co(r"""
fidelity = pd.DataFrame([
    {'metric': 'test accuracy', 'reproduced': 0.769973, 'paper': 0.7697, 'paper_sd': 0.0019},
    {'metric': 'val accuracy',  'reproduced': 0.775932, 'paper': 0.7749, 'paper_sd': 0.0017},
])
fidelity['delta'] = (fidelity.reproduced - fidelity.paper).round(5)
fidelity['within_paper_sd'] = fidelity.delta.abs() <= fidelity.paper_sd
print('GLEM + RevGAT on ogbn-arxiv, published hyperparameters, nothing tuned')
print(fidelity.to_string(index=False))
""")

md(r"""
## Phase 2 — the preregistered measurement

`EXPERIMENT.md` fixed every bin, threshold, axis and decision rule **before any
measurement existed** (commit `33a3601`). Eighteen amendments record every deviation
since, with dates and reasons, including the ones that hurt.

The rule (§11), per (dataset, direction):

- **Supported** — NCS significantly negative out-of-bias, positive elsewhere, sign
  stable across seeds, **and** absent from the α=β=0 control.
- **Weakly supported** — NCS positive everywhere but significantly *lower*
  out-of-bias, with teacher accuracy degrading across the axis.
- **Not supported** — flat, **or** the pattern reproduces in the control.
""")

co(r"""
show = verdicts[['dataset', 'direction', 'published_ncs_oob', 'published_ncs_in',
                 'published_ncs_gap', 'published_gap_sd', 'published_sign_stable',
                 'published_tacc_oob', 'published_tacc_in', 'verdict']].copy()
show['dataset'] = show.dataset.str.replace('_TAG', '', regex=False).str.replace('_TA', '', regex=False)
print(show.sort_values(['verdict', 'dataset']).round(4).to_string(index=False))
print()
print(verdicts.verdict.value_counts().to_string())
""")

md(r"""
### What the verdict table says

**One weakly-supported cell: arxiv `gnn->lm`.** NCS is positive in both bins
(+0.0170 in-bias, +0.0062 out-of-bias) but the gap is **−0.0108 ± 0.0017**,
sign-stable across three seeds, p = 6.2e−19, with teacher accuracy falling
**0.957 → 0.580** across the axis. The α=β=0 control cannot reproduce it: its
per-seed gaps are +0.0001 / +0.0103 / −0.0169, sd 0.0137 — eight times the published
spread, flipping sign twice.

**Everything else is null or untestable**, and the *reasons* matter more than the
counts:

- **A6 — degenerate bins.** cora, citeseer and pubmed have median local homophily
  **1.000** with ~65% of nodes tied there, so the median split leaves one bin empty.
  Not a null: a missing measurement.
- **A4 — no power.** WebKB's E-step sees 6–11 nodes per step under
  `lm_pl_ratio=0.1`. Fixed later by re-running those datasets on RevGAT (A13), which
  uses `lm_pl_ratio=1` and lifts the population to 71–101.
- **Sign instability.** Four cells had the *predicted* negative sign but flipped
  across seeds — §10 rejects those regardless of p-value.

The single positive result exists only because arxiv is the one dataset with both a
non-degenerate homophily axis **and** full pseudo-label coverage.
""")

md(r"""
## Phase 3 — why the prediction failed

Two post-hoc analyses, both labelled as such and excluded from the verdict.

**The headroom confound (A8).** NCS divides by bin size, but corruption is capped by
how many nodes the student had right to begin with — and student accuracy is 6–19
points *lower* out-of-bias on every dataset. Conditioning on corruptible nodes, the
published arm does show elevated corruption out-of-bias (1.26–5.83×) — **but the
α=β=0 control reproduces it at equal or larger magnitude**. That fragility belongs
to low-margin nodes under any retraining, not to the pseudo-label term.

**Absolute vs relative teacher weakness — the actual explanation.** The hypothesis
assumed "teacher outside its inductive bias" implies "teacher worse than the
student". It does not.
""")

co(r"""
t = ncs[(ncs.axis == 'teacher') & (ncs.bin_scheme == 'median')
        & (ncs.arm == 'published') & (~ncs.low_n_flag)]
OOB = {'gnn->lm': 'low', 'lm->gnn': 'high'}
e = (t.groupby(['direction', 'dataset', 'bin'])
     .agg(n=('n', 'sum'), teacher=('teacher_acc', 'mean'),
          student_before=('student_acc_before', 'mean')).reset_index())
e['role'] = np.where([b == OOB[d] for d, b in zip(e.direction, e['bin'])],
                     'OUT-OF-BIAS', 'in-bias')
e['teacher_edge'] = (e.teacher - e.student_before).round(4)
e['dataset'] = e.dataset.str.replace('_TAG', '', regex=False).str.replace('_TA', '', regex=False)
print('teacher_edge = teacher accuracy - student accuracy BEFORE the step')
print('negative => the teacher really is worse than the student on those nodes')
print(e[['direction', 'dataset', 'role', 'n', 'teacher', 'student_before',
         'teacher_edge']].sort_values(['direction', 'dataset', 'role'])
      .round(3).to_string(index=False))
""")

md(r"""
On arxiv's E-step the GNN teacher is **58.1%** accurate out-of-bias — but the LM
student was **57.2%**. Bad in absolute terms, still a net upgrade. Positive NCS there
is the *correct* prediction, and NCS shrinking from +0.0170 to +0.0062 is exactly
that shrinking margin made visible.

Where the premise *does* hold — the LM teaching the GNN, where `teacher_edge` is
negative on 14 of 15 cells and reaches −0.20 — the hypothesis gets a fair test and
still comes back null.

**So the signal predicts teacher error extremely well and student harm poorly**,
because teacher error ≠ teacher being worse than the student. That is the project's
central finding.
""")

md(r"""
## Phase 4 — the mechanism is real, NCS was the wrong instrument

NCS nets corrections against corruptions and never looks at *which* label the student
moved to. Keying on the teacher's actual prediction instead: restrict to nodes the
student had **right** and the teacher had **wrong**, then ask how often the student
landed on the teacher's *specific* label. The α=β=0 control gives the null, since
there the teacher's label has no causal path.

| direction | dataset | published | control | **amplification** |
|---|---|---|---|---|
| gnn→lm | arxiv | 0.470 | 0.124 | **3.79×** |
| lm→gnn | arxiv | 0.287 | 0.154 | 1.87× |
| lm→gnn | citeseer | 0.148 | 0.026 | **5.76×** |
| lm→gnn | cora | 0.152 | 0.049 | 3.08× |
| lm→gnn | pubmed | 0.433 | 0.251 | 1.72× |

*(P(node breaks AND lands on the teacher's exact label), chance ≈ 1/(C−1).)*

**The student really does adopt the teacher's wrong labels, at 1.7–5.8× the
retraining baseline.** The causal story was right; NCS simply could not see it,
because on arxiv's low-homophily bin 4,000 corruptions sit alongside 4,420
corrections.

But the amplification is **flat across the axis** — 3.79× in-bias vs 3.33×
out-of-bias. The student follows the teacher *everywhere*; harm concentrates
out-of-bias only because the teacher is **wrong more often** there (4,039 at-risk
nodes vs 353).
""")

md(r"""
## Phase 5 — is there headroom? The oracle

If the null means "uniform α is fine", that should show up as *no gain* from perfect
gating. So: restrict the pseudo-label set to nodes the teacher gets **right** (uses
gold labels — an upper bound, not a method), against a **size-matched random**
control that drops the same number arbitrarily. The random arm is not optional: the
oracle also shrinks the set, and shrinking alone changes the result.
""")

co(r"""
def final_accuracy(probe=PROBE):
    rows = []
    for run in sorted(probe.glob('*/standard/*/seed*')):
        sp = run / 'splits.npz'
        if not sp.exists():
            continue
        z = np.load(sp); y = z['labels']
        for kind in ('gnn', 'lm'):
            f = run / 'logits' / ('iter1_%s.npy' % kind)
            if not f.exists():
                continue
            pred = np.load(f).astype(np.float32).argmax(1)
            rows.append({'dataset': run.parts[-4], 'arm': run.parts[-2],
                         'seed': int(run.name[4:]), 'model': kind,
                         'test_acc': float((pred[z['test_x']] == y[z['test_x']]).mean())})
    return pd.DataFrame(rows)

acc = final_accuracy()

# --- the oracle across EVERY dataset, paired by seed ---
w = acc.pivot_table(index=['dataset', 'model', 'seed'], columns='arm', values='test_acc')
w = w.dropna(subset=['oracle', 'oracle_random'])
w['gain'] = w['oracle'] - w['oracle_random']
g = (w.groupby(['dataset', 'model'])
       .agg(seeds=('gain', 'size'), published=('published', 'mean'),
            random=('oracle_random', 'mean'), oracle=('oracle', 'mean'),
            gain=('gain', 'mean'), gain_sd=('gain', 'std')).reset_index())
# t on the paired per-seed differences. Meaningless where gain_sd is exactly 0,
# which happens on the 19-26 node WebKB test sets through sheer quantisation.
g['t'] = np.where(g.gain_sd > 1e-9, g.gain / (g.gain_sd / np.sqrt(g.seeds)), np.nan)
g['dataset'] = g.dataset.str.replace('_TAG', '', regex=False).str.replace('_TA', '', regex=False)
print('ORACLE minus size-matched random control, paired by seed, ALL datasets:')
print(g.round(4).to_string(index=False))
print()
print('cells with a positive gain: %d / %d' % ((g.gain > 0).sum(), len(g)))
print('cells with t > 2         : %d' % (g.t > 2).sum())
print()

a = acc[acc.dataset == 'arxiv_TA']
order = ['published', 'oracle_random', 'conf_gate60', 'conf_gate80', 'conf_gate90',
         'sig_gate80_gnn', 'sig_gate80_lm', 'sig_gate80', 'sig_gate90', 'oracle']
piv = a.pivot_table(index='model', columns='arm', values='test_acc', aggfunc=['mean', 'count'])
have = [c for c in order if ('mean', c) in piv.columns]
out = pd.concat([piv['mean'][have], piv['count'][have].add_suffix('_n')], axis=1)
print('arxiv test accuracy by arm (mean over seeds; *_n = seeds available)')
print(out.round(4).to_string())
""")

md(r"""
**Perfect gating pays, and not only on arxiv — 17 of 18 cells show a positive
gain.** Where the test set is large enough to resolve it:

| dataset | model | random | oracle | gain | t |
|---|---|---|---|---|---|
| arxiv | LM | 0.7468 | **0.7902** | **+4.35pp** | **98.3** |
| arxiv | GNN | 0.7657 | **0.7981** | **+3.24pp** | **25.4** |
| pubmed | GNN | 0.9517 | 0.9607 | +0.90pp | **6.65** |
| pubmed | LM | 0.9497 | 0.9597 | +1.00pp | **4.39** |
| cora | GNN | 0.8801 | 0.9188 | +3.87pp | **3.44** |

The remaining cells are positive but underpowered: cora LM (t = 1.74), citeseer LM
(2.00), and the four WebKB sets, whose 19–26 test nodes cannot resolve anything.
`wisconsin gnn` reports an absurd t because its `gain_sd` is *exactly* zero — a
quantisation artifact of 26 test nodes, not precision, and it is masked in the table
above.

arxiv is nonetheless the cleanest cell by a wide margin: signal-to-noise of 18:1 and
72:1, against pubmed's 3.8:1 and cora's 2:1. Its tiny seed variance, not its effect
size, is what makes it the dataset where a *realisable* gate could be measured.

Note `oracle_random` sits *below* published on arxiv — dropping 23% of pseudo-labels
at random **costs** 0.42pp — so a real gate must clear that before showing any gain.

An earlier version of this analysis dismissed the oracle result as transductive label
leakage. **That was wrong and is withdrawn (A15):** training a student on
pseudo-labels for evaluation nodes is GLEM's method, not contamination, and all arms
are trained and evaluated identically.
""")

md(r"""
## Phase 6 — the confidence gate fails, and the reason is the finding

GLEM already ships a per-node confidence gate: `pl_filter` keeps the top-k by
max-softmax. The arxiv config leaves it **unset**. Selection precision predicted this
would recover ~36% of the oracle's headroom, so A16 preregistered **+0.8pp**.

**Actual: ≈0.** conf_gate90 gives −0.09pp (GNN) and +0.14pp (LM), and the result is
monotone in keep-rate — the more you drop, the worse you do. Confidence selection buys
only +0.13pp (GNN) / +0.73pp (LM) over *random* selection at a comparable keep-rate.

The prediction was wrong for a diagnosable reason. Confidence discriminates teacher
error well overall (AUROC ≈ 0.78) but **collapses to 0.628 inside the out-of-bias
region** where it would need to work. At the oracle keep-rate:

| gate | all errors excluded | **confidently-wrong excluded** |
|---|---|---|
| confidence, gnn→lm | 0.516 | **0.000** |
| homophily, gnn→lm | 0.485 | **0.116** |
| confidence, lm→gnn | 0.507 | **0.000** |
| ambiguity, lm→gnn | 0.423 | **0.177** |

**Confidence strips the harmless errors and leaves the damaging ones — it is
structurally blind to exactly the population the original hypothesis identified.**
That is a clean negative result about the field's standard mechanism, and it
vindicates the hypothesis's *premise* while refuting its prediction.
""")

md(r"""
## Phase 7 — the exogenous-signal gate (running)

Three arms (A18), decomposed by teacher so the effect can be attributed:

| arm | gates E-step (GNN teaches) | gates M-step (LM teaches) |
|---|---|---|
| `sig_gate80_gnn` | yes — top 80% by GLANCE soft homophily | no |
| `sig_gate80_lm` | no | yes — top 80% by inverted kNN ambiguity |
| `sig_gate80` | yes | yes |

Keep-rates deliberately match `conf_gate80/90`, so the difference isolates the
**signal** with shrinkage held constant.

**Preflight on arxiv** — the gate does raise kept-set teacher accuracy: 0.767 → 0.833
(GNN teaching) and 0.755 → 0.805 (LM teaching) at 80% keep.

**Prediction recorded before running: +0.3 to +0.5pp** over published — the oracle's
+2.82pp scaled by the 12–18% of the confidently-wrong population these signals reach.
Deliberately more modest than A16's failed +0.8pp, and grounded in the mechanism that
explains that failure rather than in selection precision, which is now known not to
transfer.

**Scope limit (found by smoke test, not by luck):** on cornell the GLANCE gate
*lowers* kept-set accuracy (−0.050) because GLANCE's sign inverts on heterophilous
graphs. The arms are registered for arxiv only.
""")

md(r"""
## What is established, and what is not

**Established:**

1. **GLEM reproduces faithfully** — 0.76997 test vs the paper's 0.7697.
2. **Uniform α does not produce net harm** where the teacher is out of its bias — 13
   of 14 scorable cells null, across 8 datasets and 3 seeds.
3. **The harm mechanism is nonetheless real** — students adopt the teacher's specific
   wrong label at 1.7–5.8× the retraining baseline.
4. **It does not net out because the teacher stays better than the student**, even at
   58% accuracy against a 57% student.
5. **Perfect gating pays across datasets** — positive on 17 of 18 cells, and
   significant on arxiv (+3.2 / +4.3pp), pubmed (+0.9 / +1.0pp) and cora (+3.9pp).
   Headroom is not an arxiv artifact.
6. **Confidence-based gating captures almost none of it**, because it is blind to the
   confidently-wrong population — a negative result about the standard fix.
7. **arxiv's E-step is weakly supported** — the one cell where the axis, the power and
   the control all line up.

**Not established:**

- **Whether exogenous-signal gating pays.** Running; predicted +0.3–0.5pp.
- **The low-label regime.** Arm 3 (few-shot 3/5/10 labels per class) was withdrawn
  (A5). §8 predicted harm would be strongest where the gold-label CE term is too weak
  to anchor the student, so every null here is compatible with "harm exists but is
  masked at ~54% labelled". **This is the largest untested lever.**
- **Heterophilous graphs at scale.** WebKB is 187–265 nodes; arxiv is homophilous.
  Nothing here tests a large heterophilous TAG.
- **Attribution is imperfect.** The α=β=0 control is a clean single-variable ablation
  only for the E-step; on the M-step the GNN still consumes LM embeddings (A9/A10).
""")

md(r"""
## Appendix — provenance

| artefact | what it is |
|---|---|
| `EXPERIMENT.md` | preregistration (`33a3601`, before any data) + amendments A1–A18 |
| `src/probe/` | the instrument: archive, signals, gating, NCS/McNemar, verdicts |
| `temp/probe_output/` | per-run archive — logits, teacher snapshots, pl node ids, splits |
| `temp/probe_analysis/` | `ncs_long.csv`, `quadrants.csv`, `verdicts.csv`, figures |
| `logs/probe/manifest.tsv` | every run's exit status and wall time |
| `scripts/` | `probe_run.sh`, `probe_sweep.sh`, `parallel_seed_sweep.sh`, `perfect_teacher_sweep.sh` |

The amendment log is the honest record. A15 withdraws a wrong claim outright; A17
discloses that a stricter reading of §11 would erase the study's only positive
verdict; A4, A6 and A9 record limitations that cost coverage. They are worth reading
alongside the results.
""")

nb['cells'] = C
nb.metadata.update({
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.8.12'},
})
out = Path(__file__).resolve().parent / 'glem_project_summary.ipynb'
nbf.write(nb, str(out))
print(f'wrote {out} — {len(C)} cells ({sum(c.cell_type == "code" for c in C)} code)')
