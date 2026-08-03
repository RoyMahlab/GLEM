"""Arm 3: subsample the train split to k labels per class, holding val/test fixed.

Applied at the single point where every consumer reads the split
(``utils.data.preprocess.load_graph_info``), *after* the pickle is loaded and
without writing back. Three reasons that specific seam:

* ``graph.info`` is cached per dataset with a ``processed.flag`` and is shared by
  every run of that dataset; rewriting it would poison the standard-split runs.
* Both ``GLEMConfig._exp_init`` (for ``em_info.n_train_nodes`` / ``n_pl_nodes``)
  and ``SeqGraph.init`` (for ``is_gold`` / ``pl_nodes`` / ``labeled_nodes``) read
  it independently. Patching one and not the other yields a run whose EM schedule
  disagrees with its own supervision.
* It keeps the un-probed path untouched: with no ``GLEM_PROBE_REGIME=fewshot<k>``
  this function returns its argument unchanged.

Nodes dropped from the train split become unlabeled, so they join ``pl_nodes``
(``SeqGraph.init`` derives it as ``~is_gold``) and are pseudo-labeled like any
other unlabeled node. That is the intended transductive few-shot setting. They are
nonetheless **excluded from the analysis population**, which stays val ∪ test in
every regime -- see EXPERIMENT.md section 14.
"""
import numpy as np

from probe import context


def apply_fewshot(g_info, cf):
    """Return ``g_info`` with the train split cut to k labels per class.

    No-op unless the regime is ``fewshot<k>``. The draw is seeded by ``cf.seed``,
    so each seed gets a different k-shot split and the across-seed variance
    required by EXPERIMENT.md section 10 includes split variance rather than only
    optimizer noise.
    """
    k = context.fewshot_k()
    if not context.enabled() or k is None:
        return g_info

    if hasattr(g_info, 'IDs'):
        # subset_ratio < 1 remaps every index through g_info.IDs; the split
        # surgery below assumes raw node ids. Fail rather than silently mislabel.
        raise NotImplementedError(
            'few-shot regimes are not supported on subset datasets (subset_ratio < 1)')

    labels = np.asarray(g_info.labels).reshape(-1)
    train_x = np.asarray(g_info.splits['train_x']).reshape(-1)
    rng = np.random.default_rng(int(cf.seed))

    picked, short = [], []
    for c in np.unique(labels[train_x]):
        pool = train_x[labels[train_x] == c]
        if len(pool) <= k:
            short.append((int(c), len(pool)))
            picked.append(pool)
        else:
            picked.append(rng.choice(pool, size=k, replace=False))
    new_train = np.sort(np.concatenate(picked)).astype(train_x.dtype)

    is_gold = np.zeros(g_info.n_nodes, dtype=bool)
    is_gold[new_train] = True
    g_info.splits['train_x'] = new_train
    g_info.is_gold = is_gold
    # val_test is deliberately untouched: val and test are held fixed across all
    # regimes so the analysis population is identical and regimes are comparable.

    msg = (f'[probe] few-shot k={k} seed={cf.seed}: train {len(train_x)} -> '
           f'{len(new_train)} nodes over {len(np.unique(labels[new_train]))} classes')
    if short:
        msg += f'; classes with fewer than k train nodes, all taken: {short}'
    print(msg)
    return g_info
