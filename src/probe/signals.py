"""Per-node signals: the two binning axes, plus the label-free secondary axis.

Both primary signals are computable **without the model being evaluated**, so
nothing reported from them is circular (EXPERIMENT.md section 5):

* ``local_homophily`` -- fraction of a node's neighbours sharing its label. The
  teacher-side axis for the **E-step**, where the GNN teaches: low homophily is
  where a neighbourhood-aggregating teacher is outside its inductive bias. NaN on
  isolated nodes, which are excluded everywhere.
* ``knn_ambiguity`` -- normalized entropy of the true labels of a node's k=15
  nearest neighbours in SBERT space, drawn only from labelled training nodes. The
  teacher-side axis for the **M-step**, where the LM teaches.
* ``soft_homophily`` -- GLANCE's label-free estimate ``h_v = p_v . mean p_u``,
  reported as a secondary axis because true local homophily needs neighbour labels
  and so is unusable at inference time.

Embeddings come from ``sentence-transformers/all-MiniLM-L6-v2`` computed here
(mean pooling over the attention mask, then L2 normalization -- the pooling this
model's sentence-transformers config specifies), rather than from the
``sbert_x.pt`` tensors some datasets ship. Those are 384-dim and so are very
probably the same model, but "very probably" is not good enough for a signal that
is half the experiment, and computing all datasets the same way makes them
comparable by construction.

Ambiguity is cached per ``(dataset, regime, seed)`` because the few-shot regimes
shrink the labelled pool it draws neighbours from; homophily and the embeddings
are cached per dataset.
"""
import numpy as np
import torch

SBERT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
KNN_K = 15


# ───────────────────────────────────────────── graph / text access


def load_tag_data(dataset):
    """The TAGDataset ``Data`` object for a GLEM dataset string.

    ``dataset`` is GLEM's name (``arxiv_TA``, ``cornell_TAG``, ...). The leading
    token before ``_`` is GLEM's DATA_INFO key, which is NOT always the TAGDataset
    name: a re-split variant such as ``citeseer60`` carries ``tag_name='citeseer'``
    plus its own ``resplit``, and asking TAGDataset for a dataset called
    ``citeseer60`` would simply fail. The resolution mirrors
    ``utils.data.preprocess_tag._load_tag_data`` -- including that the split seed is
    the re-split's own and never the run seed, so the split does not move between
    seeds while pretrain checkpoints that carry no seed are being reused.

    Returns the Data object, which carries ``edge_index``, ``y``, ``raw_texts`` and
    the ``{train,val,test}_mask`` GLEM itself splits on (``preprocess_tag`` derives
    its ``split_idx`` from exactly these masks).
    """
    import sys
    from omegaconf import OmegaConf
    from utils.settings import DATA_PATH, PROJ_DIR, get_d_info
    if PROJ_DIR not in sys.path:
        sys.path.insert(0, PROJ_DIR)
    from tag_data import TAGDataset

    d_info = get_d_info(dataset)
    name = d_info.get('tag_name', dataset.split('_')[0])
    rs = d_info.get('resplit', None)
    resplit_cfg = ({'enabled': False} if rs is None else
                   {'enabled': True, 'train_ratio': float(rs['train']),
                    'val_ratio': float(rs['val']), 'test_ratio': float(rs['test'])})
    root = f'{DATA_PATH}tag'
    cfg = OmegaConf.create({'seed': 0 if rs is None else int(rs['split_seed']),
                            'dirs': {'local_dir': root, 'cache_dir': root},
                            'data': {'resplit': resplit_cfg}})
    return TAGDataset(cfg, name=name).data


def load_tag_graph(dataset):
    """``(edge_index, y, raw_texts)`` for a GLEM dataset string, via TAGDataset."""
    d = load_tag_data(dataset)
    return d.edge_index, d.y.view(-1).numpy(), list(d.raw_texts)


def assert_aligned(y_tag, labels_run, dataset):
    """Guard that TAGDataset node order matches the order GLEM trained in.

    Signals are joined to logits by raw node id, so a reordering between the two
    loaders would silently mislabel every node. GLEM reads arxiv through its own
    OGB path and the TAG datasets through TAGDataset, so this is checked rather
    than assumed: identical label vectors over all N nodes is strong evidence of
    identical ordering, and cheap.
    """
    y_tag = np.asarray(y_tag).reshape(-1)
    labels_run = np.asarray(labels_run).reshape(-1)
    if y_tag.shape != labels_run.shape:
        raise ValueError(f'{dataset}: node count mismatch, TAGDataset {y_tag.shape} '
                         f'vs run splits {labels_run.shape}')
    bad = int((y_tag != labels_run).sum())
    if bad:
        raise ValueError(f'{dataset}: {bad}/{len(y_tag)} labels disagree between '
                         f'TAGDataset and the run\'s archived split -- node order '
                         f'differs, signals cannot be joined by node id')


# ───────────────────────────────────────────── signals


def local_homophily(edge_index, y, n_nodes):
    """Fraction of each node's neighbours sharing its label; NaN if isolated."""
    src, dst = np.asarray(edge_index[0]).reshape(-1), np.asarray(edge_index[1]).reshape(-1)
    same = (y[src] == y[dst]).astype(np.float64)
    num = np.zeros(n_nodes, dtype=np.float64)
    deg = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(num, src, same)
    np.add.at(deg, src, 1.0)
    out = np.full(n_nodes, np.nan)
    nz = deg > 0
    out[nz] = num[nz] / deg[nz]
    return out


def soft_local_homophily(edge_index, probs):
    """GLANCE's label-free estimate ``h_v = p_v . mean_{u in N(v)} p_u``; NaN if isolated."""
    src = torch.as_tensor(np.asarray(edge_index[0]).reshape(-1)).long()
    dst = torch.as_tensor(np.asarray(edge_index[1]).reshape(-1)).long()
    probs = torch.as_tensor(probs).float()
    n, c = probs.shape
    nbr_sum = torch.zeros(n, c).index_add_(0, src, probs[dst])
    deg = torch.zeros(n).index_add_(0, src, torch.ones(src.numel()))
    h = (probs * (nbr_sum / deg.clamp(min=1).unsqueeze(1))).sum(1)
    return torch.where(deg > 0, h, torch.tensor(float('nan'))).numpy()


def _resolve_sbert_dir():
    """Local snapshot dir for SBERT_MODEL in the HuggingFace hub cache.

    ``transformers`` 4.17 with ``huggingface_hub`` 0.4 predates the
    ``models--org--name/snapshots/<rev>`` cache layout and cannot resolve the model
    by name offline, so the directory is located directly.
    """
    import glob
    import os
    roots = [os.environ.get('HF_HOME', ''), os.environ.get('TRANSFORMERS_CACHE', ''),
             os.path.expanduser('~/.cache/huggingface')]
    for r in roots:
        if not r:
            continue
        for base in (r, os.path.join(r, 'hub')):
            pat = os.path.join(base, 'models--' + SBERT_MODEL.replace('/', '--'),
                               'snapshots', '*')
            for d in sorted(glob.glob(pat)):
                if os.path.exists(os.path.join(d, 'config.json')):
                    return d
    raise FileNotFoundError(
        f'{SBERT_MODEL} not found in the local HuggingFace cache; the ambiguity '
        f'signal needs it (EXPERIMENT.md section 5)')


def _load_safetensors(path):
    """Minimal safetensors reader: ``{name: tensor}``.

    The pinned ``transformers`` 4.17 cannot read ``model.safetensors`` and the
    ``safetensors`` package is not installed. The format is a little-endian uint64
    header length, a JSON header mapping each tensor to its dtype/shape/byte range,
    then one contiguous data buffer -- so reading it here avoids adding a
    dependency to a deliberately pinned Python 3.8 environment.
    """
    import json as _json
    import struct
    dtypes = {'F64': np.float64, 'F32': np.float32, 'F16': np.float16,
              'I64': np.int64, 'I32': np.int32, 'I16': np.int16, 'I8': np.int8,
              'U8': np.uint8, 'BOOL': np.bool_}
    with open(path, 'rb') as f:
        (hlen,) = struct.unpack('<Q', f.read(8))
        header = _json.loads(f.read(hlen).decode('utf-8'))
        blob = f.read()
    out = {}
    for name, meta in header.items():
        if name == '__metadata__':
            continue
        s, e = meta['data_offsets']
        arr = np.frombuffer(blob[s:e], dtype=dtypes[meta['dtype']])
        out[name] = torch.from_numpy(arr.reshape(meta['shape']).copy())
    return out


def _load_sbert_model(device):
    """all-MiniLM-L6-v2 encoder + tokenizer from the local cache."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    d = _resolve_sbert_dir()
    tok = AutoTokenizer.from_pretrained(d)
    model = AutoModel.from_config(AutoConfig.from_pretrained(d))
    sd = _load_safetensors(f'{d}/model.safetensors')
    # The sentence-transformers checkpoint stores bare encoder keys; BertModel here
    # is instantiated without a head, so pooler/head keys are simply absent.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if 'position_ids' not in k]
    if real_missing:
        raise RuntimeError(f'SBERT weights missing keys: {real_missing[:8]}')
    return tok, model.to(device).eval()


def sbert_embed(texts, device=None, batch=64, max_length=256):
    """L2-normalized mean-pooled all-MiniLM-L6-v2 embeddings, ``[N, 384]``."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tok, model = _load_sbert_model(device)
    out = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = [str(t) for t in texts[i:i + batch]]
            enc = tok(chunk, padding=True, truncation=True,
                      max_length=max_length, return_tensors='pt').to(device)
            h = model(**enc).last_hidden_state
            m = enc['attention_mask'].unsqueeze(-1).float()
            emb = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)  # mean pooling
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            out[i:i + batch] = emb.float().cpu().numpy()
    return out


def knn_ambiguity(emb, y, train_idx, num_classes, k=KNN_K, batch=2048):
    """Normalized kNN label-entropy in [0, 1]; neighbours from train nodes only.

    Self is excluded, which matters for train nodes: without it a train node's own
    label always appears among its own neighbours and depresses the entropy.
    """
    emb = np.ascontiguousarray(emb, dtype=np.float32)
    train_idx = np.asarray(train_idx).reshape(-1)
    train_emb = emb[train_idx]
    log_c = np.log(num_classes)
    out = np.empty(len(emb), dtype=np.float64)
    for start in range(0, len(emb), batch):
        end = min(start + batch, len(emb))
        sims = emb[start:end] @ train_emb.T
        kth = min(k, sims.shape[1] - 1)
        top = np.argpartition(-sims, kth=kth, axis=1)[:, :k + 1]
        for r, gi in enumerate(range(start, end)):
            cand = train_idx[top[r]]
            cand = cand[np.argsort(-sims[r, top[r]])]
            cand = cand[cand != gi][:k]
            p = np.bincount(y[cand], minlength=num_classes).astype(np.float64)
            p /= p.sum()
            out[gi] = -(p * np.log(p + 1e-12)).sum() / log_c
    return out


# ───────────────────────────────────────────── cached driver


def signals_for_run(run_dir, dataset, regime, seed, cache_dir):
    """Per-node signals for one run, using that run's archived split.

    Returns a dict with ``labels``, ``local_homophily``, ``knn_ambiguity``,
    ``degree``, ``edge_index``, ``num_classes`` and the three split id arrays.
    """
    from pathlib import Path
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ds_key = dataset.split('_')[0]

    sp = np.load(Path(run_dir) / 'splits.npz')
    labels, n_nodes = sp['labels'], int(sp['n_nodes'])
    num_classes = int(labels.max()) + 1

    edge_index, y_tag, texts = load_tag_graph(dataset)
    assert_aligned(y_tag, labels, dataset)
    edge_index = np.asarray(edge_index)

    hom_p = cache_dir / f'{ds_key}_homophily.npy'
    if hom_p.exists():
        hom = np.load(hom_p)
    else:
        hom = local_homophily(edge_index, labels, n_nodes)
        np.save(hom_p, hom)

    emb_p = cache_dir / f'{ds_key}_sbert.npy'
    if emb_p.exists():
        emb = np.load(emb_p)
    else:
        emb = sbert_embed(texts)
        np.save(emb_p, emb)

    amb_p = cache_dir / f'{ds_key}_{regime}_s{seed}_ambiguity.npy'
    if amb_p.exists():
        amb = np.load(amb_p)
    else:
        amb = knn_ambiguity(emb, labels, sp['train_x'], num_classes)
        np.save(amb_p, amb)

    deg = np.bincount(edge_index[0], minlength=n_nodes)
    return {'labels': labels, 'n_nodes': n_nodes, 'num_classes': num_classes,
            'local_homophily': hom, 'knn_ambiguity': amb, 'degree': deg,
            'edge_index': edge_index, 'train_x': sp['train_x'],
            'valid_x': sp['valid_x'], 'test_x': sp['test_x']}
