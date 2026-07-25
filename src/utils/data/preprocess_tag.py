"""Bridge that lets GLEM consume the non-OGB text-attributed graphs exposed by
``tag_data.TAGDataset`` (cora, citeseer, pubmed, wikics, book*, sportsfit, WebKB).

GLEM's native pipeline only knows the OGB datasets. Datasets whose ``DATA_INFO``
entry carries ``loader='tag'`` are routed here from ``preprocess.py``:

* ``load_tag_graph_structure`` mirrors ``load_ogb_graph_structure_only`` -> a DGL
  graph + numpy labels + a ``{train,valid,test}`` split-index dict.
* ``tokenize_tag_dataset`` mirrors ``_tokenize_ogb_arxiv_datasets`` -> writes the
  ``input_ids/attention_mask/token_type_ids`` memmaps ``SeqGraph`` expects.

Node ordering is ``0..N-1`` in both GLEM and ``TAGDataset``, so labels, tokens and
the LM embeddings the GNN later uses as features all line up.
"""
import sys

import dgl
import numpy as np
import torch as th
from tqdm import tqdm
from transformers import AutoTokenizer

import utils.function as uf
from utils.settings import DATA_PATH, PROJ_DIR

# Cache the loaded PyG ``Data`` per dataset so graph-structure and tokenization
# passes within one process don't reload/redownload it.
_TAG_CACHE = {}


def _load_tag_data(cf):
    """Return the ``TAGDataset`` PyG ``Data`` object for ``cf``'s dataset."""
    d = cf.data
    name = d.tag_name
    if name not in _TAG_CACHE:
        if PROJ_DIR not in sys.path:
            sys.path.insert(0, PROJ_DIR)
        from omegaconf import OmegaConf
        from tag_data import TAGDataset

        tag_root = f"{DATA_PATH}tag"
        tag_cfg = OmegaConf.create(
            {
                "seed": int(cf.seed),
                "dirs": {"local_dir": tag_root, "cache_dir": tag_root},
                "data": {"resplit": {"enabled": False}},
            }
        )
        _TAG_CACHE[name] = TAGDataset(tag_cfg, name=name).data
    return _TAG_CACHE[name]


def load_tag_graph_structure(cf):
    """DGL graph + labels + split index for a ``loader='tag'`` dataset.

    Signature matches ``load_ogb_graph_structure_only`` so every caller
    (``load_graph_info`` and the GNN trainers) works unchanged.
    """
    data = _load_tag_data(cf)
    n_nodes = data.x.shape[0]
    src, dst = data.edge_index
    g = dgl.graph((src.long(), dst.long()), num_nodes=n_nodes)
    labels = data.y.view(-1).numpy()
    split_idx = {
        "train": th.nonzero(data.train_mask, as_tuple=False).view(-1),
        "valid": th.nonzero(data.val_mask, as_tuple=False).view(-1),
        "test": th.nonzero(data.test_mask, as_tuple=False).view(-1),
    }
    return g, labels, split_idx


def tokenize_tag_dataset(d, labels, chunk_size=50000):
    """Tokenize ``TAGDataset.raw_texts`` into the memmaps ``SeqGraph`` reads.

    Mirrors ``_tokenize_ogb_arxiv_datasets``: the per-node text is already a full
    string, so we only truncate to ``cut_off`` words before the tokenizer's own
    ``max_length`` truncation.
    """
    data = _load_tag_data(d.cf)
    raw_texts = data.raw_texts
    n_nodes, max_length = d.info["input_ids"].shape
    assert len(raw_texts) == n_nodes, (len(raw_texts), n_nodes)

    tokenizer = AutoTokenizer.from_pretrained(d.hf_model)
    token_info = {
        k: np.memmap(d.info[k].path, dtype=d.info[k].type, mode="w+", shape=d.info[k].shape)
        for k in d.token_keys
    }
    for start in tqdm(range(0, n_nodes, chunk_size), desc="Tokenizing TAG"):
        end = min(start + chunk_size, n_nodes)
        texts = [" ".join(str(t).split(" ")[: d.cut_off]) for t in raw_texts[start:end]]
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_token_type_ids=True,
        ).data
        idx = np.arange(start, end)
        for k in d.token_keys:
            token_info[k][idx] = np.array(tokenized[k], dtype=d.info[k].type)
    uf.pickle_save("processed", d._processed_flag["token"])
    return
