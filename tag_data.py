from __future__ import annotations

from torch_geometric.data import Dataset, Data
from torch_geometric.utils import to_undirected

from src.utils.settings import DATA_PATH
try:
    from huggingface_hub.errors import RemoteEntryNotFoundError
except ImportError:  # older huggingface_hub (e.g. the pinned GLEM `ct` env)
    # Only the HuggingFace-hosted datasets rely on this error type; the OGB
    # datasets (ogbn-arxiv / ogbn-products) never touch the Hub, so a fallback
    # keeps the module importable on envs with an old huggingface_hub.
    class RemoteEntryNotFoundError(Exception):
        pass
import ast
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
import tqdm


class TAGDataset(Dataset):
    REPO = "Graph-COM/Text-Attributed-Graphs"
    # WebKB datasets ship their graph under a capitalized name (e.g. `Cornell.pt`),
    # but those `.pt`/`.txt` files are stored in a non-loadable upstream format.
    # We reconstruct them from the self-consistent `<Name>.csv` (structure, labels,
    # text) plus the sBERT embeddings as node features.
    WEBKB_DATASETS = ["cornell", "texas", "washington", "wisconsin"]
    AVAILABLE_DATASETS = [
            "bookchild",
            "bookhis",
            "citeseer",
            "cora",
            "pubmed",
            "sportsfit",
            "wikics",
        ]
    # OGB node-classification TAGs. These are NOT hosted in the HuggingFace REPO
    # above; graph/features/labels/splits come from the OGB package and the raw
    # node text is fetched from the datasets' original text sources (see
    # `_load_ogb_arxiv_texts` / `_load_ogb_products_texts`).
    OGB_DATASETS = ["ogbn-arxiv", "ogbn-products"]
    OGB_ALIASES = {"arxiv": "ogbn-arxiv", "products": "ogbn-products"}
    OGB_ARXIV_TEXT_URL = "https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz"
    OGB_PRODUCTS_TEXT_URL = "https://drive.google.com/u/0/uc?id=1gsabsx8KR2N9jJz16jTcA0QASXsNuKnN&export=download"

    def __init__(self, cfg: OmegaConf, name: str):
        super().__init__()
        name = self.OGB_ALIASES.get(name, name)
        if (
            name not in self.AVAILABLE_DATASETS
            and name not in self.WEBKB_DATASETS
            and name not in self.OGB_DATASETS
        ):
            raise ValueError(
                f"Dataset '{name}' not found. Available datasets: "
                f"{self.AVAILABLE_DATASETS + self.WEBKB_DATASETS + self.OGB_DATASETS}"
            )
        self.name = name
        self.cfg = cfg
        self._cache: dict = {}  # Cache for loaded tensors to avoid re-downloading
        self.data = self._load_tag()

    def _load_tag(self) -> Data:
        """Load the processed data tensor from HuggingFace Hub."""
        if self.name in self.OGB_DATASETS:
            return self._load_ogb()
        if self.name in self.WEBKB_DATASETS:
            return self._load_webkb()
        process_data: Data = self.load_processed_data
        raw_texts = self.load_raw_texts
        train_mask = self._select_split(process_data.train_mask)
        val_mask = self._select_split(process_data.val_mask)
        test_mask = self._select_split(process_data.test_mask)
        train_mask, val_mask, test_mask = self._maybe_resplit(
            train_mask, val_mask, test_mask
        )
        # Some datasets (bookchild/bookhis/sportsfit) ship no node features in
        # processed_data; fall back to the SBERT embeddings, matching WebKB.
        x = process_data.x if process_data.x is not None else self.load_sbert_x
        return Data(
            x=x,
            edge_index=process_data.edge_index,
            y=process_data.y,
            raw_texts=raw_texts,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )

    @staticmethod
    def _select_split(mask: torch.Tensor, col: int = 0) -> torch.Tensor:
        """Collapse a multi-split mask to a single 1-D boolean split.

        Some datasets (e.g. WikiCS) ship masks of shape ``[num_nodes, num_splits]``;
        downstream code (dataloaders, analysis) assumes a single 1-D boolean mask,
        so we take one split column.
        """
        if mask.dim() > 1:
            mask = mask[:, col]
        return mask.bool()

    def _maybe_resplit(self, train_mask, val_mask, test_mask):
        """Replace the shipped masks with a consistent deterministic split.

        When ``cfg.data.resplit.enabled`` is set, *every* dataset is re-split with
        the same seed-controlled ratios (configs/data.yaml), so splits are uniform
        and reproducible across datasets and across the GNN / LLM stages. Absent or
        disabled -> keep the original shipped masks.
        """
        rs = OmegaConf.select(self.cfg, "data.resplit", default=None)
        if not rs or not rs.get("enabled", False):
            return train_mask, val_mask, test_mask
        new = self._random_split(train_mask.shape[0])
        print(
            f"[TAGDataset] '{self.name}': deterministic re-split to "
            f"{int(new[0].sum())}/{int(new[1].sum())}/{int(new[2].sum())} "
            f"train/val/test (seed={int(self.cfg.seed)})."
        )
        return new

    def _load_webkb(self) -> Data:
        """Reconstruct a WebKB dataset from its HuggingFace CSV.

        The CSV is self-consistent: one row per node (ordered by ``node_id``)
        carrying the label, raw text and the neighbor adjacency list. Node
        features are taken from the sBERT embeddings, which share the same node
        ordering. Since the CSV ships no splits, we build a deterministic,
        seed-controlled train/val/test split.
        """
        df = self._load_csv(f"{self.name.capitalize()}.csv")
        df = df.sort_values("node_id").reset_index(drop=True)
        num_nodes = len(df)

        y = torch.tensor(df["label"].to_numpy(), dtype=torch.long)
        raw_texts = df["raw_text"].astype(str).tolist()

        edges: list[tuple[int, int]] = []
        for src, neighbors in zip(df["node_id"].tolist(), df["neighbor_ids"].tolist()):
            if not isinstance(neighbors, str):
                continue
            for dst in ast.literal_eval(neighbors):
                edges.append((int(src), int(dst)))
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        x = self.load_sbert_x
        train_mask, val_mask, test_mask = self._random_split(num_nodes)

        return Data(
            x=x,
            edge_index=edge_index,
            y=y,
            raw_texts=raw_texts,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )

    def _load_ogb(self) -> Data:
        """Load an OGB node-classification TAG (ogbn-arxiv / ogbn-products).

        Graph structure, node features, labels and the official train/val/test
        split come from the OGB package; raw node text is fetched from each
        dataset's original text source. The returned ``Data`` matches the same
        field contract as the HuggingFace-hosted datasets (``x``, ``edge_index``,
        ``y``, ``raw_texts``, ``{train,val,test}_mask``). If ``cfg.data.resplit``
        is enabled the official split is replaced by the deterministic one, just
        like every other dataset here.
        """
        dataset = self._build_ogb_dataset()
        g = dataset[0]
        split_idx = dataset.get_idx_split()
        num_nodes = int(g.num_nodes)

        y = g.y.view(-1).long()
        edge_index = g.edge_index
        if self.name == "ogbn-arxiv":
            # arxiv ships a directed citation graph; the leaderboard convention
            # (and GLEM) symmetrize it.
            edge_index = to_undirected(edge_index, num_nodes=num_nodes)
            raw_texts = self._load_ogb_arxiv_texts(num_nodes)
        else:  # ogbn-products
            raw_texts = self._load_ogb_products_texts(num_nodes)

        train_mask = self._idx_to_mask(split_idx["train"], num_nodes)
        val_mask = self._idx_to_mask(split_idx["valid"], num_nodes)
        test_mask = self._idx_to_mask(split_idx["test"], num_nodes)
        train_mask, val_mask, test_mask = self._maybe_resplit(
            train_mask, val_mask, test_mask
        )

        return Data(
            x=g.x,
            edge_index=edge_index,
            y=y,
            raw_texts=raw_texts,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )

    def _build_ogb_dataset(self):
        """Construct the OGB dataset non-interactively.

        Two compatibility shims are applied only for the duration of the call:
        (1) ``torch.load`` is forced to ``weights_only=False`` so the torch>=2.6
        default doesn't break unpickling of OGB's cached PyG graph; (2) OGB's
        interactive "This will download X GB, proceed?" prompt is auto-accepted
        so first-use downloads work in a subprocess / training run with no stdin.
        """
        import builtins

        from ogb.nodeproppred import PygNodePropPredDataset
        from ogb.nodeproppred import dataset_pyg as _pyg_mod

        orig_load = torch.load

        def _patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            try:
                return orig_load(*args, **kwargs)
            except TypeError:
                # torch < 1.13 has no ``weights_only`` argument.
                kwargs.pop("weights_only", None)
                return orig_load(*args, **kwargs)

        # ``dataset_pyg`` does ``from ogb.utils.url import decide_download``, so we
        # patch the name in that module (patching ogb.utils.url is too late).
        orig_decide = getattr(_pyg_mod, "decide_download", None)
        orig_input = builtins.input

        torch.load = _patched_load
        if orig_decide is not None:
            _pyg_mod.decide_download = lambda url: True
        # OGB may block on interactive prompts (download confirmation, dataset
        # version update). Auto-accept so this runs headless / in a subprocess.
        builtins.input = lambda *a, **k: "y"
        try:
            return PygNodePropPredDataset(name=self.name, root=self._ogb_root())
        finally:
            torch.load = orig_load
            builtins.input = orig_input
            if orig_decide is not None:
                _pyg_mod.decide_download = orig_decide

    def _ogb_root(self) -> str:
        """Directory OGB downloads into (``<local_dir|cache_dir|data>/ogb``)."""
        import os

        root = (
            OmegaConf.select(self.cfg, "dirs.local_dir", default=None)
            or OmegaConf.select(self.cfg, "dirs.cache_dir", default=None)
            or "data"
        )
        return os.path.join(str(root), "ogb")

    @staticmethod
    def _idx_to_mask(idx: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Convert an OGB split index tensor to a 1-D boolean node mask."""
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx.view(-1)] = True
        return mask

    def _load_ogb_arxiv_texts(self, num_nodes: int) -> list[str]:
        """Build per-node ``"Title: .. Abstract: .."`` strings for ogbn-arxiv.

        Uses OGB's ``mapping/nodeidx2paperid`` (node idx -> MAG paper id) joined
        with the ``titleabs.tsv`` title/abstract table, downloading the latter on
        first use. Output is ordered by node index and has length ``num_nodes``.
        """
        import os
        import urllib.request

        data_root = os.path.join(self._ogb_root(), "ogbn_arxiv")
        titleabs_path = os.path.join(data_root, "titleabs.tsv.gz")
        if not os.path.exists(titleabs_path):
            os.makedirs(data_root, exist_ok=True)
            urllib.request.urlretrieve(self.OGB_ARXIV_TEXT_URL, titleabs_path)

        paper_ids = pd.read_csv(
            os.path.join(data_root, "mapping", "nodeidx2paperid.csv.gz")
        )
        paper_ids.columns = ["ID", "mag_id"]
        meta = pd.read_table(
            titleabs_path,
            header=None,
            skiprows=[0],
            names=["mag_id", "title", "abstract"],
        )
        meta = meta.dropna(subset=["mag_id"])
        meta["mag_id"] = meta["mag_id"].astype("int64")

        merged = paper_ids.merge(meta, how="left", on="mag_id").sort_values("ID")
        titles = merged["title"].fillna("").astype(str).tolist()
        abstracts = merged["abstract"].fillna("").astype(str).tolist()
        texts = [f"Title: {t}. Abstract: {a}" for t, a in zip(titles, abstracts)]
        assert len(texts) == num_nodes, (len(texts), num_nodes)
        return texts

    def _load_ogb_products_texts(self, num_nodes: int) -> list[str]:
        """Build per-node ``"Title: .. Content: .."`` strings for ogbn-products.

        Uses OGB's ``mapping/nodeidx2asin`` (node idx -> Amazon asin) joined with
        the Amazon-3M raw text (``trn.json``/``tst.json``, keyed by asin/uid). The
        Amazon-3M archive (~1.5 GB) is downloaded from Google Drive via ``gdown``
        on first use. Output is ordered by node index and has length ``num_nodes``.
        """
        import json
        import os

        data_root = os.path.join(self._ogb_root(), "ogbn_products")
        raw_dir = os.path.join(data_root, "Amazon-3M.raw")
        self._maybe_download_products_raw(data_root, raw_dir)

        # asin (== "uid" in the raw json) -> (title, content)
        records: dict[str, dict] = {}
        for split_file in ("trn.json", "tst.json"):
            fpath = os.path.join(raw_dir, split_file)
            with open(fpath, "r", encoding="utf_8_sig") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    dic = json.loads(line)
                    records[dic["uid"]] = {
                        "title": str(dic.get("title", "")).strip("\"\n"),
                        "content": str(dic.get("content", "")),
                    }

        asin = pd.read_csv(
            os.path.join(data_root, "mapping", "nodeidx2asin.csv.gz")
        )
        asin.columns = ["ID", "asin"]
        asin = asin.sort_values("ID")

        texts = []
        for a in asin["asin"].astype(str).tolist():
            rec = records.get(a)
            texts.append(
                "" if rec is None else f"Title: {rec['title']}. Content: {rec['content']}"
            )
        assert len(texts) == num_nodes, (len(texts), num_nodes)
        return texts

    def _maybe_download_products_raw(self, data_root: str, raw_dir: str) -> None:
        """Download + extract the Amazon-3M raw text for ogbn-products (once)."""
        import gzip
        import os

        from ogb.utils.url import extract_zip

        if os.path.exists(os.path.join(raw_dir, "trn.json")) and os.path.exists(
            os.path.join(raw_dir, "tst.json")
        ):
            return

        os.makedirs(data_root, exist_ok=True)
        zip_path = os.path.join(data_root, "Amazon-3M.raw.zip")
        if not os.path.exists(zip_path) and not os.path.exists(raw_dir):
            try:
                import gdown
            except ImportError as e:
                raise ImportError(
                    "ogbn-products raw text is hosted on Google Drive and requires "
                    "`gdown` to download (~1.5 GB). Install it with `pip install gdown`."
                ) from e

            gdown.download(
                url=self.OGB_PRODUCTS_TEXT_URL, output=zip_path, quiet=False, fuzzy=True
            )
        if not os.path.exists(raw_dir):
            extract_zip(zip_path, data_root)

        # The archive ships gzipped json; decompress in place if needed.
        for name in ("trn.json", "tst.json"):
            gz = os.path.join(raw_dir, name + ".gz")
            out = os.path.join(raw_dir, name)
            if os.path.exists(gz) and not os.path.exists(out):
                with gzip.GzipFile(gz) as fin, open(out, "wb") as fout:
                    fout.write(fin.read())

    def _split_ratios(self) -> tuple[float, float, float]:
        """(train, val, test) ratios from ``cfg.data.resplit`` (default 48/32/10)."""
        rs = OmegaConf.select(self.cfg, "data.resplit", default=None)
        if rs is None:
            return 0.48, 0.32, 0.10
        return (
            float(rs.get("train_ratio", 0.48)),
            float(rs.get("val_ratio", 0.32)),
            float(rs.get("test_ratio", 0.10)),
        )

    def _random_split(
        self,
        num_nodes: int,
        train_ratio: float | None = None,
        val_ratio: float | None = None,
        test_ratio: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Deterministic train/val/test node split, seeded by ``cfg.seed``.

        Ratios default to ``cfg.data.resplit`` (48/32/10). The permutation depends
        only on the seed and node count, so the GNN (in-process) and LLM
        (subprocess) always receive identical splits. Ratios need not sum to 1 --
        any leftover nodes are simply excluded from all three masks.
        """
        d_tr, d_va, d_te = self._split_ratios()
        train_ratio = d_tr if train_ratio is None else train_ratio
        val_ratio = d_va if val_ratio is None else val_ratio
        test_ratio = d_te if test_ratio is None else test_ratio

        generator = torch.Generator().manual_seed(int(self.cfg.seed))
        perm = torch.randperm(num_nodes, generator=generator)
        n_train = int(train_ratio * num_nodes)
        n_val = int(val_ratio * num_nodes)
        n_test = int(test_ratio * num_nodes)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[perm[:n_train]] = True
        val_mask[perm[n_train : n_train + n_val]] = True
        test_mask[perm[n_train + n_val : n_train + n_val + n_test]] = True
        return train_mask, val_mask, test_mask

    def _load_tensor(self, filename: str) -> torch.Tensor:
        """Load and cache tensor files from the dataset REPO."""
        if filename not in self._cache:
            path = self._download_hub_file(f"{self.name}/{filename}")
            self._cache[filename] = self._torch_load(path)
        return self._cache[filename]

    def _load_csv(self, filename: str) -> pd.DataFrame:
        """Load and cache a CSV file from the dataset REPO."""
        if filename not in self._cache:
            path = self._download_hub_file(f"{self.name}/{filename}")
            self._cache[filename] = pd.read_csv(path)
        return self._cache[filename]

    def _download_hub_file(self, rel_path: str) -> str:
        """Fetch a file from the dataset REPO via its direct ``resolve`` URL.

        Bypasses ``huggingface_hub`` on purpose: the pinned GLEM `ct` env ships an
        old hub client that can't follow HuggingFace's relative redirects and
        lacks the ``local_dir`` argument. ``urllib`` resolves those redirects
        correctly, so this works on every env. Files are cached on disk under
        ``cfg.dirs.local_dir`` (mirroring the repo's ``<name>/<file>`` layout) and
        a missing file raises ``RemoteEntryNotFoundError`` so the existing
        fallbacks (e.g. ``processed_data.pt`` -> ``<Name>.pt``) still work.
        """
        import os
        import urllib.error
        import urllib.request

        root = (
            OmegaConf.select(self.cfg, "dirs.local_dir", default=None)
            or OmegaConf.select(self.cfg, "dirs.cache_dir", default=None)
            or "data"
        )
        local_path = os.path.join(str(root), rel_path)
        if not os.path.exists(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            url = (
                f"https://huggingface.co/datasets/{self.REPO}/resolve/main/{rel_path}"
            )
            tmp = local_path + ".tmp"
            try:
                urllib.request.urlretrieve(url, tmp)
            except urllib.error.HTTPError as e:
                if e.code in (401, 403, 404):
                    raise RemoteEntryNotFoundError(rel_path) from e
                raise
            os.replace(tmp, local_path)
        return local_path

    @staticmethod
    def _torch_load(path: str):
        """``torch.load`` that works across torch *and* torch_geometric versions.

        Handles the ``weights_only`` default that flipped in torch>=2.6 (and is
        absent before 1.13), and injects stub classes for PyG's newer
        ``DataEdgeAttr`` / ``DataTensorAttr`` schema types when loading a graph
        pickled by a newer torch_geometric than the one installed (e.g. the GLEM
        `ct` env's PyG 2.0.3). Only the schema classes are stubbed -- the actual
        ``x`` / ``edge_index`` / ``y`` tensors are plain and load unchanged.
        """
        try:
            import torch_geometric.data.data as _tg_data

            for _cls in ("DataEdgeAttr", "DataTensorAttr"):
                if not hasattr(_tg_data, _cls):
                    setattr(_tg_data, _cls, type(_cls, (), {}))
        except Exception:
            pass

        try:
            return torch.load(path, weights_only=False)
        except TypeError:
            return torch.load(path)

    @property
    def load_processed_data(self) -> Data:
        try:
            if self.name in self.AVAILABLE_DATASETS:
                return self._load_tensor("processed_data.pt")
            else:
                return self._load_tensor(f"{self.name.capitalize()}.pt")
        except RemoteEntryNotFoundError:
            return self._load_tensor(f"{self.name.capitalize()}.pt")

    @property
    def load_raw_texts(self) -> list[str]:
        try:
            return self._load_tensor("raw_texts.pt")
        except RemoteEntryNotFoundError:
            return self._load_tensor(f"{self.name.capitalize()}.txt")

    @property
    def load_sbert_x(self) -> torch.Tensor:
        return self._load_tensor("sbert_x.pt")

    @property
    def load_roberta_x(self) -> torch.Tensor:
        return self._load_tensor("roberta_x.pt")

    @property
    def load_llmgpt_text_embedding_3_large_x(self) -> torch.Tensor:
        return self._load_tensor("llmgpt_text-embedding-3-large_x.pt")

    @property
    def num_classes(self) -> int:
        return int(self.data.y.max().item()) + 1


if __name__ == "__main__":
    from hydra import main
    from loguru import logger

    root = f'{DATA_PATH}tag'
    cfg = OmegaConf.create({'seed': 0,
                            'dirs': {'local_dir': root, 'cache_dir': root},
                            'data': {'resplit': {'enabled': False}}})
    def test_load_dataset(cfg: OmegaConf):
        loadable = (
            TAGDataset.AVAILABLE_DATASETS
            + TAGDataset.WEBKB_DATASETS
        )
        for dataset_name in loadable:
            logger.info(f"Testing dataset: {dataset_name}")
            dataset = TAGDataset(cfg, name=dataset_name)
            logger.info(dataset.data)
            logger.info(f"Sample text: {dataset.data.raw_texts[0]}")

    test_load_dataset(cfg)
