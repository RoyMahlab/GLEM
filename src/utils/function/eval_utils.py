"""Node-classification evaluator factory.

OGB datasets use ``ogb.nodeproppred.Evaluator``, which requires a valid OGB
dataset name. The non-OGB TAG datasets (``loader='tag'``) have ``ogb_name=None``,
so for them we return a drop-in accuracy evaluator with the same
``eval({'y_true','y_pred'}) -> {'acc': float}`` interface the callers expect.
"""
import numpy as np


class AccEvaluator:
    """Minimal OGB-Evaluator-compatible accuracy evaluator.

    ``eval({'y_true': (N,1) int, 'y_pred': (N,1) int}) -> {'acc': float}`` —
    matches how GLEM's LM/GNN trainers call the OGB evaluator (predictions are
    already argmax'd class indices).
    """

    def eval(self, input_dict):
        y_true, y_pred = input_dict["y_true"], input_dict["y_pred"]
        if hasattr(y_true, "detach"):
            y_true = y_true.detach().cpu().numpy()
        if hasattr(y_pred, "detach"):
            y_pred = y_pred.detach().cpu().numpy()
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        return {"acc": float((y_true == y_pred).mean())}


def get_node_evaluator(cf):
    """OGB ``Evaluator`` for OGB datasets, else the TAG accuracy evaluator."""
    if getattr(cf.data, "loader", None) == "tag":
        return AccEvaluator()
    from ogb.nodeproppred import Evaluator

    return Evaluator(name=cf.data.ogb_name)
