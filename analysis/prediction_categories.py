"""Prediction-category analysis: LLM vs GNN agreement breakdown.

Every node falls into one of four *prediction categories* based on which model
gets it right:

    * Both Correct  -- LLM ✓ and GNN ✓
    * Both Wrong    -- LLM ✗ and GNN ✗
    * LLM Only      -- LLM ✓, GNN ✗   (text clear, neighbourhood misleading)
    * GNN Only      -- GNN ✓, LLM ✗   (text ambiguous, neighbourhood informative)

This module reproduces the notebook prototypes (idea_fomulation.ipynb cells 13 &
14) as reusable functions that *return* Matplotlib figures (so callers can log
them to wandb) instead of calling ``plt.show``/``savefig`` directly:

    * ``plot_homophily_by_category``  -> "Local Homophily by Prediction Category"
    * ``plot_confidence_by_category`` -> "Model Confidence by Prediction Category"
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Category name -> plotting colour (green / red / blue / orange), matching the
# notebook prototypes.
CATEGORY_COLORS = ["#4CAF50", "#F44336", "#2196F3", "#FF9800"]
CATEGORY_NAMES = ["Both\nCorrect", "Both\nWrong", "LLM\nOnly", "GNN\nOnly"]


def category_masks(
    llm_correct: torch.Tensor,
    gnn_correct: torch.Tensor,
    node_mask: torch.Tensor | None = None,
) -> dict[str, np.ndarray]:
    """Boolean membership mask (numpy) for each of the four prediction categories.

    ``node_mask`` optionally restricts the analysis to a split (e.g. the test
    nodes); when ``None`` every node is considered.
    """
    llm = llm_correct.bool()
    gnn = gnn_correct.bool()
    keep = node_mask.bool() if node_mask is not None else torch.ones_like(llm)
    return {
        "Both\nCorrect": (llm & gnn & keep).cpu().numpy(),
        "Both\nWrong": (~llm & ~gnn & keep).cpu().numpy(),
        "LLM\nOnly": (llm & ~gnn & keep).cpu().numpy(),
        "GNN\nOnly": (~llm & gnn & keep).cpu().numpy(),
    }


def category_counts(
    llm_correct: torch.Tensor,
    gnn_correct: torch.Tensor,
    node_mask: torch.Tensor | None = None,
) -> dict[str, int]:
    """Node count per prediction category (keys flattened, e.g. ``both_correct``)."""
    masks = category_masks(llm_correct, gnn_correct, node_mask)
    return {
        name.replace("\n", "_").lower(): int(mask.sum())
        for name, mask in masks.items()
    }


def plot_homophily_by_category(
    homophily: torch.Tensor,
    llm_correct: torch.Tensor,
    gnn_correct: torch.Tensor,
    node_mask: torch.Tensor | None = None,
):
    """Boxplot of local homophily per prediction category (notebook cell 13)."""
    h = homophily.cpu().numpy()
    masks = category_masks(llm_correct, gnn_correct, node_mask)

    groups = {name: h[mask] for name, mask in masks.items()}
    groups = {k: v[~np.isnan(v)] for k, v in groups.items()}  # drop isolated nodes

    fig, ax = plt.subplots(figsize=(12, 7))
    positions = range(1, len(groups) + 1)
    # boxplot() errors on empty sequences; substitute NaN so the slot is drawn.
    values = [v if len(v) else np.array([np.nan]) for v in groups.values()]
    bp = ax.boxplot(
        values,
        patch_artist=True,
        notch=False,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], CATEGORY_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(list(groups.keys()), fontsize=12)
    ax.set_ylabel("Local Homophily", fontsize=12)
    ax.set_title("Local Homophily by Prediction Category", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    for i, vals in zip(positions, groups.values()):
        ax.text(i, -0.04, f"n={len(vals)}", ha="center", fontsize=9, color="gray")

    fig.tight_layout()
    return fig


def plot_confidence_by_category(
    llm_logits: torch.Tensor,
    gnn_logits: torch.Tensor,
    llm_correct: torch.Tensor,
    gnn_correct: torch.Tensor,
    node_mask: torch.Tensor | None = None,
):
    """Side-by-side GNN/LLM confidence boxplots per category (notebook cell 14)."""
    llm_conf = F.softmax(llm_logits, dim=-1).max(dim=-1).values.detach().cpu().numpy()
    gnn_conf = F.softmax(gnn_logits, dim=-1).max(dim=-1).values.detach().cpu().numpy()
    masks = category_masks(llm_correct, gnn_correct, node_mask)
    categories = list(masks.keys())

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=False)
    for ax, (conf, model_name) in zip(axes, [(gnn_conf, "GNN"), (llm_conf, "LLM")]):
        groups = [conf[mask] for mask in masks.values()]
        values = [g if len(g) else np.array([np.nan]) for g in groups]
        bp = ax.boxplot(
            values,
            patch_artist=True,
            notch=False,
            medianprops=dict(color="black", linewidth=2),
            flierprops=dict(marker="o", markersize=3, alpha=0.4),
        )
        for patch, color in zip(bp["boxes"], CATEGORY_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks(range(1, len(categories) + 1))
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylabel("Confidence (max softmax)", fontsize=11)
        ax.set_title(f"{model_name} Confidence by Category", fontsize=13)
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        for i, vals in enumerate(groups, start=1):
            if len(vals):
                ax.text(i, np.median(vals) + 0.02, f"{np.median(vals):.2f}",
                        ha="center", fontsize=8, color="black")
            ax.text(i, 1.02, f"n={len(vals)}", ha="center", fontsize=8, color="gray")

    fig.suptitle("Model Confidence by Prediction Category", fontsize=14,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig
