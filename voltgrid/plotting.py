"""Plot helpers for the VoltGrid scikit-learn masterclass.

Every chart in the course comes from this file, so the whole book reads as
one visual system. Colours are a validated palette - categorical hues are
assigned in a FIXED order and never cycled.

Design rules enforced here:
  * categorical hues in fixed order, never generated
  * one y-axis, never dual
  * sequential = one hue light->dark; diverging = blue/red with a GREY midpoint
  * legend whenever there are 2+ series; text never wears the series colour
  * recessive grid and axes, thin marks
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# --------------------------------------------------------------- design tokens
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e3e2de"

# Categorical slots - FIXED ORDER. Slot 1 is always blue, slot 2 always orange.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Sequential (magnitude): one hue, light -> dark
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Diverging (polarity around a meaningful midpoint, e.g. p = 0.5)
DIV_LOW, DIV_MID, DIV_HIGH = "#2a78d6", "#f0efec", "#d03b3b"

STATUS = {"good": "#0ca30c", "warning": "#fab219",
          "serious": "#ec835a", "critical": "#d03b3b"}

CMAP_SEQ = LinearSegmentedColormap.from_list("vg_seq", SEQ_BLUE)
CMAP_DIV = LinearSegmentedColormap.from_list("vg_div", [DIV_LOW, DIV_MID, DIV_HIGH])

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SOFT,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 2.0,
    "font.size": 10,
    "figure.dpi": 110,
}


def use_style() -> None:
    """Apply the VoltGrid look to every subsequent matplotlib figure."""
    mpl.rcParams.update(STYLE)
    # remove the top/right spines by default
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False


def _finish(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


# ------------------------------------------------------------------- 2-D views
def plot_decision_boundary(model, X, y, feature_names=None, ax=None,
                           title="Decision boundary", resolution=260):
    """Shade predicted probability over a 2-D feature plane.

    X must have exactly two columns. The probability surface uses the
    DIVERGING ramp because 0.5 is a real midpoint, and the grey midpoint
    makes "the model is unsure" visible.
    """
    X = np.asarray(X, dtype=float)
    if X.shape[1] != 2:
        raise ValueError(f"Need exactly 2 features to draw a boundary, got {X.shape[1]}.")
    y = np.asarray(y)

    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 5.0))

    pad_x = 0.06 * (X[:, 0].max() - X[:, 0].min())
    pad_y = 0.06 * (X[:, 1].max() - X[:, 1].min())
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - pad_x, X[:, 0].max() + pad_x, resolution),
        np.linspace(X[:, 1].min() - pad_y, X[:, 1].max() + pad_y, resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    if hasattr(model, "predict_proba"):
        zz = model.predict_proba(grid)[:, 1]
    elif hasattr(model, "decision_function"):
        d = model.decision_function(grid)
        zz = 1.0 / (1.0 + np.exp(-d))
    else:
        zz = model.predict(grid).astype(float)

    ax.contourf(xx, yy, zz.reshape(xx.shape), levels=24,
                cmap=CMAP_DIV, alpha=0.85, vmin=0, vmax=1)
    ax.contour(xx, yy, zz.reshape(xx.shape), levels=[0.5],
               colors=[INK], linewidths=1.4, linestyles="--")

    for i, cls in enumerate(np.unique(y)):
        m = y == cls
        ax.scatter(X[m, 0], X[m, 1], s=26, c=SERIES[i],
                   edgecolor=SURFACE, linewidth=1.1, label=f"class {cls}", zorder=3)

    ax.legend(loc="upper right")           # 2 series -> legend always present
    names = feature_names or ["feature 1", "feature 2"]
    return _finish(ax, title, names[0], names[1])


def plot_confusion(cm, labels=("negative", "positive"), ax=None,
                   title="Confusion matrix", normalize=False):
    """Confusion matrix on the SEQUENTIAL ramp with every cell labelled."""
    cm = np.asarray(cm, dtype=float)
    shown = cm / cm.sum(axis=1, keepdims=True) if normalize else cm

    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 4.2))

    ax.imshow(shown, cmap=CMAP_SEQ, vmin=0, vmax=shown.max())
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)

    hi = shown.max()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{shown[i, j]:.2f}" if normalize else f"{int(cm[i, j]):,}"
            # text wears ink, not the series colour; flip to white on dark cells
            ax.text(j, i, txt, ha="center", va="center", fontsize=11,
                    color="#ffffff" if shown[i, j] > 0.6 * hi else INK)
    return _finish(ax, title, "predicted", "actual")


def plot_roc(curves, ax=None, title="ROC curve"):
    """One or more ROC curves. `curves` is {label: (fpr, tpr, auc)}."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 5.0))

    for i, (label, (fpr, tpr, auc)) in enumerate(curves.items()):
        ax.plot(fpr, tpr, color=SERIES[i], label=f"{label}  AUC {auc:.3f}")
    ax.plot([0, 1], [0, 1], color=INK_SOFT, linewidth=1.2,
            linestyle=":", label="chance  AUC 0.500")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    return _finish(ax, title, "false positive rate", "true positive rate")


def plot_residuals(y_true, y_pred, ax=None, title="Residuals vs predicted"):
    """Residual scatter with a zero line - the fastest way to see bias."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 4.4))

    ax.scatter(y_pred, y_true - y_pred, s=12, c=SERIES[0],
               alpha=0.35, edgecolor="none")
    ax.axhline(0, color=INK, linewidth=1.3, linestyle="--")
    return _finish(ax, title, "predicted", "residual (actual - predicted)")


def plot_permutation_importance(importances, names, ax=None, top=12,
                                title="Permutation importance"):
    """Horizontal bars, sorted, directly labelled.

    Single series -> no legend box; the title names the measure.
    """
    importances = np.asarray(importances, dtype=float)
    order = np.argsort(importances)[::-1][:top][::-1]
    vals, labs = importances[order], np.asarray(names)[order]

    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 0.36 * len(vals) + 1.5))

    ax.barh(range(len(vals)), vals, color=SERIES[0], height=0.62)
    ax.set_yticks(range(len(vals)), labs)
    ax.grid(axis="y", visible=False)

    span = vals.max() - min(vals.min(), 0)
    for i, v in enumerate(vals):                   # direct labels
        ax.text(v + 0.012 * span, i, f"{v:.4f}",
                va="center", fontsize=8.5, color=INK_SOFT)
    ax.set_xlim(min(vals.min(), 0), vals.max() + 0.16 * span)
    return _finish(ax, title, "drop in score when shuffled", None)
