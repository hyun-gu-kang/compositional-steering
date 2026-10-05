"""Figure style and the plotting functions for the main-text figures.

Style: black spines/ticks, bold axis labels, viridis palette for ordered
series, light grid, frameless legends, 300-dpi PDF output with a tight bbox.
Panels are sized so that the fixed point-size fonts stay legible after LaTeX
scales the figure to the paper width.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import config as C

sns.set_style("whitegrid")
mpl.rcParams["font.family"] = "sans-serif"

MODEL_COLORS = dict(zip(C.MODEL_NAMES, sns.color_palette("viridis", len(C.MODEL_NAMES))))
BASELINE_LINE_STYLES = ("--", ":", "-.")


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------
def style_axes(ax):
    """Black spines and ticks with a light grid. Call on every new Axes."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.2)
    ax.tick_params(axis="both", colors="black")
    ax.grid(True, alpha=0.35)


def save_fig(fig, output_path, min_width_in=None):
    """Save as a 300-dpi PDF with a tight bounding box.

    min_width_in pads the bbox on the left when the tight crop would be
    narrower than a sibling panel (e.g. because of a shorter y-label).
    """
    output_path = os.path.splitext(output_path)[0] + ".pdf"
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    bbox = "tight"
    if min_width_in is not None:
        fig.canvas.draw()
        bbox = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.1)
        if bbox.width < min_width_in:
            bbox = mpl.transforms.Bbox.from_extents(
                bbox.x0 - (min_width_in - bbox.width), bbox.y0, bbox.x1, bbox.y1
            )
    fig.savefig(output_path, dpi=300, bbox_inches=bbox)
    plt.close(fig)
    print(f"Saved {output_path}")


def _fmt(value):
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:g}"


def _header(label):
    """Invisible legend entry used as an inline legend title."""
    return Line2D([], [], color="none", linestyle="none"), label


# ---------------------------------------------------------------------------
# Figure 2: single-attribute steering across layers and strengths
# ---------------------------------------------------------------------------
def plot_single_attribute(panels, output_path, ylabel="Steering score", ylim=(0.0, 1.0)):
    """One panel per model. Lines: steering strengths; bands: uncertainty;
    black horizontal lines: prompt-based baselines.

    panels: {model_name: {"stats": DataFrame[layer, alpha, mean, lo, hi],
                          "baselines": DataFrame[prompt_variant, score]} or None}
    Colors encode the strength *rank*, so Llama and Qwen alphas share entries.
    """
    model_names = list(panels)
    family_alphas = {}
    for name, panel in panels.items():
        if panel is not None:
            family_alphas.setdefault(C.MODELS[name]["family"], sorted(panel["stats"]["alpha"].unique()))
    n_ranks = max((len(a) for a in family_alphas.values()), default=0)
    palette = sns.color_palette("viridis", n_ranks)

    width, height = 2.3 * len(model_names), 2.6
    fig, axes = plt.subplots(1, len(model_names), figsize=(width, height), sharey=True, squeeze=False)
    rank_handles, baseline_handles = {}, {}

    for ax, name in zip(axes.ravel(), model_names):
        panel = panels[name]
        if panel is None:
            ax.text(0.5, 0.5, "Missing data", ha="center", va="center", transform=ax.transAxes)
        else:
            stats = panel["stats"]
            layers = sorted(stats["layer"].astype(int).unique())
            for rank, alpha in enumerate(sorted(stats["alpha"].unique())):
                s = stats[stats["alpha"] == alpha].set_index("layer").reindex(layers)
                (line,) = ax.plot(layers, s["mean"], marker="o", linewidth=2.8,
                                  markersize=5.0, color=palette[rank])
                ax.fill_between(layers, np.clip(s["lo"], *ylim), np.clip(s["hi"], *ylim),
                                color=palette[rank], alpha=0.18, linewidth=0)
                rank_handles.setdefault(rank, line)

            baselines = panel["baselines"].sort_values("prompt_variant")
            for i, row in enumerate(baselines.itertuples(index=False)):
                line = ax.axhline(row.score, color="black", linewidth=1.2, alpha=0.7,
                                  linestyle=BASELINE_LINE_STYLES[i % len(BASELINE_LINE_STYLES)])
                baseline_handles.setdefault(int(row.prompt_variant), line)
            ax.set_xticks(layers)

        ax.set_title(C.MODELS[name]["title"], fontsize=12, fontweight="bold", pad=6)
        ax.set_xlabel("Layer", fontsize=12, fontweight="bold", labelpad=5)
        ax.set_ylim(*ylim)
        style_axes(ax)
        ax.tick_params(axis="both", labelsize=10)

    families = [f for f in ("llama", "qwen") if f in family_alphas]
    handles, labels = [], []
    if rank_handles:
        h, l = _header(rf"Steering $\alpha$ ({' / '.join(f.capitalize() for f in families)}):")
        handles.append(h), labels.append(l)
        for rank in sorted(rank_handles):
            handles.append(rank_handles[rank])
            labels.append(" / ".join(
                _fmt(family_alphas[f][rank]) if rank < len(family_alphas[f]) else "--"
                for f in families
            ))
    if baseline_handles:
        h, l = _header("Prompt baseline:")
        handles.append(h), labels.append(l)
        for variant in sorted(baseline_handles):
            handles.append(baseline_handles[variant])
            labels.append(f"Prompt {variant}")

    fig.legend(handles=handles, labels=labels, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=len(handles), frameon=False, fontsize=10, columnspacing=0.6,
               handletextpad=0.3, handlelength=1.6, borderaxespad=0.0)
    fig.supylabel(ylabel, fontsize=12, fontweight="bold", x=-0.005)
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.22, top=0.70, wspace=0.12)
    save_fig(fig, output_path, min_width_in=width)


# ---------------------------------------------------------------------------
# Figures 3 and 4: compositional steering across models and language pairs
# ---------------------------------------------------------------------------
def plot_composition(scores, baselines, output_path, ylabel="Steering Score",
                     figsize=(3.1, 2.45), seed=C.SEED):
    """Box plot per model over language pairs, one point per language pair,
    and a white diamond at the mean prompt-baseline score.

    scores:    {model_name: array of per-language-pair steering scores or None}
    baselines: {model_name: mean prompt-baseline score or None}
    """
    model_names = list(scores)
    fig, ax = plt.subplots(figsize=figsize)
    rng = np.random.default_rng(seed)

    for x, name in enumerate(model_names):
        values = scores[name]
        if values is None or len(values) == 0:
            ax.text(x, 0.5, "N/A", ha="center", va="center", fontsize=8, color="gray")
            continue
        color = MODEL_COLORS[name]
        box = ax.boxplot(
            [values], positions=[x], widths=0.52, patch_artist=True, showfliers=False,
            medianprops={"color": "black", "linewidth": 1.3},
            boxprops={"edgecolor": color, "linewidth": 1.2},
            whiskerprops={"color": color, "linewidth": 1.0},
            capprops={"color": color, "linewidth": 1.0},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
        ax.scatter(x + rng.uniform(-0.08, 0.08, size=len(values)), values, s=18, color=color,
                   edgecolor="white", linewidth=0.4, alpha=0.85, zorder=3)
        if baselines.get(name) is not None:
            ax.scatter(x, baselines[name], marker="D", s=30, facecolor="white",
                       edgecolor="black", linewidth=1.0, zorder=4)

    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels([])
    ax.set_xlim(-0.55, len(model_names) - 0.45)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(ylabel, fontsize=9, fontweight="bold")
    style_axes(ax)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", visible=False)

    handles = [
        Patch(facecolor=MODEL_COLORS[n], edgecolor=MODEL_COLORS[n], alpha=0.45, label=C.MODELS[n]["title"])
        for n in model_names
    ]
    if any(v is not None for v in baselines.values()):
        handles.append(Line2D([0], [0], marker="D", linestyle="none", markerfacecolor="white",
                              markeredgecolor="black", markeredgewidth=1.0, markersize=4.5,
                              label="Prompt baseline"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.71, 0.96, 0.22),
               ncols=3, mode="expand", frameon=False, fontsize=7, handlelength=1.0,
               handletextpad=0.35, columnspacing=0.6, borderaxespad=0.0)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.68, bottom=0.12)
    save_fig(fig, output_path, min_width_in=figsize[0])
