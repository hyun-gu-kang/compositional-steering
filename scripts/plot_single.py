"""Figure 2: single-attribute steering across layers and steering strengths.

    python scripts/plot_single.py
    python scripts/plot_single.py --models llama_8b qwen_14b

For each (layer, alpha) the steering score is the harmonic mean of the
metric averages over the 70 generations (LFS+OR, JBS+OR, CCS+OR).
  language:              line = mean over the nine EN->X pairs,
                         band = +/- 1 std across pairs
  jailbreak/conciseness: band = +/- 1 std across prompts of the per-prompt
                         harmonic mean
Horizontal lines are the three prompt baselines (averaged over pairs for language).
"""

import argparse
import os

import pandas as pd

from compsteer import config as C
from compsteer import paths
from compsteer.aggregate import condition_scores, harmonic_mean, load_aligned
from compsteer.plotting import plot_single_attribute

FIGURES = {"language": "fig2a_language", "jailbreak": "fig2b_jailbreak", "conciseness": "fig2c_conciseness"}


def steering_stats(root, model, exp):
    metrics = list(C.exp_metrics(exp))
    aligned = load_aligned(root, "steering", model, exp, metrics)
    scores = condition_scores(aligned, metrics)

    if exp == "language":
        stats = scores.groupby(["layer", "alpha"], as_index=False)["score"].agg(mean="mean", std="std")
    else:
        # One condition per (layer, alpha): the spread is taken over prompts.
        aligned["prompt_score"] = harmonic_mean(aligned[metrics].to_numpy())
        spread = aligned.groupby("condition", as_index=False)["prompt_score"].std().rename(
            columns={"prompt_score": "std"})
        stats = scores.merge(spread, on="condition", validate="one_to_one")
        stats = stats[["layer", "alpha", "score", "std"]].rename(columns={"score": "mean"})

    stats["lo"] = stats["mean"] - stats["std"]
    stats["hi"] = stats["mean"] + stats["std"]
    return stats


def baseline_scores(root, model, exp):
    metrics = list(C.exp_metrics(exp))
    scores = condition_scores(load_aligned(root, "prompt_baseline", model, exp, metrics), metrics)
    return scores.groupby("prompt_variant", as_index=False)["score"].mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=C.MODEL_NAMES, choices=C.MODEL_NAMES)
    parser.add_argument("--exps", nargs="+", default=list(C.SINGLE_EXPS), choices=C.SINGLE_EXPS)
    parser.add_argument("--root", default=paths.DEFAULT_ROOT)
    args = parser.parse_args()

    out_dir = paths.figure_dir(args.root)
    os.makedirs(out_dir, exist_ok=True)

    for exp in args.exps:
        panels, tables = {}, []
        for model in args.models:
            try:
                stats = steering_stats(args.root, model, exp)
                baselines = baseline_scores(args.root, model, exp)
            except FileNotFoundError as error:
                print(f"Skipping {model}/{exp}: {error}")
                panels[model] = None
                continue
            panels[model] = {"stats": stats, "baselines": baselines}
            tables.append(stats.assign(model=model, series="steering"))
            tables.append(baselines.rename(columns={"score": "mean"}).assign(model=model, series="prompt_baseline"))

        name = FIGURES[exp]
        plot_single_attribute(panels, os.path.join(out_dir, f"{name}.pdf"))
        if tables:
            pd.concat(tables, ignore_index=True).to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)


if __name__ == "__main__":
    main()
