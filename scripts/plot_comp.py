"""Figures 3 and 4: compositional steering across models and language pairs.

    python scripts/plot_comp.py
    python scripts/plot_comp.py --exps lang_jb

For every EN->X pair the steering score is the harmonic mean of the metric
averages over the 70 generations (L+J: LFS, JBS, OR; L+C: LFS, CCS, OR;
L+J+C: LFS, JBS, CCS, OR). Boxes show the distribution over the nine pairs;
the white diamond is the prompt baseline (best suffix combination) averaged
over pairs.
"""

import argparse
import os

import pandas as pd

from compsteer import config as C
from compsteer import paths
from compsteer.aggregate import condition_scores, load_aligned
from compsteer.plotting import plot_composition

FIGURES = {"lang_jb": "fig3a_lang_jb", "lang_conc": "fig3b_lang_conc", "lang_jb_conc": "fig4_lang_jb_conc"}


def pair_scores(root, run, model, exp):
    metrics = list(C.exp_metrics(exp))
    return condition_scores(load_aligned(root, run, model, exp, metrics), metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=C.MODEL_NAMES, choices=C.MODEL_NAMES)
    parser.add_argument("--exps", nargs="+", default=list(C.COMP_EXPS), choices=C.COMP_EXPS)
    parser.add_argument("--root", default=paths.DEFAULT_ROOT)
    args = parser.parse_args()

    out_dir = paths.figure_dir(args.root)
    os.makedirs(out_dir, exist_ok=True)

    for exp in args.exps:
        scores, baselines, tables = {}, {}, []
        for model in args.models:
            try:
                steering = pair_scores(args.root, "steering", model, exp)
                baseline = pair_scores(args.root, "prompt_baseline", model, exp)
            except FileNotFoundError as error:
                print(f"Skipping {model}/{exp}: {error}")
                scores[model], baselines[model] = None, None
                continue
            baseline = baseline[baseline["prompt_variant"] == 1]  # best suffix combination
            scores[model] = steering["score"].to_numpy()
            baselines[model] = float(baseline["score"].mean())
            tables += [steering, baseline]

        name = FIGURES[exp]
        plot_composition(scores, baselines, os.path.join(out_dir, f"{name}.pdf"))
        if tables:
            pd.concat(tables, ignore_index=True).to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)


if __name__ == "__main__":
    main()
