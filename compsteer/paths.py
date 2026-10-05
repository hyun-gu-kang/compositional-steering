"""Output directory layout.

outputs/
  mean_acts/{model}/{attribute}/{condition}.pt
  generations/{run}/{model}/{exp}[_{target_lang}].json
  eval/{run}/{model}/{exp}/{metric}_per_sample.csv
  eval/{run}/{model}/{exp}/{metric}_summary.csv
  figures/*.pdf

`run` is one of:
  steering         activation steering (single-attribute or compositional)
  prompt_baseline  explicit instruction suffixes instead of steering
  ccs_reference    L+J steering with 512 tokens, the CCS reference for L+J+C
"""

import os

from . import config as C

DEFAULT_ROOT = "outputs"
RUNS = ("steering", "prompt_baseline", "ccs_reference")


def mean_act_path(root, model, attribute, condition):
    return os.path.join(root, "mean_acts", model, attribute, f"{condition}.pt")


def generation_stem(exp, target_lang):
    if "language" in C.exp_attributes(exp):
        return f"{exp}_{target_lang}"
    return exp


def generation_path(root, run, model, exp, target_lang):
    return os.path.join(
        root, "generations", run, model, f"{generation_stem(exp, target_lang)}.json"
    )


def eval_dir(root, run, model, exp):
    return os.path.join(root, "eval", run, model, exp)


def per_sample_path(root, run, model, exp, metric):
    return os.path.join(eval_dir(root, run, model, exp), f"{metric}_per_sample.csv")


def summary_path(root, run, model, exp, metric):
    return os.path.join(eval_dir(root, run, model, exp), f"{metric}_summary.csv")


def figure_dir(root):
    return os.path.join(root, "figures")
