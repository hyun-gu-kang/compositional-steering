"""Aggregation of per-sample metrics into steering scores.

For every condition, each metric is first averaged over the generations and
the steering score is the harmonic mean of these averages (Section 4.2).
"""

import os

import numpy as np
import pandas as pd

from . import config as C
from . import paths
from .records import DESC_COLS, KEY_COLS

SAMPLE_COLS = KEY_COLS + ["prompt_id"]


def summarize(per_sample):
    """Mean score per condition."""
    return (
        per_sample.groupby(KEY_COLS, as_index=False, dropna=False)
        .agg(
            layer=("layer", "first"),
            alpha=("alpha", "first"),
            prompt_variant=("prompt_variant", "first"),
            score=("score", "mean"),
            n=("score", "size"),
            n_valid=("score", "count"),
        )
    )


def save_metric(per_sample, root, run, model, exp, metric):
    out = paths.per_sample_path(root, run, model, exp, metric)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    per_sample.to_csv(out, index=False)
    summary = summarize(per_sample)
    summary.to_csv(paths.summary_path(root, run, model, exp, metric), index=False)
    print(f"Saved {out}")
    return summary


def load_metric(root, run, model, exp, metric):
    df = pd.read_csv(paths.per_sample_path(root, run, model, exp, metric))
    df["target_lang"] = df["target_lang"].astype(str)
    df["condition"] = df["condition"].astype(str)
    return df


def load_aligned(root, run, model, exp, metrics=None):
    """Per-sample scores of all metrics of `exp`, one column per metric.

    Rows are aligned one-to-one on (condition, prompt_id); a mismatch between
    metric files raises instead of silently dropping samples.
    """
    metrics = list(metrics or C.exp_metrics(exp))
    merged = None
    for metric in metrics:
        df = load_metric(root, run, model, exp, metric)
        df = df[SAMPLE_COLS + DESC_COLS + ["score"]].rename(columns={"score": metric})
        if merged is None:
            merged = df
            continue
        merged = merged.merge(
            df.drop(columns=DESC_COLS),
            on=SAMPLE_COLS,
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        unmatched = merged["_merge"] != "both"
        if unmatched.any():
            raise ValueError(
                f"{run}/{model}/{exp}: {int(unmatched.sum())} samples are missing in "
                f"one of {metrics}. Example: {merged.loc[unmatched, SAMPLE_COLS].head(3).to_dict('records')}"
            )
        merged = merged.drop(columns="_merge")
    return merged


def harmonic_mean(values):
    """Row-wise harmonic mean over the last axis; 0 if any component is 0."""
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape[:-1], np.nan)
    finite = np.isfinite(values).all(axis=-1)
    positive = finite & (values > 0).all(axis=-1)
    out[finite & ~positive] = 0.0
    out[positive] = values.shape[-1] / np.sum(1.0 / values[positive], axis=-1)
    return out


def condition_scores(aligned, metrics):
    """Metric means and their harmonic mean (the steering score) per condition."""
    grouped = aligned.groupby(KEY_COLS, as_index=False, dropna=False)
    scores = grouped.agg(
        layer=("layer", "first"),
        alpha=("alpha", "first"),
        prompt_variant=("prompt_variant", "first"),
        n=("prompt_id", "size"),
        **{metric: (metric, "mean") for metric in metrics},
    )
    scores["score"] = harmonic_mean(scores[list(metrics)].to_numpy())
    return scores
