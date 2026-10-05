"""Score generations with one metric (LFS, OR, JBS, or CCS).

    python scripts/evaluate.py --metric lfs
    python scripts/evaluate.py --metric or --models llama_8b --exps language lang_jb
    python scripts/evaluate.py --metric ccs --runs steering prompt_baseline

Each metric is computed only for the experiments that use it (Table 2).
Missing generation files are skipped. Writes
outputs/eval/{run}/{model}/{exp}/{metric}_per_sample.csv and _summary.csv.
"""

import argparse

from compsteer import config as C
from compsteer import metrics as M
from compsteer import paths
from compsteer.aggregate import save_metric
from compsteer.records import load_records

ALL_EXPS = list(C.SINGLE_EXPS) + list(C.COMP_EXPS)


def use_ccs_reference(records, root):
    """L+J+C: measure length against Llama-3.1-8B L+J outputs (footnote 4)."""
    reference = load_records(root, "ccs_reference", C.CCS_REFERENCE_MODEL, C.CCS_REFERENCE_EXP)
    lookup = {(r["target_lang"], r["prompt_id"]): r["response"] for r in reference}
    missing = [(r["target_lang"], r["prompt_id"]) for r in records
               if (r["target_lang"], r["prompt_id"]) not in lookup]
    if missing:
        raise ValueError(f"{len(missing)} records have no CCS reference output, e.g. {missing[:3]}")
    for record in records:
        record["baseline"] = lookup[(record["target_lang"], record["prompt_id"])]
    return records


class Evaluator:
    """Loads the scorer for one metric once and applies it to many record sets."""

    def __init__(self, metric, batch_size):
        self.metric = metric
        self.batch_size = batch_size
        self._scorer = None
        self._tokenizers = {}

    def _load_scorer(self):
        if self._scorer is None:
            if self.metric == "lfs":
                self._scorer = M.load_lid_model()
            elif self.metric == "or":
                self._scorer = M.load_judge(C.OR_JUDGE)
            elif self.metric == "jbs":
                self._scorer = M.load_judge(C.JBS_JUDGE)
        return self._scorer

    def __call__(self, records, model_name, root):
        if self.metric == "lfs":
            return M.eval_lfs(records, self._load_scorer())

        if self.metric == "or":
            return M.eval_or(records, *self._load_scorer(), batch_size=self.batch_size or 128)

        if self.metric == "jbs":
            return M.eval_jbs(records, *self._load_scorer(), batch_size=self.batch_size or 64)

        if self.metric == "ccs":
            if model_name not in self._tokenizers:
                from transformers import AutoTokenizer

                self._tokenizers[model_name] = AutoTokenizer.from_pretrained(C.MODELS[model_name]["hf_id"])
            if records[0]["exp"] == "lang_jb_conc":
                records = use_ccs_reference(records, root)
            return M.eval_ccs(records, self._tokenizers[model_name])

        raise ValueError(self.metric)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metric", required=True, choices=C.METRICS)
    parser.add_argument("--models", nargs="+", default=C.MODEL_NAMES, choices=C.MODEL_NAMES)
    parser.add_argument("--exps", nargs="+", default=ALL_EXPS, choices=ALL_EXPS)
    parser.add_argument("--runs", nargs="+", default=["steering", "prompt_baseline"],
                        choices=["steering", "prompt_baseline"])
    parser.add_argument("--batch-size", type=int, default=None, help="Judge batch size.")
    parser.add_argument("--root", default=paths.DEFAULT_ROOT)
    args = parser.parse_args()

    evaluator = Evaluator(args.metric, args.batch_size)

    for model_name in args.models:
        for exp in args.exps:
            if args.metric not in C.exp_metrics(exp):
                continue
            for run in args.runs:
                records = load_records(args.root, run, model_name, exp, strict=False)
                if not records:
                    print(f"Skipping {run}/{model_name}/{exp}: no generations.")
                    continue
                print(f"\n=== {args.metric.upper()} | {run} | {model_name} | {exp} ===", flush=True)
                per_sample = evaluator(records, model_name, args.root)
                summary = save_metric(per_sample, args.root, run, model_name, exp, args.metric)
                print(summary[["target_lang", "condition", "score"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
