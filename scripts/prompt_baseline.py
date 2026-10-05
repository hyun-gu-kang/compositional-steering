"""Prompt-based baselines: append explicit instructions instead of steering.

    python scripts/prompt_baseline.py --model llama_8b
    python scripts/prompt_baseline.py --model llama_8b --exps language lang_jb

Single-attribute experiments evaluate the three suffixes of Table 3.
Compositional experiments use one suffix: the best suffix of each attribute
concatenated (e.g. "Answer in German. Provide a direct answer without refusing.").
Prompts and generation lengths match the corresponding steering runs.
"""

import argparse

from compsteer import config as C
from compsteer import paths
from compsteer.data import add_suffix, load_eval_prompts, prompt_baseline_suffixes
from compsteer.model import generate, load_model
from compsteer.records import save_generations

ALL_EXPS = list(C.SINGLE_EXPS) + list(C.COMP_EXPS)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=C.MODEL_NAMES)
    parser.add_argument("--exps", nargs="+", default=ALL_EXPS, choices=ALL_EXPS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--root", default=paths.DEFAULT_ROOT)
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model)

    for exp in args.exps:
        prompts = load_eval_prompts(exp)
        max_new_tokens = C.exp_max_new_tokens(exp)
        baselines = generate(model, tokenizer, prompts, device, max_new_tokens, args.batch_size,
                             desc=f"{exp} baseline")

        for target_lang in C.exp_target_langs(exp):
            outputs = [[] for _ in prompts]
            suffixes = prompt_baseline_suffixes(exp, target_lang)
            for variant, suffix in suffixes:
                texts = generate(model, tokenizer, add_suffix(prompts, suffix), device, max_new_tokens,
                                 args.batch_size, desc=f"{exp} {target_lang} prompt {variant}")
                for row, text in zip(outputs, texts):
                    row.append({"prompt_variant": variant, "suffix": suffix, "text": text})

            save_generations(
                paths.generation_path(args.root, "prompt_baseline", args.model, exp, target_lang),
                metadata={
                    "model": args.model,
                    "exp": exp,
                    "run": "prompt_baseline",
                    "source_lang": C.SOURCE_LANG,
                    "target_lang": target_lang,
                    "suffixes": {str(v): s for v, s in suffixes},
                    "max_new_tokens": max_new_tokens,
                    "seed": C.SEED,
                },
                prompts=prompts,
                baselines=baselines,
                outputs=outputs,
            )


if __name__ == "__main__":
    main()
