"""Single-attribute steering: sweep intervention layers and steering strengths.

    python scripts/single_steer.py --model llama_8b
    python scripts/single_steer.py --model qwen_14b --exps jailbreak --batch-size 32

For every layer in `config.MODELS[model]["layers"]` and every alpha in
`config.SINGLE_ALPHAS[family]`, the unit-norm vector is added to the residual
stream at that layer. Language steering is EN -> X for every target language.
"""

import argparse

from compsteer import config as C
from compsteer import paths
from compsteer.data import load_eval_prompts
from compsteer.model import generate, load_model
from compsteer.records import save_generations
from compsteer.steering import build_hooks, load_vector


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=C.MODEL_NAMES)
    parser.add_argument("--exps", nargs="+", default=list(C.SINGLE_EXPS), choices=C.SINGLE_EXPS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--root", default=paths.DEFAULT_ROOT)
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model)
    layers = C.MODELS[args.model]["layers"]
    alphas = C.single_alphas(args.model)

    for exp in args.exps:
        (attribute,) = C.exp_attributes(exp)
        prompts = load_eval_prompts(exp)
        max_new_tokens = C.exp_max_new_tokens(exp)
        baselines = generate(model, tokenizer, prompts, device, max_new_tokens, args.batch_size,
                             desc=f"{exp} baseline")

        for target_lang in C.exp_target_langs(exp):
            outputs = [[] for _ in prompts]
            for layer in layers:
                vec, _ = load_vector(args.root, args.model, attribute, layer,
                                     target_lang if attribute == "language" else None)
                for alpha in alphas:
                    texts = generate(model, tokenizer, prompts, device, max_new_tokens, args.batch_size,
                                     hooks=build_hooks([(layer, vec, alpha)]),
                                     desc=f"{exp} {target_lang} L{layer} a={alpha:g}")
                    for row, text in zip(outputs, texts):
                        row.append({"layer": layer, "alpha": alpha, "text": text})

            save_generations(
                paths.generation_path(args.root, "steering", args.model, exp, target_lang),
                metadata={
                    "model": args.model,
                    "exp": exp,
                    "run": "steering",
                    "source_lang": C.SOURCE_LANG,
                    "target_lang": target_lang,
                    "layers": layers,
                    "alphas": alphas,
                    "max_new_tokens": max_new_tokens,
                    "seed": C.SEED,
                },
                prompts=prompts,
                baselines=baselines,
                outputs=outputs,
            )


if __name__ == "__main__":
    main()
