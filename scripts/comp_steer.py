"""Compositional steering: inject several attribute vectors in one forward pass.

    python scripts/comp_steer.py --model llama_8b
    python scripts/comp_steer.py --model llama_8b --exps lang_jb_conc

Each attribute vector is added at its own layer with its own strength
(Table 5, `config.COMP_SETTINGS`); vectors sharing a layer are summed.
Language steering is EN -> X for every target language X.

CCS reference for L+J+C (footnote 4): L+J steering of Llama-3.1-8B with 512
new tokens, used as the length baseline of all models:

    python scripts/comp_steer.py --model llama_8b --ccs-reference
"""

import argparse

from compsteer import config as C
from compsteer import paths
from compsteer.data import load_eval_prompts
from compsteer.model import generate, load_model
from compsteer.records import save_generations
from compsteer.steering import build_hooks, load_vector


def run(model, tokenizer, device, args, exp, run_name, max_new_tokens):
    prompts = load_eval_prompts(exp)
    settings = C.COMP_SETTINGS[args.model]
    attributes = C.exp_attributes(exp)

    print(f"[{args.model}] {exp}: unsteered baseline ({max_new_tokens} tokens)", flush=True)
    baselines = generate(model, tokenizer, prompts, device, max_new_tokens, args.batch_size, desc="baseline")

    # Behavioral vectors do not depend on the target language.
    fixed_terms = []
    for attribute in attributes:
        if attribute != "language":
            layer, alpha = settings[attribute]
            vec, _ = load_vector(args.root, args.model, attribute, layer)
            fixed_terms.append((layer, vec, alpha))

    for target_lang in C.exp_target_langs(exp):
        layer, alpha = settings["language"]
        lang_vec, _ = load_vector(args.root, args.model, "language", layer, target_lang)
        hooks = build_hooks([(layer, lang_vec, alpha)] + fixed_terms)

        outputs = generate(model, tokenizer, prompts, device, max_new_tokens, args.batch_size,
                           hooks=hooks, desc=f"{exp} en->{target_lang}")
        save_generations(
            paths.generation_path(args.root, run_name, args.model, exp, target_lang),
            metadata={
                "model": args.model,
                "exp": exp,
                "run": run_name,
                "source_lang": C.SOURCE_LANG,
                "target_lang": target_lang,
                "steering": {a: {"layer": settings[a][0], "alpha": settings[a][1]} for a in attributes},
                "max_new_tokens": max_new_tokens,
                "seed": C.SEED,
            },
            prompts=prompts,
            baselines=baselines,
            outputs=[[{"text": text}] for text in outputs],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=C.MODEL_NAMES)
    parser.add_argument("--exps", nargs="+", default=list(C.COMP_EXPS), choices=C.COMP_EXPS)
    parser.add_argument("--ccs-reference", action="store_true",
                        help="Generate the L+J reference outputs (512 tokens) used for L+J+C CCS.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--root", default=paths.DEFAULT_ROOT)
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model)

    if args.ccs_reference:
        run(model, tokenizer, device, args, C.CCS_REFERENCE_EXP, "ccs_reference",
            C.CCS_REFERENCE_MAX_NEW_TOKENS)
        return

    for exp in args.exps:
        run(model, tokenizer, device, args, exp, "steering", C.exp_max_new_tokens(exp))


if __name__ == "__main__":
    main()
