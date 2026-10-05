"""Extract mean last-token activations for every contrastive condition.

    python scripts/extract.py --model llama_8b
    python scripts/extract.py --model qwen_14b --attributes language --batch-size 64

Saves outputs/mean_acts/{model}/{attribute}/{condition}.pt with the mean
activation at every layer. Steering vectors are built from these files.
"""

import argparse
import gc

import torch

from compsteer import config as C
from compsteer import paths
from compsteer.data import extraction_sets
from compsteer.model import load_model
from compsteer.steering import extract_mean_acts, save_mean_acts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=C.MODEL_NAMES)
    parser.add_argument("--attributes", nargs="+", default=list(C.ATTRIBUTES), choices=C.ATTRIBUTES)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--root", default=paths.DEFAULT_ROOT)
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model)
    layers = list(range(model.cfg.n_layers))

    for attribute in args.attributes:
        for condition, prompts in extraction_sets(attribute).items():
            print(f"[{args.model}] {attribute}/{condition}: {len(prompts)} prompts", flush=True)
            mean_acts = extract_mean_acts(model, tokenizer, prompts, layers, device, args.batch_size)
            save_mean_acts(
                paths.mean_act_path(args.root, args.model, attribute, condition),
                mean_acts,
                metadata={
                    "model": args.model,
                    "attribute": attribute,
                    "condition": condition,
                    "n_prompts": len(prompts),
                    "position": "last input token",
                    "hook": "hook_resid_post",
                },
            )
            del mean_acts
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
