"""Reading and writing generation files.

Every generation file has the same schema:

{
  "metadata": {"model", "exp", "run", "source_lang", "target_lang",
               "max_new_tokens", ...},
  "results": [
    {"prompt_id": 0, "prompt": "...", "baseline": "<unsteered output>",
     "outputs": [{<condition fields>, "text": "..."}, ...]},
    ...
  ]
}

Condition fields are {"layer", "alpha"} for single-attribute steering,
{"prompt_variant", "suffix"} for prompt baselines, and empty for
compositional steering (its fixed configuration is stored in metadata).
"""

import json
import os

from . import config as C
from . import paths

# Columns identifying one evaluation condition (one point / box in a figure).
KEY_COLS = ["model", "exp", "run", "target_lang", "condition"]
# Descriptive columns that are constant within a condition.
DESC_COLS = ["layer", "alpha", "prompt_variant"]


def condition_label(output):
    if "layer" in output:
        return f"layer{int(output['layer'])}_alpha{float(output['alpha']):g}"
    if "prompt_variant" in output:
        return f"prompt{int(output['prompt_variant'])}"
    return "steered"


def save_generations(path, metadata, prompts, baselines, outputs):
    """outputs[i] is the list of condition dicts (with "text") for prompt i."""
    if not (len(prompts) == len(baselines) == len(outputs)):
        raise ValueError("prompts, baselines and outputs must have the same length.")
    payload = {
        "metadata": metadata,
        "results": [
            {"prompt_id": i, "prompt": p, "baseline": b, "outputs": o}
            for i, (p, b, o) in enumerate(zip(prompts, baselines, outputs))
        ],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved {path}")


def load_records(root, run, model, exp, strict=True):
    """Flatten all generation files of one (run, model, exp) into records."""
    records = []
    for target_lang in C.exp_target_langs(exp):
        path = paths.generation_path(root, run, model, exp, target_lang)
        if not os.path.isfile(path):
            if strict:
                raise FileNotFoundError(path)
            print(f"Missing: {path}")
            continue

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        for item in data["results"]:
            for output in item["outputs"]:
                records.append({
                    "model": model,
                    "exp": exp,
                    "run": run,
                    "target_lang": target_lang,
                    "condition": condition_label(output),
                    "layer": output.get("layer"),
                    "alpha": output.get("alpha"),
                    "prompt_variant": output.get("prompt_variant"),
                    "prompt_id": item["prompt_id"],
                    "question": item["prompt"],
                    "baseline": item["baseline"],
                    "response": str(output.get("text") or "").strip(),
                })
    return records
