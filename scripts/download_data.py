"""Download the third-party prompt data into data/ (not redistributed here).

    python scripts/download_data.py

- Harmful/harmless instruction splits of Arditi et al. (2024),
  https://github.com/andyrdt/refusal_direction
  -> data/refusal_direction/{harmful_train,harmful_test,harmless_train}.json
- Instruction-stripped IFEval prompts of Stolfo et al. (2025),
  https://github.com/microsoft/llm-steer-instruct
  -> data/ifeval_en.json (the 541 `model_output` fields as a JSON list)
"""

import argparse
import json
import os
import urllib.request

from compsteer import config as C

REFUSAL_URL = "https://raw.githubusercontent.com/andyrdt/refusal_direction/main/dataset/splits/{name}.json"
REFUSAL_SPLITS = ("harmful_train", "harmful_test", "harmless_train")
IFEVAL_URL = "https://raw.githubusercontent.com/microsoft/llm-steer-instruct/main/data/ifeval_wo_instructions.jsonl"


def download_refusal_splits(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name in REFUSAL_SPLITS:
        path = os.path.join(out_dir, f"{name}.json")
        urllib.request.urlretrieve(REFUSAL_URL.format(name=name), path)
        with open(path, encoding="utf-8") as f:
            print(f"Saved {path} ({len(json.load(f))} instructions)")


def download_ifeval(path, url=IFEVAL_URL):
    """Keep the non-empty `model_output` field, i.e. the IFEval prompt with
    its formatting instructions removed."""
    prompts = []
    with urllib.request.urlopen(url) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            value = json.loads(line).get("model_output")
            if isinstance(value, str) and value.strip():
                prompts.append(value.strip())

    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)
    print(f"Saved {path} ({len(prompts)} prompts)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refusal-dir", default=C.REFUSAL_SPLITS_DIR)
    parser.add_argument("--ifeval-path", default=C.IFEVAL_PATH)
    args = parser.parse_args()

    download_refusal_splits(args.refusal_dir)
    download_ifeval(args.ifeval_path)


if __name__ == "__main__":
    main()
