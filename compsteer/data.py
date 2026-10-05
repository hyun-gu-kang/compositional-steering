"""Prompt datasets for vector extraction, steering, and prompt baselines."""

import json
import os
import random

from . import config as C


# ---------------------------------------------------------------------------
# Raw datasets
# ---------------------------------------------------------------------------
def load_flores(lang, n=C.FLORES_SAMPLES, split=C.FLORES_SPLIT, seed=C.SEED):
    """Sample `n` FLORES+ sentences for one language.

    FLORES+ is a gated dataset on the Hugging Face Hub; accept its terms first.
    """
    from datasets import load_dataset

    dataset = load_dataset("openlanguagedata/flores_plus", C.FLORES_CODES[lang], split=split)
    texts = [row["text"] for row in dataset]
    return random.Random(seed).sample(texts, n)


def load_refusal_split(harmtype, split, splits_dir=C.REFUSAL_SPLITS_DIR):
    """Instructions from the harmful/harmless splits of Arditi et al. (2024).

    Expects `{splits_dir}/{harmtype}_{split}.json` as distributed in
    https://github.com/andyrdt/refusal_direction (dataset/splits/).
    """
    path = os.path.join(splits_dir, f"{harmtype}_{split}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    with open(path, encoding="utf-8") as f:
        return [item["instruction"] for item in json.load(f)]


def load_clas(lang=C.SOURCE_LANG, n=C.EVAL_SAMPLE_SIZE):
    """The first `n` CLaS-Bench questions in `lang` (deterministic stream order)."""
    from datasets import load_dataset

    dataset = load_dataset("DGurgurov/CLaS-Bench", split="train", streaming=True)
    questions = []
    for row in dataset:
        if row["language_code"] != lang:
            continue
        questions.append(row["question"].strip())
        if len(questions) >= n:
            break
    if len(questions) < n:
        raise ValueError(f"Only {len(questions)} CLaS-Bench prompts found for '{lang}'.")
    return questions


def load_ifeval(path=C.IFEVAL_PATH):
    """Instruction-stripped IFEval prompts (Zhou et al., 2023; Stolfo et al., 2025)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found. Run `python scripts/download_data.py` first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Suffixes
# ---------------------------------------------------------------------------
def add_suffix(prompts, suffix):
    return [f"{prompt.rstrip()} {suffix.strip()}" for prompt in prompts]


def add_random_suffix(prompts, choices, seed=C.SEED):
    rng = random.Random(seed)
    return [f"{prompt.rstrip()} {rng.choice(choices).strip()}" for prompt in prompts]


# ---------------------------------------------------------------------------
# Contrastive sets for vector extraction (Section 4.1)
# ---------------------------------------------------------------------------
def extraction_sets(attribute):
    """Return {condition: prompts} for the two (or more) contrastive conditions.

    language:    FLORES+ sentences per language; vector = mean(X) - mean(en)
    jailbreak:   harmful vs. harmless instructions; vector = harmless - harmful
    conciseness: IFEval prompts without/with a shortness instruction;
                 vector = concise - no_suffix
    """
    if attribute == "language":
        return {lang: load_flores(lang) for lang in C.EXTRACTION_LANGS}

    if attribute == "jailbreak":
        harmful = load_refusal_split("harmful", "train")
        harmless = load_refusal_split("harmless", "train")
        harmless = random.Random(C.SEED).sample(harmless, min(C.HARMLESS_SAMPLES, len(harmless)))
        return {"harmful": harmful, "harmless": harmless}

    if attribute == "conciseness":
        prompts = load_ifeval()
        return {
            "no_suffix": prompts,
            "concise": add_random_suffix(prompts, C.CONCISE_INSTRUCTIONS),
        }

    raise ValueError(f"Unknown attribute: {attribute}")


def vector_conditions(attribute, target_lang=None):
    """(positive, negative) extraction conditions defining each steering vector."""
    if attribute == "language":
        if target_lang is None:
            raise ValueError("target_lang is required for the language vector.")
        return target_lang, C.SOURCE_LANG
    if attribute == "jailbreak":
        return "harmless", "harmful"
    if attribute == "conciseness":
        return "concise", "no_suffix"
    raise ValueError(f"Unknown attribute: {attribute}")


# ---------------------------------------------------------------------------
# Evaluation prompts (Section 4.1)
# ---------------------------------------------------------------------------
def load_eval_prompts(exp):
    """70 English CLaS-Bench prompts, or 70 harmful test prompts for jailbreak."""
    if C.exp_prompt_set(exp) == "harmful":
        harmful = load_refusal_split("harmful", "test")
        return random.Random(C.SEED).sample(harmful, C.EVAL_SAMPLE_SIZE)
    return load_clas(C.SOURCE_LANG, C.EVAL_SAMPLE_SIZE)


def prompt_baseline_suffixes(exp, target_lang):
    """Instruction suffixes used as prompt-based baselines (Table 3).

    Single-attribute baselines evaluate all three suffixes. Compositional
    baselines use only the concatenation of the best suffix per attribute.
    Returns a list of (variant_id, suffix).
    """
    attributes = C.exp_attributes(exp)
    language_name = C.LANG_NAMES[target_lang]

    if len(attributes) == 1:
        templates = C.PROMPT_SUFFIXES[attributes[0]]
        return [(i + 1, t.format(language_name)) for i, t in enumerate(templates)]

    parts = [
        C.PROMPT_SUFFIXES[attr][C.BEST_SUFFIX_INDEX[attr]].format(language_name)
        for attr in attributes
    ]
    return [(1, " ".join(parts))]
