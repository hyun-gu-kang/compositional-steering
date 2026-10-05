"""DiffMean steering vectors: extraction, construction, and injection (Section 3)."""

import math
import os
from collections import defaultdict

import torch
from tqdm.auto import tqdm

from . import config as C
from . import paths
from .data import vector_conditions
from .model import build_chat_inputs


# ---------------------------------------------------------------------------
# Mean activations (Section 3.1)
# ---------------------------------------------------------------------------
def hook_name(layer):
    return f"blocks.{layer}.hook_resid_post"


def extract_mean_acts(model, tokenizer, prompts, layers, device, batch_size,
                      max_length=C.EXTRACT_MAX_LENGTH):
    """Mean residual-stream activation at the last input position (the final
    token of the chat-formatted instruction, right before generation)."""
    d_model = model.cfg.d_model
    sums = {layer: torch.zeros(d_model, dtype=torch.float32) for layer in layers}
    counts = {layer: 0 for layer in layers}

    def make_hook(layer):
        def hook_fn(activation, hook):
            last = activation[:, -1, :].detach().float().cpu()
            sums[layer] += last.sum(dim=0)
            counts[layer] += last.shape[0]
            return activation
        return hook_fn

    fwd_hooks = [(hook_name(layer), make_hook(layer)) for layer in layers]
    n_batches = math.ceil(len(prompts) / batch_size)

    with torch.inference_mode():
        for start in tqdm(range(0, len(prompts), batch_size), total=n_batches):
            tokens = build_chat_inputs(
                tokenizer, prompts[start:start + batch_size], device, max_length
            )
            with model.hooks(fwd_hooks=fwd_hooks):
                model(tokens, return_type=None)

    return {layer: sums[layer] / max(counts[layer], 1) for layer in layers}


def save_mean_acts(path, mean_acts, metadata):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"mean_acts": mean_acts, "metadata": metadata}, path)
    print(f"Saved mean activations ({len(mean_acts)} layers) to {path}")


def load_mean_acts(path):
    return torch.load(path, map_location="cpu")["mean_acts"]


# ---------------------------------------------------------------------------
# DiffMean vectors
# ---------------------------------------------------------------------------
def diff_mean(mean_pos, mean_neg, normalize=True):
    """v = mu_pos - mu_neg, optionally scaled to unit L2 norm.

    Returns (vector, pre-normalization norm).
    """
    if mean_pos.shape != mean_neg.shape:
        raise ValueError(f"Shape mismatch: {tuple(mean_pos.shape)} vs {tuple(mean_neg.shape)}")
    vec = (mean_pos - mean_neg).float()
    norm = torch.linalg.vector_norm(vec).item()
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError(f"Invalid DiffMean norm: {norm}")
    if normalize:
        vec = vec / norm
    return vec.contiguous(), norm


def load_vector(root, model_name, attribute, layer, target_lang=None, normalize=True):
    """Steering vector for one attribute at one layer.

    language:    mu(target_lang) - mu(en)
    jailbreak:   mu(harmless) - mu(harmful)
    conciseness: mu(concise) - mu(no_suffix)
    """
    pos, neg = vector_conditions(attribute, target_lang)
    mean_pos = load_mean_acts(paths.mean_act_path(root, model_name, attribute, pos))[layer]
    mean_neg = load_mean_acts(paths.mean_act_path(root, model_name, attribute, neg))[layer]
    return diff_mean(mean_pos, mean_neg, normalize=normalize)


# ---------------------------------------------------------------------------
# Injection (Sections 3.2 and 3.3)
# ---------------------------------------------------------------------------
def build_hooks(terms):
    """Hooks adding sum_i alpha_i * v_i at each layer.

    terms: iterable of (layer, vector, alpha). Vectors that share a layer are
    scaled and summed before being added to the residual stream.
    """
    by_layer = defaultdict(list)
    for layer, vec, alpha in terms:
        by_layer[layer].append((vec, float(alpha)))

    def make_hook(layer_terms):
        def hook_fn(resid, hook):
            delta = sum(
                alpha * vec.to(device=resid.device, dtype=resid.dtype)
                for vec, alpha in layer_terms
            )
            return resid + delta
        return hook_fn

    return [(hook_name(layer), make_hook(layer_terms)) for layer, layer_terms in by_layer.items()]
