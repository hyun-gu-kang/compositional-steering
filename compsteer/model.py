"""Model loading (TransformerLens, multi-GPU) and batched chat generation."""

import math

import torch
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utilities import get_device_for_block_index

from . import config as C


# ---------------------------------------------------------------------------
# Multi-GPU pipeline layout
# ---------------------------------------------------------------------------
# HookedTransformer.forward() moves activations to the device that
# get_device_for_block_index() expects for every block. With n_devices > 1 the
# default module placement can disagree with that mapping, so modules are moved
# explicitly to the expected devices.
def _module_devices(module):
    devices = {str(p.device) for p in module.parameters(recurse=True)}
    devices |= {str(b.device) for b in module.buffers(recurse=True)}
    return sorted(devices)


def force_tlens_pipeline_layout(model):
    cfg = model.cfg
    if cfg.device is None:
        cfg.device = "cuda:0"

    first_device = get_device_for_block_index(0, cfg)
    last_device = get_device_for_block_index(cfg.n_layers - 1, cfg)

    for name in ("embed", "hook_embed", "hook_tokens"):
        if hasattr(model, name):
            getattr(model, name).to(first_device)
    if getattr(cfg, "positional_embedding_type", None) != "rotary":
        for name in ("pos_embed", "hook_pos_embed"):
            if hasattr(model, name):
                getattr(model, name).to(first_device)

    for layer_idx, block in enumerate(model.blocks):
        block.to(get_device_for_block_index(layer_idx, cfg))

    for name in ("ln_final", "unembed"):
        if hasattr(model, name):
            getattr(model, name).to(last_device)
    return model


def verify_tlens_pipeline_layout(model):
    mismatches = []
    for layer_idx, block in enumerate(model.blocks):
        expected = str(get_device_for_block_index(layer_idx, model.cfg))
        actual = _module_devices(block)
        if actual != [expected]:
            mismatches.append((layer_idx, expected, actual))
    return mismatches


def _patch_tlens_module_placement():
    def _move_model_modules_to_device(self):
        force_tlens_pipeline_layout(self)

    HookedTransformer.move_model_modules_to_device = _move_model_modules_to_device


def load_model(model_name, dtype=torch.bfloat16):
    """Load a HookedTransformer split across all visible GPUs."""
    n_devices = torch.cuda.device_count()
    if n_devices == 0:
        raise RuntimeError("No CUDA device is available.")
    device = "cuda:0" if n_devices > 1 else "cuda"

    for idx in range(n_devices):
        props = torch.cuda.get_device_properties(idx)
        print(f"cuda:{idx}: {props.name}, {props.total_memory / 1024**3:.1f} GiB", flush=True)

    _patch_tlens_module_placement()
    model = HookedTransformer.from_pretrained_no_processing(
        C.MODELS[model_name]["hf_id"],
        device=device,
        dtype=dtype,
        n_devices=n_devices,
    )
    model = force_tlens_pipeline_layout(model)
    mismatches = verify_tlens_pipeline_layout(model)
    if mismatches:
        raise RuntimeError(f"Inconsistent TransformerLens device layout: {mismatches[:5]}")

    model.eval()
    tokenizer = model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer, device


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def build_chat_inputs(tokenizer, prompts, device, max_length=C.EXTRACT_MAX_LENGTH):
    """Left-padded input ids of single-turn chat prompts ending in the
    assistant generation header."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for prompt in prompts
    ]
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=max_length is not None,
        max_length=max_length,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return inputs["input_ids"].to(device)


def _generate_batch(model, tokenizer, prompts, device, max_new_tokens, hooks):
    input_ids = build_chat_inputs(tokenizer, prompts, device)
    kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        verbose=False,
    )
    with torch.inference_mode():
        if hooks:
            with model.hooks(fwd_hooks=hooks):
                output = model.generate(input_ids, **kwargs)
        else:
            output = model.generate(input_ids, **kwargs)
    new_tokens = output[:, input_ids.shape[1]:]
    return [text.strip() for text in tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]


def generate(model, tokenizer, prompts, device, max_new_tokens, batch_size, hooks=None, desc=None):
    """Greedy decoding for a list of prompts, optionally with steering hooks."""
    outputs = []
    n_batches = math.ceil(len(prompts) / batch_size)
    for start in tqdm(range(0, len(prompts), batch_size), total=n_batches, desc=desc):
        outputs.extend(
            _generate_batch(
                model, tokenizer, prompts[start:start + batch_size], device, max_new_tokens, hooks
            )
        )
        torch.cuda.empty_cache()
    return outputs
