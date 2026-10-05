# Compositional Multilingual and Behavioral Attribute Steering

Code for *Compositional Multilingual and Behavioral Attribute Steering*
(Hyun Gu Kang, Daniil Gurgurov, Tanja Baeumel, Josef van Genabith, Simon Ostermann).

We extract DiffMean steering vectors for **language** (EN → X), **jailbreak**, and
**conciseness**, steer each attribute in isolation across layers and strengths, and
compose two or three vectors additively, each injected at its own layer.

> **Content warning.** Jailbreak steering elicits responses to harmful instructions.

## Setup

```bash
pip install -e .
huggingface-cli login            # Llama 3.1 and FLORES+ are gated on the Hub
python scripts/download_data.py  # harmful/harmless splits and IFEval prompts
```

The data script fetches the harmful/harmless splits of Arditi et al. (2024) and
the instruction-stripped IFEval prompts of Stolfo et al. (2025) into `data/`;
they are not redistributed with this repository. Large models are split across
all visible GPUs with TransformerLens (`n_devices = torch.cuda.device_count()`).

## Reproducing the paper

Run steps 1–4 once per model (`llama_8b`, `llama_70b`, `qwen_14b`, `qwen_32b`).
All outputs go to `outputs/` (change with `--root`).

```bash
M=llama_8b

# 1. Mean activations of every contrastive condition (all layers)
python scripts/extract.py --model $M

# 2. Single-attribute steering: 4 layers x 5 strengths
python scripts/single_steer.py --model $M

# 3. Compositional steering: L+J, L+C, L+J+C with the Table 5 settings
python scripts/comp_steer.py --model $M

# 4. Prompt-based baselines (Table 3 suffixes)
python scripts/prompt_baseline.py --model $M
```

The L+J+C conciseness score uses one shared length reference (Llama-3.1-8B
under L+J steering with 512 new tokens, footnote 4):

```bash
python scripts/comp_steer.py --model llama_8b --ccs-reference
```

Then score everything and draw the figures:

```bash
# 5. Metrics (each runs only on the experiments that use it, Table 2)
python scripts/evaluate.py --metric lfs    # fastText LID
python scripts/evaluate.py --metric or     # Qwen3-8B judge
python scripts/evaluate.py --metric jbs    # Qwen3Guard-Gen-8B
python scripts/evaluate.py --metric ccs    # token-length ratio

# 6. Figures
python scripts/plot_single.py    # Figure 2
python scripts/plot_comp.py      # Figures 3 and 4
```

| Figure | Script | Output (`outputs/figures/`) |
|---|---|---|
| 2a–c Single-attribute steering | `plot_single.py` | `fig2a_language.pdf`, `fig2b_jailbreak.pdf`, `fig2c_conciseness.pdf` |
| 3a L+J, 3b L+C | `plot_comp.py` | `fig3a_lang_jb.pdf`, `fig3b_lang_conc.pdf` |
| 4 L+J+C | `plot_comp.py` | `fig4_lang_jb_conc.pdf` |

Each figure is saved together with a CSV of the plotted numbers.

## Method summary

| | |
|---|---|
| Vectors | DiffMean of last-input-token `hook_resid_post` activations, unit-normalized. Language: 260 FLORES+ dev sentences per language, μ(X) − μ(en). Jailbreak: 260 harmful vs. 260 harmless instructions, μ(harmless) − μ(harmful). Conciseness: IFEval prompts with vs. without a random shortness instruction. |
| Steering | h̃ₗ = hₗ + α·v̂ at every position; vectors sharing a layer are summed. |
| Prompts | 70 English CLaS-Bench prompts (language, conciseness) or 70 harmful test prompts (jailbreak). Greedy decoding, 512 new tokens with conciseness, 64 otherwise. |
| Metrics | LFS (target language with p ≥ 0.95), OR (0/1/2 → [0, 1]), JBS (Safe/Controversial/Unsafe → 0/0.5/1), CCS = 1 − min(L_steered / L_baseline, 1). |
| Score | Each metric is averaged over generations, then combined with the harmonic mean. |

All settings live in [`compsteer/config.py`](compsteer/config.py).

## Repository layout

```
compsteer/
  config.py      models, layers, strengths, languages, suffixes, judges
  data.py        FLORES+, CLaS-Bench, harmful/harmless, IFEval loaders
  model.py       TransformerLens loading (multi-GPU) and chat generation
  steering.py    mean activations, DiffMean vectors, steering hooks
  records.py     generation file schema
  metrics.py     LFS, OR, JBS, CCS
  aggregate.py   metric alignment and harmonic mean
  plotting.py    figure style and plotting functions
scripts/         one entry point per pipeline step (see above)
```

```
outputs/
  mean_acts/{model}/{attribute}/{condition}.pt
  generations/{steering,prompt_baseline,ccs_reference}/{model}/{exp}[_{lang}].json
  eval/{run}/{model}/{exp}/{metric}_{per_sample,summary}.csv
  figures/
```
