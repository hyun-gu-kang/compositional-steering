# Compositional Multilingual and Behavioral Attribute Steering

[![arXiv](https://img.shields.io/badge/arXiv-none-b31b1b.svg)]()
(Paper link will appear soon!)

This repository contains the code for **"Compositional Multilingual and Behavioral Attribute Steering"**, to appear at **BlackBoxNLP @ EMNLP 2026**.

## Abstract

This study examines the compositionality of steering vectors for language and behavioral control in large language models. Focusing on language, jailbreak, and conciseness, we investigate whether additive, training-free composition of attribute steering vectors can preserve the intended steering effect of each attribute, across four instruction-tuned models from two model families and two size scales. We find that single-attribute steering is reliable for all three attributes, but only within an appropriate combination of intervention layer and steering strength, with abstract behaviors (jailbreak, conciseness) favoring middle layers and language favoring earlier layers. We show that additive composition of two attribute vectors succeeds in steering both attributes simultaneously when each is injected at its own best-performing layer, and that this partially extends to three simultaneously composed attributes, addressing an inconsistency left open by prior work on training-free composition. We further analyze the geometric properties of these steering vectors, finding that they are approximately orthogonal in the residual stream, consistent with their compositional behavior.

## Code

The repository is organized into three sets of experiments...

The corresponding Python scripts will be uploaded soon.

## Generations

All generated samples are available in the interactive `/viewer` for all tested models—`Llama-3.1-8B-Instruct`, `Llama-3.1-70B-Instruct`, `Qwen2.5-14B-Instruct`, and `Qwen2.5-32B-Instruct`—and all three behaviors: <span style="color:#365f8a">**ℒ** language</span> (steering from English to non-English languages), <span style="color:#4c9a63">**𝒥** jailbreak</span> (steering towards compliance with harmful instructions), and <span style="color:#d98b24">**𝒞** conciseness</span> (steering towards brief generations).

## Citation

(The bibtex will be updated soon!)
```bibtex

```
