# Data

`python scripts/download_data.py` creates the following files. They come from
other repositories and are not distributed with this one.

- `refusal_direction/{harmful_train,harmful_test,harmless_train}.json`:
  harmful/harmless instructions of Arditi et al. (2024),
  https://github.com/andyrdt/refusal_direction
- `ifeval_en.json`: the 541 instruction-stripped IFEval prompts (`model_output`
  field of `data/ifeval_wo_instructions.jsonl`) of Stolfo et al. (2025),
  https://github.com/microsoft/llm-steer-instruct

FLORES+ and CLaS-Bench are loaded from the Hugging Face Hub.
