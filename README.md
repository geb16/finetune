# Fine-tuning (SFT) with LoRA on CPU — modern, enterprise-style repo

This repo trains a small instruction-following model adapter (LoRA) to produce strict JSON outputs.

## Why LoRA
LoRA trains a tiny adapter instead of all weights, reducing compute and making CPU training feasible.

## Setup (Windows PowerShell or Linux)
python -m venv .venv
# Windows:
`.venv\Scripts\activate`
# Linux/macOS:
`source .venv/bin/activate`

```
pip install -U pip
pip install -e .
```

## Data
- data/sft_train.jsonl
- data/sft_valid.jsonl
Each line contains {"system": "...", "user": "...", "assistant": "..."}.

## Train (CPU)
```
python -m ft.train_sft_lora
```

## Evaluate
```
python -m ft.eval_json --path runs/latest/preds.jsonl
```

## Inference
```
python -m ft.infer --prompt "Summarize this ticket: user can't login, 2FA failing after phone change."
```
