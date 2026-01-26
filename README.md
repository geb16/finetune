# Fine-tuning (SFT) with LoRA on CPU - learning demo

This repo is a minimal, CPU-friendly fine-tuning demo. It trains a LoRA adapter on a small JSONL dataset so the model outputs strict JSON.

## What this demo does
- Uses a small seq2seq base model (`google/flan-t5-small`) for CPU training.
- Trains LoRA adapters only (fast, light, easy to move to GPU later).
- Produces JSON-only responses and validates them.

## Requirements
- Python 3.10+
- CPU only is fine. If CUDA is available, inference will use it automatically.

## Quick start (Windows PowerShell)
Run these commands from the repo root (`E:\AWS\finetune-lora-cpu`):

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install -e .
python -m ft.train_sft_lora
python -m ft.generate_preds
python -m ft.eval_json --path runs/latest/preds.jsonl
python -m ft.infer --prompt "Ticket: user cannot login, 2FA fails after phone change."
```

## Quick start (Linux/macOS)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
python -m ft.train_sft_lora
python -m ft.generate_preds
python -m ft.eval_json --path runs/latest/preds.jsonl
python -m ft.infer --prompt "Ticket: user cannot login, 2FA fails after phone change."
```

## Data format (no ambiguity)
Files:
- `data/sft_train.jsonl`
- `data/sft_valid.jsonl`

Each line must be a single JSON object with exactly these keys:
```
{"system": "...", "user": "...", "assistant": "..."}
```

Example line:
```
{"system":"You are a support incident summarizer. Output STRICT JSON only with keys: category, severity, summary, next_steps.","user":"Ticket: user cannot login, 2FA fails after phone change.","assistant":"{\"category\":\"auth\",\"severity\":\"high\",\"summary\":\"2FA fails after phone change\",\"next_steps\":[\"verify identity\",\"reset 2FA method\"]}"}
```

## Training
```
python -m ft.train_sft_lora
```

Outputs (created under `runs/latest`):
- `lora_adapter/` (LoRA weights)
- `tokenizer/`
- `manifest.json` (metadata: base model, lengths, counts)

If you want to keep multiple runs, edit `TrainConfig.out_dir` in `src/ft/config.py` before training.

## Generate predictions (validation set)
```
python -m ft.generate_preds
```

This writes `runs/latest/preds.jsonl`, one JSON object per line.

## Evaluate JSON validity
```
python -m ft.eval_json --path runs/latest/preds.jsonl
```

## Inference (single prompt)
```
python -m ft.infer --prompt "Ticket: user cannot login, 2FA fails after phone change."
```

## Troubleshooting
- Warning about `temperature` being ignored: expected when `do_sample=False`.
- If generation is not JSON: add more examples to the dataset or train for more epochs.
- Adapter size mismatch: make sure the base model and tokenizer come from the same run, or retrain.
