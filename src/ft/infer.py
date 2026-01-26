from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from ft.config import TrainConfig, load_manifest
from ft.data import build_prompt


SYSTEM_PROMPT = (
    "You are a support incident summarizer. "
    "Output STRICT JSON only with keys: category, severity, summary, next_steps."
)


def _resolve_base_model(run_dir: Path) -> str:
    manifest = load_manifest(run_dir)
    return manifest.get("base_model", TrainConfig().base_model)


def _clean_answer(text: str) -> str:
    text = text.strip()
    if text.startswith("### Assistant:"):
        return text.split("### Assistant:", 1)[-1].strip()
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--run_dir", default="runs/latest")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    tokenizer = AutoTokenizer.from_pretrained(run_dir / "tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = _resolve_base_model(run_dir)
    base = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base, run_dir / "lora_adapter")
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    prompt = build_prompt(SYSTEM_PROMPT, args.prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.0,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    answer = _clean_answer(text)
    print(answer)

if __name__ == "__main__":
    main()
