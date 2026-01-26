import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from ft.config import TrainConfig, load_manifest
from ft.data import build_prompt

DATA = "data/sft_valid.jsonl"
OUT = "runs/latest/preds.jsonl"


def main():
    run_dir = Path("runs/latest")
    tokenizer = AutoTokenizer.from_pretrained(run_dir / "tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    manifest = load_manifest(run_dir)
    base_model = manifest.get("base_model", TrainConfig().base_model)
    base = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base, run_dir / "lora_adapter")
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    run_dir.mkdir(exist_ok=True)

    with open(DATA, "r", encoding="utf-8") as f, open(OUT, "w", encoding="utf-8") as out:
        for line in f:
            row = json.loads(line)
            prompt = build_prompt(row["system"], row["user"])
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.inference_mode():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=120,
                    do_sample=False,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

            text = tokenizer.decode(gen[0], skip_special_tokens=True)
            answer = text.strip()
            if answer.startswith("### Assistant:"):
                answer = answer.split("### Assistant:", 1)[-1].strip()
            out.write(answer + "\n")

if __name__ == "__main__":
    main()
