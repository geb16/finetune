import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def build_prompt(system: str, user: str) -> str:
    """
    Enterprise pattern:
    - Repeat the SYSTEM constraint in every sample (helps generalize when dataset is small).
    - Keep prompts portable across model families.
    """
    system = system.strip()
    user = user.strip()
    return f"### System:\n{system}\n\n### User:\n{user}\n\n### Assistant:\n"


def format_chat_example(ex: Dict) -> Tuple[str, str]:
    prompt = build_prompt(ex["system"], ex["user"])
    assistant = ex["assistant"].strip()
    return prompt, assistant
