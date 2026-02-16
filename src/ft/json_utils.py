import json
import re
from typing import Optional

SMART_QUOTE_MAP = {
    "\u201c": '"',
    "\u201d": '"',
    "\u2018": "'",
    "\u2019": "'",
}


def _normalize_quotes(text: str) -> str:
    for src, dst in SMART_QUOTE_MAP.items():
        text = text.replace(src, dst)
    return text


def repair_json(text: str) -> Optional[str]:
    cleaned = text.strip()
    if not cleaned:
        return None

    cleaned = _normalize_quotes(cleaned)

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    if "### Assistant:" in cleaned:
        cleaned = cleaned.split("### Assistant:", 1)[-1].strip()

    if "{" in cleaned and "}" in cleaned:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        cleaned = cleaned[start : end + 1]
    else:
        cleaned = "{" + cleaned + "}"

    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

    try:
        json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return cleaned
