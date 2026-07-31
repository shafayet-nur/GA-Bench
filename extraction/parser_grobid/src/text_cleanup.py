from __future__ import annotations
import re
import unicodedata

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_NL_RE = re.compile(r"\n{3,}")

_DEHYPHEN_RE = re.compile(r"\b([A-Za-z]{3,})-\s+([a-z]{2,})\b")

_SYMBOL_REPLACEMENTS = {
    "1⁄4": "=",
    "¼": "=",
    "À": "-",
    "þ": "+",
    "∕": "/",
    "": "±",
    "": "°",
    "": "-",
    "‐": "-",
    "‑": "-",
    "‒": "-",
}

def normalize_math_symbols(text: str) -> str:

    if not text:
        return ""
    s = str(text)
    for bad, good in _SYMBOL_REPLACEMENTS.items():
        s = s.replace(bad, good)

    s = re.sub(r"(?<=[0-9)\]])\s*Â\s*(?=[0-9A-Za-z(])", " × ", s)
    s = re.sub(r"(?<=[A-Za-z])\s+Â\s*(?=[0-9(])", " × ", s)

    s = re.sub(r"\bAE\s+SEM\b", "± SEM", s)
    s = re.sub(r"(?<=\d)\s+AE\s+(?=\d)", " ± ", s)
    return s

def clean_text_artifacts(text: str) -> str:

    if not text:
        return ""
    s = normalize_math_symbols(str(text))
    s = unicodedata.normalize("NFKC", s)
    s = normalize_math_symbols(s)
    s = s.replace("\u00ad", "")
    s = s.replace("\ufffd", "")
    s = _CONTROL_RE.sub(" ", s)
    s = _DEHYPHEN_RE.sub(r"\1\2", s)

    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"([([{])\s+", r"\1", s)
    s = re.sub(r"\s+([)\]}])", r"\1", s)
    s = re.sub(r"\bFig\s*\.\s*", "Fig. ", s, flags=re.IGNORECASE)
    s = re.sub(r"\bEq\s*\.\s*", "Eq. ", s, flags=re.IGNORECASE)
    s = re.sub(r"\bTab\s*\.\s*", "Table ", s, flags=re.IGNORECASE)

    lines = [_MULTI_SPACE_RE.sub(" ", line).strip() for line in s.splitlines()]
    s = "\n".join(lines)
    s = _MULTI_NL_RE.sub("\n\n", s)
    return s.strip()

def clean_equation_text(text: str) -> str:

    if not text:
        return ""
    s = clean_text_artifacts(text)

    s = s.replace("•", " ")
    s = s.replace("−", "-")
    s = s.replace("–", "-")
    s = s.replace("—", "-")
    s = s.replace("Ã", "*")
    s = s.replace("ð", "(").replace("Þ", ")")
    s = s.replace("×", " × ")
    s = re.sub(r"\s*([=+\-*/^])\s*", r" \1 ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

_FRAGMENT_EQ_RE = re.compile(
    r"\[Equation\s+(?P<id>\d+)(?:,\s*low-confidence)?:\s*(?P<body>.*?)\]",
    re.DOTALL,
)

def _is_equation_fragment(body: str) -> bool:

    b = (body or "").strip()
    if not b:
        return True
    if re.fullmatch(r"[()\[\]{}\s,.;:]+", b):
        return True
    if re.fullmatch(r"\(?\s*\d+[A-Za-z]?\s*\)?", b):
        return True
    if len(b) <= 3 and not re.search(r"[A-Za-zα-ωΑ-Ωψλσ∂ημνκΩ]", b):
        return True
    return False

def merge_broken_equation_blocks(text: str) -> str:

    if not text:
        return ""
    out = []
    last = 0
    kept_any = False
    for m in _FRAGMENT_EQ_RE.finditer(text):
        body = m.group("body")
        out.append(text[last:m.start()])
        if _is_equation_fragment(body) and kept_any:

            out.append(" ")
        else:
            out.append(m.group(0))
            kept_any = True
        last = m.end()
    out.append(text[last:])
    return clean_text_artifacts("".join(out))

def equation_block(eq_id: int | str, formula_or_text: str, *, low_confidence: bool = False) -> str:

    formula = clean_equation_text(formula_or_text)
    if not formula:
        formula = "formula not recovered"
        low_confidence = True
    prefix = f"Equation {eq_id}, low-confidence" if low_confidence else f"Equation {eq_id}"
    return f"[{prefix}: {formula}]"
