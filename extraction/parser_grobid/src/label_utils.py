from __future__ import annotations
import re

_KIND_WORDS = {
    "figure": r"fig(?:ure|s|\.|\b)",
    "table":  r"tab(?:le|les|\.|\b)",
    "scheme": r"sch(?:eme|emes|\.|\b)",
}

_QUALIFIER_PREFIX = [
    (re.compile(r"\b(?:supplement(?:ary|al)?|suppl?|supp)\b", re.IGNORECASE), "S"),
    (re.compile(r"\bextended\s+data\b", re.IGNORECASE), "E"),
    (re.compile(r"\bappendix\b", re.IGNORECASE), "A"),
    (re.compile(r"\bonline\b", re.IGNORECASE), "S"),
]

_PREFIX_LETTERS = "SAE"

_NUM_TOKEN = re.compile(
    rf"(?P<prefix>[{_PREFIX_LETTERS}])?"
    r"(?P<sep>\.?)"
    r"(?P<number>\d{1,3})"
    r"(?P<panel>[a-z])?"
    r"\b",
    re.IGNORECASE,
)

def _kind_label_re(kind_word_pat: str) -> re.Pattern:
    return re.compile(
        r"(?P<qual>(?:supplement(?:ary|al)?|suppl?|supp|extended\s+data|appendix|online)\s+)?"
        rf"(?:{kind_word_pat})"
        r"\s*\.?\s*"
        rf"(?P<prefix>[{_PREFIX_LETTERS}])?"
        r"(?P<sep>\.?)"
        r"(?P<number>\d{1,3})"
        r"(?P<panel>[a-z])?"
        r"(?![0-9])",
        re.IGNORECASE,
    )

_FIGURE_LABEL_RE = _kind_label_re(_KIND_WORDS["figure"])
_TABLE_LABEL_RE = _kind_label_re(_KIND_WORDS["table"])
_SCHEME_LABEL_RE = _kind_label_re(_KIND_WORDS["scheme"])

_KIND_RES = [
    ("figure", _FIGURE_LABEL_RE),
    ("table", _TABLE_LABEL_RE),
    ("scheme", _SCHEME_LABEL_RE),
]

_ID_PREFIX = {"figure": "fig", "table": "table", "scheme": "scheme"}

_HEADER_NOISE = [
    re.compile(r"\b[A-Z][A-Za-z'’.\-]+\.\s+Long-term[^.]*\.\s+Am J Obstet Gynecol\s+\d{4}\.?", re.IGNORECASE),
    re.compile(r"\bAm J Obstet Gynecol\s+\d{4}\.?", re.IGNORECASE),
    re.compile(r"\bOriginal Research\b", re.IGNORECASE),
    re.compile(r"\bOBSTETRICS\b"),
    re.compile(r"\bajog\.org\b", re.IGNORECASE),
]

def _normalize_prefix(prefix: str | None, qualifier: str | None) -> str:

    if prefix:
        return prefix.upper()
    if qualifier:
        q = qualifier.strip()
        for pat, implied in _QUALIFIER_PREFIX:
            if pat.search(q):
                return implied
    return ""

def _make_record(kind: str, prefix: str, number: str, panel: str, sep: str = "") -> dict:
    prefix = (prefix or "").upper()
    number = number.lstrip("0") or "0"
    panel = (panel or "").lower()
    sep = "." if (sep and prefix) else ""

    id_prefix = _ID_PREFIX[kind]
    norm_kind = {"figure": "Figure", "table": "Table", "scheme": "Scheme"}[kind]

    if prefix:

        display_number = f"{prefix}{sep}{number}"
        if sep:
            asset = f"{id_prefix}_{prefix}_{number}"
        else:
            asset = f"{id_prefix}_{prefix}{number}"
        norm = f"{norm_kind} {display_number}"
    else:

        try:
            asset = f"{id_prefix}_{int(number):02d}"
        except ValueError:
            asset = f"{id_prefix}_{number}"
        display_number = number
        norm = f"{norm_kind} {display_number}"

    return {
        "kind": kind,
        "prefix": prefix,
        "number": number,
        "panel": panel,
        "separator": sep,
        "key": f"{kind}:{prefix}:{number}",
        "id": asset,
        "norm": norm,
        "normalized_label": display_number,
    }

def parse_label(text: str | None) -> dict | None:

    if not text:
        return None
    s = str(text).strip()
    if not s:
        return None

    best = None
    best_pos = None
    for kind, rx in _KIND_RES:
        m = rx.search(s)
        if m is None:
            continue
        pos = m.start()
        if best_pos is None or pos < best_pos:
            prefix = _normalize_prefix(m.group("prefix"), m.group("qual"))
            best = _make_record(kind, prefix, m.group("number"), m.group("panel") or "", m.groupdict().get("sep", ""))
            best_pos = pos
    return best

def canonical_key(text: str | None) -> str:

    rec = parse_label(text)
    return rec["key"] if rec else ""

_SPACED_TAGGED = r"T\s+[Aa]\s+[Gg]\s+[Gg]\s+[Ee]\s+[Dd]"

def _spaced_opt(word: str) -> str:
    return r"\s*".join(re.escape(c) for c in word)

_TAG_NAME = (
    r"(?:" + _spaced_opt("End") + r"|" + _spaced_opt("Start") + r"|H\s*\d|P\b|"
    + _spaced_opt("Table") + r"|" + _spaced_opt("Figure") + r"|"
    + _spaced_opt("Box") + r"|" + _spaced_opt("List") + r")"
)

_TAGGED_MARKER_RE = re.compile(
    r"(?:" + _SPACED_TAGGED + r")\s*(?:" + _TAG_NAME + r")?"
    r"|Tagged\s*" + _TAG_NAME,
    re.IGNORECASE,
)

def strip_tagged_markers(text: str | None) -> str:

    if not text:
        return text or ""
    out = _TAGGED_MARKER_RE.sub(" ", text)
    return re.sub(r"\s+", " ", out).strip()

_CROSSREF_CAPTION_RE = re.compile(
    r"^\s*(?:fig(?:ure)?|table|scheme)\s+[SAE]?\d+\b[^.]{0,40}\b"
    r"(?:available\s+online|in\s+the\s+(?:supplement|appendix)|supplementary)",
    re.IGNORECASE,
)

def is_crossref_caption(caption: str | None) -> bool:

    if not caption:
        return False
    return _CROSSREF_CAPTION_RE.match(caption.strip()) is not None

def asset_id(text: str | None, fallback_index: int | None = None) -> str:

    rec = parse_label(text)
    if rec:
        return rec["id"]
    if fallback_index is not None:
        return f"fig{fallback_index:02d}x"
    return ""

def find_caption_anchors(text: str | None) -> list[dict]:

    if not text:
        return []
    hits: list[dict] = []
    for kind, rx in _KIND_RES:
        for m in rx.finditer(text):
            prefix = _normalize_prefix(m.group("prefix"), m.group("qual"))
            rec = _make_record(kind, prefix, m.group("number"), m.group("panel") or "", m.groupdict().get("sep", ""))
            rec["start"] = m.start()
            rec["match"] = m.group(0)
            hits.append(rec)
    hits.sort(key=lambda r: r["start"])
    return hits

def _strip_header_noise(text: str) -> str:
    out = text
    for pat in _HEADER_NOISE:
        out = pat.sub(" ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out

def split_merged_caption(text: str | None) -> list[dict]:

    if not text:
        return []
    cleaned = _strip_header_noise(text)
    anchors = find_caption_anchors(cleaned)
    if not anchors:
        return []

    out: list[dict] = []
    for i, a in enumerate(anchors):
        start = a["start"]
        end = anchors[i + 1]["start"] if i + 1 < len(anchors) else len(cleaned)
        body = cleaned[start:end].strip(" .;:")
        out.append({
            "id": a["id"],
            "norm": a["norm"],
            "kind": a["kind"],
            "prefix": a["prefix"],
            "number": a["number"],
            "caption": body,
        })

    by_id: dict[str, dict] = {}
    for rec in out:
        prev = by_id.get(rec["id"])
        if prev is None or len(rec["caption"]) > len(prev["caption"]):
            by_id[rec["id"]] = rec

    seen = []
    result = []
    for rec in out:
        if rec["id"] not in seen:
            seen.append(rec["id"])
            result.append(by_id[rec["id"]])
    return result

if __name__ == "__main__":
    tests = [
        "Figure 3", "Fig. 3", "FIGURE 1", "FIGURE 4 LV function of SHRs",
        "Supplemental Figure S8", "SUPPLEMENTary Fig. S12", "Fig. S1",
        "Figure A1", "Fig. A.1", "Appendix Figure A.1", "Extended Data Figure 3",
        "Supplementary Figure 8", "Table 2", "Table S2", "Scheme 1",
        "Fig. 1a", "Figure 2 (continued)", "Feng",
    ]
    for t in tests:
        r = parse_label(t)
        print(f"{t!r:42} -> {r['id'] if r else None!s:10} key={r['key'] if r else '-'}")

    print()
    merged = ("Feng. Long-term cardiovascular protection by normotensive placental "
              "extracellular vesicles. Am J Obstet Gynecol 2024. ajog.org OBSTETRICS "
              "Original Research SUPPLEMENTAL FIGURE S7 Comparison of renal arteries "
              "resistive index. SUPPLEMENTAL FIGURE S8 Comparative analysis of the "
              "effect of normotensive placental EVs on glomerular hypertrophy.")
    print("split_merged_caption:")
    for rec in split_merged_caption(merged):
        print(f"  {rec['id']:8} {rec['caption'][:60]}...")
