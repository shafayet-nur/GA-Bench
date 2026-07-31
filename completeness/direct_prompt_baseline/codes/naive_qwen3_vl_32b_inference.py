#!/usr/bin/env python3
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\


import argparse
import asyncio
import base64
import csv
import json
import io
import mimetypes
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI
from PIL import Image


MODEL_NAME = "qwen3_vl_32b"
SERVED_MODEL_NAME = "qwen3_vl_32b"
MODEL_PATH = "/path/to/research_workspace/models/awq/Qwen3-VL-32B-Instruct-AWQ"
VLLM_URL = "http://localhost:8000"


USE_SYSTEM_ROLE = True

PROJECT_BASE = Path("/path/to/research_workspace/sci_ga_paper1/task1_completeness_awq")
NAIVE_ROOT = PROJECT_BASE / "baselines" / "naive_vlm_updated"


COMPLETE_DOIS_FILE = PROJECT_BASE / "complete_dois.txt"


GA_IMAGE_CSV = Path(
    "/path/to/research_workspace/sci_ga_paper1/task2_readability/"
    "output/stage1_preprocessing/index/stage1_ga_index.csv"
)
GA_CSV_DOI_COL = "paper_id"
GA_CSV_PATH_COL = "ga_path"


VARIANT = "A"
SYSTEM_PROMPT_FILE: Path = Path("/dev/null")
USER_PROMPT_DIR: Path = Path("/dev/null")
USER_PROMPT_SUFFIX = ""
FIGURE_PATHS_CSV: Path = Path("/dev/null")
OUTPUT_ROOT: Path = Path("/dev/null")
JUDGMENT_DIR: Path = Path("/dev/null")
RAW_OUTPUT_DIR: Path = Path("/dev/null")
ERROR_DIR: Path = Path("/dev/null")
LOG_DIR: Path = Path("/dev/null")
REPORT_DIR: Path = Path("/dev/null")
CHECKPOINT_DIR: Path = Path("/dev/null")

JUDGMENT_SUFFIX = ""
RAW_OUTPUT_SUFFIX = f"_raw_output_{MODEL_NAME}.txt"
CLEANED_OUTPUT_SUFFIX = f"_cleaned_output_{MODEL_NAME}.txt"
INVALID_JUDGMENT_SUFFIX = f"_invalid_judgment_{MODEL_NAME}.json"


RESULTS_CSV: Path = Path("/dev/null")
SUMMARY_JSON: Path = Path("/dev/null")
SUMMARY_TXT: Path = Path("/dev/null")
CHECKPOINT_JSONL: Path = Path("/dev/null")

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024
DEFAULT_CONCURRENCY = 1
DEFAULT_TIMEOUT = 600.0


REPETITION_PENALTY = 1.05
MAX_MODEL_LEN = 57344

SECTIONS = ["introduction", "methods", "results", "discussion"]


MAX_FIGURES_PER_PAPER = 12


MAX_IMAGE_DIM = 640

CSV_FIELDS = [
    "model", "served_model_name", "variant", "doi", "doi_safe",
    "user_prompt_file", "ga_image_path", "judgment_file",
    "status", "error_message",
    "paper_text_found", "text_chars",
    "ga_image_found", "ga_image_bytes",
    "figures_available", "figures_sent", "figures_dropped", "figure_bytes",
    "input_tokens", "output_tokens", "inference_time", "tokens_per_second",
    "raw_output_length", "had_think_tags", "had_code_fences",
    "parse_mode", "repaired", "level_mismatch_corrected", "reported_level",
    "present_introduction", "present_methods", "present_results", "present_discussion",
    "level",
]


@dataclass
class WorkItem:
    doi: str
    doi_safe: str
    ga_image_path: str
    user_prompt_path: Path
    figure_paths: List[str] = field(default_factory=list)


try:
    from json_repair import repair_json
except ImportError:
    repair_json = None


def _input_tokens_from_error(msg: str) -> Optional[int]:
    m = re.search(r"has (\d+) input tokens", msg or "")
    return int(m.group(1)) if m else None


def _is_context_overflow(msg: str) -> bool:
    m = (msg or "").lower()
    return (
        "maximum context length" in m
        or "longer than the maximum model length" in m
        or _input_tokens_from_error(msg) is not None
    )


async def create_with_context_retry(client, *, model, messages, temperature, max_tokens, extra_body):
\

    try:
        return await client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            max_tokens=max_tokens, extra_body=extra_body,
        )
    except Exception as exc:
        in_tok = _input_tokens_from_error(str(exc))
        if in_tok is None:
            raise
        retry_max = max(256, MAX_MODEL_LEN - in_tok - 64)
        if retry_max >= max_tokens:
            raise
        return await client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            max_tokens=retry_max, extra_body=extra_body,
        )


def apply_variant(variant: str) -> None:

    global VARIANT, SYSTEM_PROMPT_FILE, USER_PROMPT_DIR, USER_PROMPT_SUFFIX
    global FIGURE_PATHS_CSV, OUTPUT_ROOT, JUDGMENT_DIR, RAW_OUTPUT_DIR
    global ERROR_DIR, LOG_DIR, REPORT_DIR, CHECKPOINT_DIR, JUDGMENT_SUFFIX
    global RESULTS_CSV, SUMMARY_JSON, SUMMARY_TXT, CHECKPOINT_JSONL

    VARIANT = variant
    variant_dir = NAIVE_ROOT / f"variant_{variant}"

    SYSTEM_PROMPT_FILE = NAIVE_ROOT / "system_prompts" / f"naive_variant{variant}_system_prompt.txt"
    USER_PROMPT_DIR = variant_dir / "user_prompts"
    USER_PROMPT_SUFFIX = f"_naive_variant{variant}_user_prompt.txt"
    FIGURE_PATHS_CSV = variant_dir / "figure_paths.csv"

    OUTPUT_ROOT = variant_dir / MODEL_NAME
    JUDGMENT_DIR = OUTPUT_ROOT / "judgments"
    RAW_OUTPUT_DIR = OUTPUT_ROOT / "raw_outputs"
    ERROR_DIR = OUTPUT_ROOT / "errors"
    LOG_DIR = OUTPUT_ROOT / "logs"
    REPORT_DIR = OUTPUT_ROOT / "reports"
    CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"

    JUDGMENT_SUFFIX = f"_naivejudge_variant{variant}_{MODEL_NAME}.json"

    RESULTS_CSV = REPORT_DIR / f"naive_{MODEL_NAME}_variant{variant}_results.csv"
    SUMMARY_JSON = REPORT_DIR / f"naive_{MODEL_NAME}_variant{variant}_summary.json"
    SUMMARY_TXT = REPORT_DIR / f"naive_{MODEL_NAME}_variant{variant}_summary.txt"
    CHECKPOINT_JSONL = CHECKPOINT_DIR / "processed_prompts.jsonl"


def apply_run_tag(run_tag: Optional[str]) -> None:

    global RESULTS_CSV, SUMMARY_JSON, SUMMARY_TXT, CHECKPOINT_JSONL
    if not run_tag:
        return
    tag = f"_{run_tag}"
    RESULTS_CSV = REPORT_DIR / f"naive_{MODEL_NAME}_variant{VARIANT}_results{tag}.csv"
    SUMMARY_JSON = REPORT_DIR / f"naive_{MODEL_NAME}_variant{VARIANT}_summary{tag}.json"
    SUMMARY_TXT = REPORT_DIR / f"naive_{MODEL_NAME}_variant{VARIANT}_summary{tag}.txt"
    CHECKPOINT_JSONL = CHECKPOINT_DIR / f"processed_prompts{tag}.jsonl"


def make_dirs() -> None:
    for path in [JUDGMENT_DIR, RAW_OUTPUT_DIR, ERROR_DIR, LOG_DIR, REPORT_DIR, CHECKPOINT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", errors="replace") as f:
        f.write(text)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, ensure_ascii=False)
    with open(path, "w", encoding="utf-8", errors="replace") as f:
        f.write(text)


def load_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def doi_to_safe_id(doi: str) -> str:
\
\
\

    doi = (doi or "").strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.I)
    doi = doi.strip()

    safe = re.sub(r"[^A-Za-z0-9]+", "_", doi)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe


def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def extract_json_candidate(text: str) -> str:
    cleaned = strip_code_fences(strip_think_tags(text))
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first != -1 and last != -1 and last > first:
        return cleaned[first:last + 1].strip()
    return cleaned


def get_gpu_info() -> Dict[str, Any]:
    info = {"gpu_model_name": "unknown", "gpu_count": 0, "gpu_vram_total_mb": 0, "gpu_vram_allocated_mb": 0}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().splitlines()
            names, total, used = [], 0, 0
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    names.append(parts[0]); total += int(float(parts[1])); used += int(float(parts[2]))
            info["gpu_model_name"] = names[0] if names else "unknown"
            info["gpu_count"] = len(lines)
            info["gpu_vram_total_mb"] = total
            info["gpu_vram_allocated_mb"] = used
    except Exception as exc:
        info["gpu_error"] = str(exc)
    return info


def load_complete_dois() -> List[str]:
    if not COMPLETE_DOIS_FILE.exists():
        raise FileNotFoundError(f"Complete-DOIs file not found: {COMPLETE_DOIS_FILE}")
    dois = []
    with open(COMPLETE_DOIS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                dois.append(s)
    return dois


def load_ga_image_map() -> Dict[str, str]:
\

    if not GA_IMAGE_CSV.exists():
        raise FileNotFoundError(f"GA image index CSV not found: {GA_IMAGE_CSV}")
    mapping: Dict[str, str] = {}
    with open(GA_IMAGE_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        for need in (GA_CSV_DOI_COL, GA_CSV_PATH_COL):
            if need not in cols:
                raise ValueError(f"GA image CSV missing column '{need}'. Found: {sorted(cols)}")
        for row in reader:
            doi = (row.get(GA_CSV_DOI_COL) or "").strip()
            path = (row.get(GA_CSV_PATH_COL) or "").strip()
            if not doi or not path:
                continue
            mapping[doi_to_safe_id(doi)] = path
    return mapping


def load_figure_map() -> Dict[str, List[str]]:
\
\
\
\
\

    if VARIANT != "B":
        return {}
    if not FIGURE_PATHS_CSV.exists():
        raise FileNotFoundError(f"Figure paths CSV not found: {FIGURE_PATHS_CSV}")

    mapping: Dict[str, List[str]] = {}
    with open(FIGURE_PATHS_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        for need in ("safe_doi", "figure_id_to_path_json"):
            if need not in cols:
                raise ValueError(f"Figure paths CSV missing column '{need}'. Found: {sorted(cols)}")
        for row in reader:
            safe = (row.get("safe_doi") or "").strip()
            blob = (row.get("figure_id_to_path_json") or "").strip()
            if not safe or not blob:
                continue
            try:
                fig_map = json.loads(blob)
            except Exception:
                continue
            if not isinstance(fig_map, dict):
                continue

            paths = [fig_map[k] for k in sorted(fig_map.keys())]
            mapping[safe] = paths[:MAX_FIGURES_PER_PAPER]
    return mapping


def load_work_items(limit: Optional[int], start_index: Optional[int], end_index: Optional[int]) -> List[WorkItem]:
    dois = load_complete_dois()
    ga_map = load_ga_image_map()
    fig_map = load_figure_map()

    items: List[WorkItem] = []
    for doi in dois:
        safe = doi_to_safe_id(doi)
        items.append(
            WorkItem(
                doi=doi,
                doi_safe=safe,
                ga_image_path=ga_map.get(safe, ""),
                user_prompt_path=USER_PROMPT_DIR / f"{safe}{USER_PROMPT_SUFFIX}",
                figure_paths=fig_map.get(safe, []),
            )
        )


    items.sort(key=lambda x: x.doi_safe)

    if start_index is not None or end_index is not None:
        start = start_index or 0
        end = end_index if end_index is not None else len(items)
        items = items[start:end]
    if limit is not None:
        items = items[:limit]
    return items


def guess_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if mime and mime.startswith("image/"):
        return mime
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


def _mime_from_bytes(data: bytes) -> str:

    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    return "image/png"


def downscale_image_bytes(data: bytes, max_dim: int = MAX_IMAGE_DIM) -> Tuple[bytes, str]:
\
\
\
\

    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            w, h = im.size
            longest = max(w, h)
            if longest > max_dim:
                scale = max_dim / float(longest)
                new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
                im = im.resize(new_size, Image.LANCZOS)
            out = io.BytesIO()
            im.save(out, format="PNG")
            return out.getvalue(), "image/png"
    except Exception:
        return data, _mime_from_bytes(data)


def load_image_block(image_path: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:

    stats = {"found": False, "bytes": 0, "error": ""}
    if not image_path:
        stats["error"] = "no image path"
        return None, stats
    p = Path(image_path)
    if not p.exists():
        stats["error"] = f"image not found: {image_path}"
        return None, stats
    if not p.is_file():
        stats["error"] = f"image is not a regular file: {image_path}"
        return None, stats
    try:
        data = p.read_bytes()
        if not data:
            stats["error"] = "image is empty"
            return None, stats


        data, mime = downscale_image_bytes(data)
        encoded = base64.b64encode(data).decode("ascii")
        block = {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}
        stats["found"] = True
        stats["bytes"] = len(data)
        return block, stats
    except Exception as exc:
        stats["error"] = f"image load error: {exc}"
        return None, stats


def load_figure_blocks(figure_paths: List[str]) -> Tuple[List[Dict[str, Any]], int]:
\
\

    blocks: List[Dict[str, Any]] = []
    total_bytes = 0
    for path in figure_paths[:MAX_FIGURES_PER_PAPER]:
        block, stats = load_image_block(path)
        if block is None:
            print(f"[WARN] figure skipped: {stats['error']}", flush=True)
            continue
        blocks.append(block)
        total_bytes += stats["bytes"]
    return blocks, total_bytes


def build_messages(system_prompt: str, user_content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def build_user_content(
    paper_text: str,
    figure_blocks: List[Dict[str, Any]],
    ga_block: Dict[str, Any],
) -> List[Dict[str, Any]]:
\
\

    content: List[Dict[str, Any]] = [{"type": "text", "text": paper_text}]
    content.extend(figure_blocks)
    content.append({"type": "text", "text": "Graphical abstract image:"})
    content.append(ga_block)
    return content


def parse_judgment_json(cleaned: str):
\
\

    try:
        return json.loads(cleaned), "raw"
    except json.JSONDecodeError:
        pass
    if repair_json is not None:
        try:
            obj = repair_json(cleaned, return_objects=True)
        except Exception:
            obj = None
        if isinstance(obj, (dict, list)) and obj:
            return obj, "repaired"
    return None, "failed"


_TRUE_STRINGS = {"true", "yes", "present", "1"}
_FALSE_STRINGS = {"false", "no", "absent", "0"}


def coerce_judgment_schema(jd: Any) -> Tuple[Any, int]:
\
\

    n = 0
    if not isinstance(jd, dict):
        return jd, n

    cp = jd.get("components_present")
    if isinstance(cp, dict):
        for sec in SECTIONS:
            if sec not in cp:
                continue
            v = cp[sec]
            if isinstance(v, bool):
                continue
            if isinstance(v, str):
                s = v.strip().lower()
                if s in _TRUE_STRINGS:
                    cp[sec] = True; n += 1
                elif s in _FALSE_STRINGS:
                    cp[sec] = False; n += 1
            elif isinstance(v, (int, float)) and v in (0, 1):
                cp[sec] = bool(v); n += 1

    if "level" in jd and not isinstance(jd["level"], bool):
        v = jd["level"]
        if isinstance(v, str) and v.strip().isdigit():
            jd["level"] = int(v.strip()); n += 1
        elif isinstance(v, float) and float(v).is_integer():
            jd["level"] = int(v); n += 1

    return jd, n


def validate_judgment(jd: Any) -> Tuple[bool, str]:
    if not isinstance(jd, dict):
        return False, "judgment root is not a JSON object"

    cp = jd.get("components_present")
    if not isinstance(cp, dict):
        return False, "components_present is not an object"
    for sec in SECTIONS:
        if sec not in cp:
            return False, f"components_present missing section: {sec}"
        if not isinstance(cp[sec], bool):
            return False, f"components_present[{sec}] is not a boolean: {cp[sec]!r}"

    if "level" not in jd:
        return False, "missing level"
    if not isinstance(jd["level"], int) or isinstance(jd["level"], bool):
        return False, f"level is not an integer: {jd['level']!r}"
    if jd["level"] < 0 or jd["level"] > 4:
        return False, f"level out of range 0-4: {jd['level']}"

    return True, ""


def reconcile_level(jd: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, int]:
\
\
\
\
\
\

    reported = int(jd.get("level", -1))
    count = sum(1 for sec in SECTIONS if jd["components_present"][sec] is True)
    corrected = reported != count
    jd["level"] = count
    return jd, corrected, reported


REPAIR_INSTRUCTION = (
    "Your previous response was not a valid judgement. Specifically: {problem}. "
    "Return ONLY the corrected JSON object with exactly this structure: "
    '{{"components_present": {{"introduction": true, "methods": true, "results": true, '
    '"discussion": true}}, "level": 0}} where each component value is a JSON boolean '
    "(true or false) and level is an integer equal to the number of true values. "
    "Do not include any text outside the JSON object."
)


async def repair_judgment_once(
    client: AsyncOpenAI,
    *,
    base_messages: List[Dict[str, Any]],
    prior_raw: str,
    problem: str,
    temperature: float,
    max_tokens: int,
):
\
\

    repair_messages = list(base_messages) + [
        {"role": "assistant", "content": prior_raw},
        {"role": "user", "content": REPAIR_INSTRUCTION.format(problem=problem)},
    ]
    response = await create_with_context_retry(
        client,
        model=SERVED_MODEL_NAME,
        messages=repair_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"repetition_penalty": REPETITION_PENALTY},
    )
    try:
        raw = response.choices[0].message.content or ""
    except Exception:
        raw = ""
    if not raw.strip():
        return None, raw
    jd, _ = parse_judgment_json(extract_json_candidate(raw))
    if jd is None:
        return None, raw
    jd, _ = coerce_judgment_schema(jd)
    return jd, raw


def default_result(item: WorkItem) -> Dict[str, Any]:
    base = {
        "model": MODEL_NAME,
        "served_model_name": SERVED_MODEL_NAME,
        "variant": VARIANT,
        "doi": item.doi,
        "doi_safe": item.doi_safe,
        "user_prompt_file": str(item.user_prompt_path),
        "ga_image_path": item.ga_image_path,
        "judgment_file": str(JUDGMENT_DIR / f"{item.doi_safe}{JUDGMENT_SUFFIX}"),
        "status": "",
        "error_message": "",
        "paper_text_found": False,
        "text_chars": 0,
        "ga_image_found": False,
        "ga_image_bytes": 0,
        "figures_available": len(item.figure_paths),
        "figures_sent": 0,
        "figures_dropped": 0,
        "figure_bytes": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "inference_time": 0.0,
        "tokens_per_second": 0.0,
        "raw_output_length": 0,
        "had_think_tags": False,
        "had_code_fences": False,
        "parse_mode": "",
        "repaired": False,
        "level_mismatch_corrected": False,
        "reported_level": "",
        "level": "",
    }
    for sec in SECTIONS:
        base[f"present_{sec}"] = ""
    return base


def save_checkpoint(result: Dict[str, Any]) -> None:
    CHECKPOINT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now().isoformat(), **result}
    with open(CHECKPOINT_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class Progress:
    def __init__(self, total: int) -> None:
        self.total = total
        self.done = 0
        self.lock = asyncio.Lock()

    async def tick(self, doi_safe: str, status: str) -> None:
        async with self.lock:
            self.done += 1
            done = self.done
        print(f"[PROGRESS] {doi_safe} | {status} | processed {done}/{self.total} | "
              f"remaining {self.total - done}", flush=True)


async def _process_one_core(
    item: WorkItem,
    system_prompt: str,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    force: bool,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    result = default_result(item)
    judgment_path = JUDGMENT_DIR / f"{item.doi_safe}{JUDGMENT_SUFFIX}"

    if judgment_path.exists() and not force:
        result["status"] = "skipped"
        print(f"[SKIP] {item.doi_safe} | existing judgment", flush=True)
        save_checkpoint(result)
        return result


    if not item.user_prompt_path.exists():
        result["status"] = "paper_text_missing"
        result["error_message"] = f"user prompt not found: {item.user_prompt_path}"
        print(f"[FAIL] {item.doi_safe} | paper_text_missing", flush=True)
        save_checkpoint(result)
        return result
    try:
        paper_text = load_text(item.user_prompt_path)
        if not paper_text:
            raise ValueError("user prompt file is empty")
    except Exception as exc:
        result["status"] = "paper_text_read_error"
        result["error_message"] = str(exc)
        print(f"[FAIL] {item.doi_safe} | paper_text_read_error | {exc}", flush=True)
        save_checkpoint(result)
        return result
    result["paper_text_found"] = True
    result["text_chars"] = len(paper_text)


    ga_block, ga_stats = load_image_block(item.ga_image_path)
    result["ga_image_found"] = ga_stats["found"]
    result["ga_image_bytes"] = ga_stats["bytes"]
    if ga_block is None:
        result["status"] = "ga_image_error"
        result["error_message"] = ga_stats["error"]
        print(f"[FAIL] {item.doi_safe} | ga_image_error | {ga_stats['error']}", flush=True)
        save_checkpoint(result)
        return result


    figure_blocks: List[Dict[str, Any]] = []
    if VARIANT == "B" and item.figure_paths:
        figure_blocks, fig_bytes = load_figure_blocks(item.figure_paths)
        result["figure_bytes"] = fig_bytes


    async with semaphore:
        start = time.time()
        response = None
        last_exc: Optional[Exception] = None
        sent_blocks = list(figure_blocks)

        while True:
            user_content = build_user_content(paper_text, sent_blocks, ga_block)
            base_messages = build_messages(system_prompt, user_content)
            try:
                response = await create_with_context_retry(
                    client,
                    model=SERVED_MODEL_NAME,
                    messages=base_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body={"repetition_penalty": REPETITION_PENALTY},
                )
                break
            except Exception as exc:
                last_exc = exc
                if sent_blocks and _is_context_overflow(str(exc)):
                    sent_blocks = sent_blocks[:-1]
                    print(f"[WARN] {item.doi_safe} | context overflow, dropping a figure "
                          f"({len(sent_blocks)} left)", flush=True)
                    continue
                break

        elapsed = time.time() - start

    result["figures_sent"] = len(sent_blocks)
    result["figures_dropped"] = len(figure_blocks) - len(sent_blocks)

    if response is None:
        result["status"] = "api_error"
        result["error_message"] = str(last_exc)
        print(f"[FAIL] {item.doi_safe} | api_error | {last_exc}", flush=True)
        save_checkpoint(result)
        return result

    user_content = build_user_content(paper_text, sent_blocks, ga_block)
    base_messages = build_messages(system_prompt, user_content)

    try:
        raw_content = response.choices[0].message.content or ""
    except Exception:
        raw_content = ""

    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    result["input_tokens"] = input_tokens or 0
    result["output_tokens"] = output_tokens or 0
    result["inference_time"] = round(elapsed, 2)
    result["tokens_per_second"] = round((output_tokens or 0) / elapsed, 2) if elapsed > 0 else 0.0
    result["raw_output_length"] = len(raw_content)
    result["had_think_tags"] = bool(re.search(r"<think>", raw_content, flags=re.IGNORECASE))
    result["had_code_fences"] = raw_content.strip().startswith("```")

    if not raw_content.strip():
        result["status"] = "empty_response_error"
        result["error_message"] = "Model returned an empty response"
        save_text(ERROR_DIR / f"{item.doi_safe}{RAW_OUTPUT_SUFFIX}", raw_content)
        print(f"[FAIL] {item.doi_safe} | empty_response_error", flush=True)
        save_checkpoint(result)
        return result

    cleaned = extract_json_candidate(raw_content)
    jd, parse_mode = parse_judgment_json(cleaned)
    result["parse_mode"] = parse_mode
    if jd is None:
        result["status"] = "json_parse_error"
        result["error_message"] = "JSON parse failed after repair"
        save_text(ERROR_DIR / f"{item.doi_safe}{RAW_OUTPUT_SUFFIX}", raw_content)
        save_text(ERROR_DIR / f"{item.doi_safe}{CLEANED_OUTPUT_SUFFIX}", cleaned)
        print(f"[FAIL] {item.doi_safe} | json_parse_error", flush=True)
        save_checkpoint(result)
        return result

    jd, _n = coerce_judgment_schema(jd)
    valid, validation_error = validate_judgment(jd)

    if not valid:
        try:
            repaired_jd, repair_raw = await repair_judgment_once(
                client,
                base_messages=base_messages,
                prior_raw=raw_content,
                problem=validation_error,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception:
            repaired_jd, repair_raw = None, ""
        if repaired_jd is not None:
            re_valid, re_error = validate_judgment(repaired_jd)
            if re_valid:
                jd = repaired_jd
                raw_content = repair_raw or raw_content
                result["repaired"] = True
                valid, validation_error = True, ""
            else:
                validation_error = re_error

    if not valid:
        result["status"] = "validation_error"
        result["error_message"] = validation_error
        save_text(ERROR_DIR / f"{item.doi_safe}{RAW_OUTPUT_SUFFIX}", raw_content)
        save_json(ERROR_DIR / f"{item.doi_safe}{INVALID_JUDGMENT_SUFFIX}", jd)
        print(f"[FAIL] {item.doi_safe} | validation_error | {validation_error}", flush=True)
        save_checkpoint(result)
        return result

    jd, corrected, reported = reconcile_level(jd)
    result["level_mismatch_corrected"] = corrected
    result["reported_level"] = reported
    if corrected:
        print(f"[WARN] {item.doi_safe} | level {reported} != count {jd['level']}, "
              f"recomputed from components_present", flush=True)

    for sec in SECTIONS:
        result[f"present_{sec}"] = jd["components_present"][sec]
    result["level"] = jd["level"]

    jd_out = {
        "doi": item.doi,
        "doi_safe": item.doi_safe,
        "variant": VARIANT,
        "model": MODEL_NAME,
        "method": "naive_vlm",
        "figures_sent": result["figures_sent"],
        **jd,
    }
    try:
        save_json(judgment_path, jd_out)
        save_text(RAW_OUTPUT_DIR / f"{item.doi_safe}{RAW_OUTPUT_SUFFIX}", raw_content)
    except Exception as exc:
        result["status"] = "save_error"
        result["error_message"] = str(exc)
        print(f"[FAIL] {item.doi_safe} | save_error | {exc}", flush=True)
        save_checkpoint(result)
        return result

    result["status"] = "success"
    result["judgment_file"] = str(judgment_path)
    print(
        f"[OK] {item.doi_safe} | level={jd['level']}/4 | figs={result['figures_sent']} | "
        f"tokens={result['input_tokens']}+{result['output_tokens']} | {elapsed:.1f}s",
        flush=True,
    )
    save_checkpoint(result)
    return result


async def process_item(
    item: WorkItem,
    system_prompt: str,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    force: bool,
    temperature: float,
    max_tokens: int,
    progress: Progress,
) -> Dict[str, Any]:
    result = await _process_one_core(item, system_prompt, client, semaphore, force, temperature, max_tokens)
    await progress.tick(item.doi_safe, result.get("status", "unknown"))
    return result


def pct(n: int, d: int) -> float:
    return round(n / d * 100, 2) if d else 0.0


def avg(xs) -> float:
    return round(statistics.mean(xs), 2) if xs else 0.0


def med(xs) -> float:
    return round(statistics.median(xs), 2) if xs else 0.0


def compute_summary(results: List[Dict[str, Any]], gpu_info: Dict[str, Any],
                    total_wall_time: float, args: argparse.Namespace) -> Dict[str, Any]:
    processed = [r for r in results if r["status"] != "skipped"]
    successful = [r for r in processed if r["status"] == "success"]
    failed = [r for r in processed if r["status"] != "success"]
    skipped = [r for r in results if r["status"] == "skipped"]

    error_breakdown: Dict[str, int] = {}
    for r in failed:
        error_breakdown[r["status"]] = error_breakdown.get(r["status"], 0) + 1

    levels = [int(r["level"]) for r in successful if r["level"] != ""]
    level_dist = {str(k): sum(1 for x in levels if x == k) for k in range(5)}
    component_rate = {
        sec: pct(sum(1 for r in successful if r[f"present_{sec}"] is True), len(successful))
        for sec in SECTIONS
    }

    return {
        "task": "task1_completeness",
        "method": "naive_vlm",
        "variant": VARIANT,
        "model_name": MODEL_NAME,
        "served_model_name": SERVED_MODEL_NAME,
        "model_path": MODEL_PATH,
        "timestamp": datetime.now().isoformat(),
        "total_wall_time_sec": round(total_wall_time, 2),
        "total_wall_time_min": round(total_wall_time / 60, 2),
        "system_prompt_file": str(SYSTEM_PROMPT_FILE),
        "user_prompt_dir": str(USER_PROMPT_DIR),
        "figure_paths_csv": str(FIGURE_PATHS_CSV) if VARIANT == "B" else "",
        "ga_image_csv": str(GA_IMAGE_CSV),
        "complete_dois_file": str(COMPLETE_DOIS_FILE),
        "output_root": str(OUTPUT_ROOT),
        "args": vars(args),
        "gpu_info": gpu_info,
        "total_rows_selected": len(results),
        "skipped_count": len(skipped),
        "processed_count": len(processed),
        "success_count": len(successful),
        "fail_count": len(failed),
        "success_rate_processed": pct(len(successful), len(processed)),
        "success_rate_all_selected": pct(len(successful), len(results)),
        "error_breakdown": error_breakdown,
        "avg_inference_time": avg([r["inference_time"] for r in successful]),
        "median_inference_time": med([r["inference_time"] for r in successful]),
        "avg_throughput_tokens_per_sec": avg([r["tokens_per_second"] for r in successful if r["tokens_per_second"] > 0]),
        "total_input_tokens": sum(int(r["input_tokens"]) for r in successful),
        "total_output_tokens": sum(int(r["output_tokens"]) for r in successful),
        "avg_input_tokens": avg([r["input_tokens"] for r in successful]),
        "avg_text_chars": avg([r["text_chars"] for r in successful]),
        "avg_figures_sent": avg([r["figures_sent"] for r in successful]),
        "total_figures_dropped": sum(int(r["figures_dropped"]) for r in successful),
        "papers_with_figures_dropped": sum(1 for r in successful if int(r["figures_dropped"]) > 0),
        "papers_with_zero_figures": sum(1 for r in successful if int(r["figures_sent"]) == 0),
        "parse_repaired_count": sum(1 for r in successful if r["parse_mode"] == "repaired"),
        "reask_repaired_count": sum(1 for r in successful if r["repaired"] is True),
        "level_mismatch_corrected_count": sum(1 for r in successful if r["level_mismatch_corrected"] is True),
        "avg_level": avg(levels),
        "level_distribution": level_dist,
        "component_present_rate_pct": component_rate,
    }


def write_reports(results: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    save_json(SUMMARY_JSON, summary)

    lines = [
        f"Naive VLM Baseline — {MODEL_NAME} — Variant {VARIANT}",
        "=" * 60,
        f"Generated: {summary['timestamp']}",
        f"Model path: {summary['model_path']}",
        f"System prompt: {summary['system_prompt_file']}",
        f"User prompts: {summary['user_prompt_dir']}",
        f"Figure paths CSV: {summary['figure_paths_csv']}",
        f"GA image CSV: {summary['ga_image_csv']}",
        f"Output root: {summary['output_root']}",
        "",
        "Run counts",
        "----------",
        f"Rows selected: {summary['total_rows_selected']}",
        f"Skipped existing judgments: {summary['skipped_count']}",
        f"Processed this run: {summary['processed_count']}",
        f"Successful: {summary['success_count']}",
        f"Failed: {summary['fail_count']}",
        f"Success rate processed: {summary['success_rate_processed']}%",
        f"Error breakdown: {summary['error_breakdown']}",
        "",
        "Inputs",
        "------",
        f"Average paper text chars: {summary['avg_text_chars']}",
        f"Average figures sent: {summary['avg_figures_sent']}",
        f"Papers with zero figures sent: {summary['papers_with_zero_figures']}",
        f"Papers with figures dropped (context): {summary['papers_with_figures_dropped']}",
        f"Total figures dropped: {summary['total_figures_dropped']}",
        "",
        "Timing and tokens",
        "-----------------",
        f"Avg inference time: {summary['avg_inference_time']} sec",
        f"Median inference time: {summary['median_inference_time']} sec",
        f"Avg throughput: {summary['avg_throughput_tokens_per_sec']} tok/sec",
        f"Avg input tokens: {summary['avg_input_tokens']}",
        f"Total input tokens: {summary['total_input_tokens']:,}",
        f"Total output tokens: {summary['total_output_tokens']:,}",
        "",
        "Output quality",
        "--------------",
        f"JSON repaired by json-repair: {summary['parse_repaired_count']}",
        f"Fixed by corrective re-ask: {summary['reask_repaired_count']}",
        f"Level recomputed from components: {summary['level_mismatch_corrected_count']}",
        "",
        "Judgements",
        "----------",
        f"Average level: {summary['avg_level']}/4",
        f"Level distribution: {summary['level_distribution']}",
        f"Component present rate (%): {summary['component_present_rate_pct']}",
        "",
        "GPU info",
        "--------",
        json.dumps(summary.get("gpu_info", {}), indent=2),
        "",
        f"Total wall time: {summary['total_wall_time_sec']} sec ({summary['total_wall_time_min']} min)",
        f"Results CSV: {RESULTS_CSV}",
        f"Summary JSON: {SUMMARY_JSON}",
    ]
    save_text(SUMMARY_TXT, "\n".join(lines) + "\n")


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Naive VLM baseline GA judgement — Qwen3-VL-32B-AWQ")
    parser.add_argument("--variant", type=str, required=True, choices=["A", "B"],
                        help="A = paper text + GA image. B = paper text + tables + paper figures + GA image.")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N selected items")
    parser.add_argument("--start-index", type=int, default=None, help="Start index after sorting by safe DOI")
    parser.add_argument("--end-index", type=int, default=None, help="End index after sorting by safe DOI, exclusive")
    parser.add_argument("--run-tag", type=str, default=None, help="Suffix for reports + checkpoint only (e.g. h1, h2)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing judgments for selected items")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Async request concurrency")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max output tokens")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="OpenAI client timeout in seconds")
    parser.add_argument("--vllm-url", type=str, default=VLLM_URL, help="Base vLLM server URL, without /v1")
    args = parser.parse_args()

    apply_variant(args.variant)
    apply_run_tag(args.run_tag)
    make_dirs()

    if repair_json is None:
        print("ERROR: json-repair not installed. Run: pip install json-repair --break-system-packages", flush=True)
        sys.exit(1)

    print("=" * 80, flush=True)
    print(f"Naive VLM Baseline — Qwen3-VL-32B-AWQ — Variant {VARIANT}", flush=True)
    print("=" * 80, flush=True)
    print(f"System prompt:    {SYSTEM_PROMPT_FILE}", flush=True)
    print(f"User prompt dir:  {USER_PROMPT_DIR}", flush=True)
    if VARIANT == "B":
        print(f"Figure paths CSV: {FIGURE_PATHS_CSV}", flush=True)
    print(f"GA image CSV:     {GA_IMAGE_CSV}", flush=True)
    print(f"Complete DOIs:    {COMPLETE_DOIS_FILE}", flush=True)
    print(f"Output root:      {OUTPUT_ROOT}", flush=True)
    print(f"Run tag:          {args.run_tag}", flush=True)
    print(f"Index range:      [{args.start_index}, {args.end_index})", flush=True)
    print(f"Force rerun:      {args.force}", flush=True)
    print(f"Concurrency:      {args.concurrency}", flush=True)
    print(f"Max tokens:       {args.max_tokens}", flush=True)
    print(f"Temperature:      {args.temperature}", flush=True)
    print(f"Results CSV:      {RESULTS_CSV}", flush=True)
    print("=" * 80, flush=True)

    if not SYSTEM_PROMPT_FILE.exists():
        raise FileNotFoundError(f"System prompt not found: {SYSTEM_PROMPT_FILE}")
    if not USER_PROMPT_DIR.exists():
        raise FileNotFoundError(f"User prompt dir not found: {USER_PROMPT_DIR}")
    system_prompt = load_text(SYSTEM_PROMPT_FILE)
    if not system_prompt:
        raise ValueError(f"System prompt file is empty: {SYSTEM_PROMPT_FILE}")

    items = load_work_items(limit=args.limit, start_index=args.start_index, end_index=args.end_index)
    print(f"Selected items: {len(items)}", flush=True)
    existing = sum(1 for it in items if (JUDGMENT_DIR / f"{it.doi_safe}{JUDGMENT_SUFFIX}").exists())
    print(f"Existing judgments among selected: {existing}", flush=True)
    print(f"Items to process if no --force: {len(items) - existing}", flush=True)

    missing_text = sum(1 for it in items if not it.user_prompt_path.exists())
    missing_ga = sum(1 for it in items if not it.ga_image_path)
    print(f"Missing user prompts in selection: {missing_text}", flush=True)
    print(f"Missing GA paths in selection: {missing_ga}", flush=True)
    if VARIANT == "B":
        zero_fig = sum(1 for it in items if not it.figure_paths)
        print(f"Items with zero figures in selection: {zero_fig}", flush=True)

    gpu_info = get_gpu_info()
    print(f"GPU info: {gpu_info}", flush=True)
    print("Connecting to vLLM server...", flush=True)

    client = AsyncOpenAI(base_url=f"{args.vllm_url.rstrip('/')}/v1", api_key="not-needed", timeout=args.timeout)
    semaphore = asyncio.Semaphore(args.concurrency)
    progress = Progress(total=len(items))

    total_start = time.time()
    tasks = [
        process_item(
            item=item, system_prompt=system_prompt, client=client, semaphore=semaphore,
            force=args.force, temperature=args.temperature, max_tokens=args.max_tokens,
            progress=progress,
        )
        for item in items
    ]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    results = []
    for _item, _r in zip(items, raw_results):
        if isinstance(_r, dict):
            results.append(_r)
        else:
            _row = default_result(_item)
            _row["status"] = "crash_error"
            _row["error_message"] = repr(_r)
            save_checkpoint(_row)
            results.append(_row)
    total_wall_time = time.time() - total_start

    summary = compute_summary(results, gpu_info, total_wall_time, args)
    write_reports(results, summary)

    print("\n" + "=" * 80, flush=True)
    print("NAIVE BASELINE INFERENCE COMPLETE", flush=True)
    print("=" * 80, flush=True)
    print(f"Variant:        {VARIANT}", flush=True)
    print(f"Rows selected:  {summary['total_rows_selected']}", flush=True)
    print(f"Skipped:        {summary['skipped_count']}", flush=True)
    print(f"Processed:      {summary['processed_count']}", flush=True)
    print(f"Successful:     {summary['success_count']}", flush=True)
    print(f"Failed:         {summary['fail_count']}", flush=True)
    print(f"Error breakdown:{summary['error_breakdown']}", flush=True)
    print(f"Avg level:      {summary['avg_level']}/4", flush=True)
    print(f"Level dist:     {summary['level_distribution']}", flush=True)
    print(f"Level corrected:{summary['level_mismatch_corrected_count']}", flush=True)
    print(f"Results CSV:    {RESULTS_CSV}", flush=True)
    print(f"Summary TXT:    {SUMMARY_TXT}", flush=True)
    print(f"Wall time:      {summary['total_wall_time_min']} min", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    if sys.platform.startswith("linux"):
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    asyncio.run(main_async())
