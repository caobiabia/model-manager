"""Eval shared utilities -- paths, data loading, client creation,
model calling, and answer extraction.

Ported from the original eval/common.py, benchmark.py, medqa_subset.py
but unified into one module so every bench uses the same call/extract path.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from threading import Lock

from openai import OpenAI

# -- path setup -----------------------------------------------------------
# eval/common.py  ->  csp_dev/  ->  workspace/  ->  /mntnlp/csp
_EVAL_DIR = Path(__file__).resolve().parent
_CSP_DEV = _EVAL_DIR.parent
_WORKSPACE = _CSP_DEV.parent
_CSP_ROOT = _WORKSPACE.parent

if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from model_config import MODELS, get_model_by_name, get_base_url  # noqa: E402

# -- data / output paths --------------------------------------------------
BENCHMARK_DATA_DIR = _CSP_ROOT / "MedicalAgentsBench-1.0" / "data"
SUBSET_DATA_FILE = _CSP_ROOT / "eval" / "test_subset.jsonl"
EVAL_OUTPUT_DIR = _EVAL_DIR / "output"


# -- data loading ----------------------------------------------------------

def load_jsonl(file_path: str | Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    data: list[dict] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_processed_ids(progress_file: str) -> set[int]:
    """Load already-processed question indices for resume support."""
    if not os.path.exists(progress_file):
        return set()
    with open(progress_file, "r") as f:
        return set(int(line.strip()) for line in f if line.strip().isdigit())


_progress_lock = Lock()


def save_progress(progress_file: str, idx: int) -> None:
    """Append a processed index to the progress file (resume checkpoint)."""
    with _progress_lock:
        with open(progress_file, "a") as f:
            f.write(f"{idx}\n")


# -- client / model resolution --------------------------------------------

def resolve_model(name: str) -> dict:
    """Look up a model by config key or served_model_name."""
    cfg = get_model_by_name(name)
    if cfg is None:
        raise SystemExit(f"Unknown model: '{name}'. Available: {list(MODELS.keys())}")
    return cfg


def create_client(cfg: dict) -> tuple[OpenAI, str, bool]:
    """Create an OpenAI client from a model config.

    Returns ``(client, served_model_name, is_deepseek)``.
    """
    is_deepseek = cfg.get("provider") == "deepseek"
    base_url = get_base_url(cfg)
    api_key = cfg.get("api_key", "not-needed") if is_deepseek else "not-needed"
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, cfg["served_model_name"], is_deepseek


def get_extract_model() -> tuple[OpenAI | None, str | None, bool]:
    """Return ``(client, model_name, is_deepseek)`` for the extraction step.

    Falls back to ``(None, None, False)`` (regex-only) when DeepSeek
    is not configured.
    """
    ds_cfg = MODELS.get("deepseek")
    if ds_cfg and ds_cfg.get("api_key"):
        client = OpenAI(api_key=ds_cfg["api_key"], base_url=ds_cfg["base_url"])
        return client, ds_cfg["served_model_name"], True
    return None, None, False


# -- model calling ---------------------------------------------------------

def call_model(
    client: OpenAI,
    model_name: str,
    messages: list[dict],
    mode: str,
    is_deepseek: bool,
    max_tokens: int | None = None,
    max_retries: int = 3,
) -> tuple[str, str, object | None]:
    """Call a chat model with mode-appropriate thinking settings.

    Returns ``(raw_response, reasoning, usage)``.
    """
    kwargs: dict = dict(model=model_name, messages=messages, seed=42)
    if mode == "zero_shot":
        kwargs["temperature"] = 0.0
        kwargs["max_tokens"] = max_tokens or 1024
        if is_deepseek:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        else:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    else:  # cot
        # Thinking models (e.g. Qwen3.6) often burn >6k tokens on hard
        # reasoning (AIME/GPQA). With max_tokens=6000 the response is
        # truncated mid-reasoning: content stays empty and scoring falls
        # back to garbage from the partial thinking trace. Keep 11000
        # (vLLM servers here are launched with max-model-len=12000).
        kwargs["max_tokens"] = max_tokens or 11000
        if is_deepseek:
            kwargs["reasoning_effort"] = "high"
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        else:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
            kwargs["temperature"] = 0.0

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(**kwargs)
            msg = completion.choices[0].message
            raw = (msg.content or "").strip()
            reasoning = (
                getattr(msg, "reasoning", None)
                or getattr(msg, "reasoning_content", None)
                or ""
            )
            return raw, reasoning, completion.usage
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    return "", "", None


# -- answer extraction -----------------------------------------------------


def extract_answer_regex(text: str, options: dict, strict: bool = False) -> str | None:
    """Extract answer letter from text using regex.

    Handles JSON objects, Chinese answer patterns, bare single
    letters, and English option-key search -- works for both the English
    MedicalAgentsBench prompts and the Chinese subset. In *strict* mode
    the loose English tail search is disabled, so a truncated thinking
    trace cannot yield a bogus answer from a stray option letter.
    """
    if not text:
        return None
    text = text.strip()

    # strip markdown code fences
    clean = re.sub(r"```(?:json)?\s*\n?", "", text)
    clean = re.sub(r"```", "", clean).strip()

    # JSON parse
    try:
        data = json.loads(clean)
        if isinstance(data, list):
            data = data[0] if data else {}
        if isinstance(data, dict):
            ans = str(data.get("answer", data.get("answer_idx", ""))).strip().upper()
            if ans and re.match(r"^[A-E]$", ans):
                return ans
    except (json.JSONDecodeError, AttributeError, IndexError):
        pass

    # explicit JSON pattern  {"answer": "B"}  or  {"answer_idx": "B"}
    m = re.search(r'\{\s*"answer(?:_idx)?"\s*:\s*"([A-E])"\s*\}', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Chinese pattern
    m = re.search(r"(?:\u6b63\u786e)?\u7b54\u6848[\u662f\u4e3a\uff1a:\s]*\**\s*([A-E])\b", text)
    if m:
        return m.group(1).upper()

    # bare single letter
    m = re.match(r"^\s*([A-E])\s*$", text)
    if m:
        return m.group(1).upper()

    # English fallback: search for option keys in the tail of the response.
    if not strict:
        tail = text[-500:]
        for key in options:
            if re.search(rf"\b{re.escape(key)}\b", tail):
                return key
    return None


def _normalize_number(s: str) -> str:
    """Normalize a numeric string for exact-match scoring (GSM8K-style)."""
    s = s.replace(",", "").replace("$", "").replace("\uffe5", "").replace("\u00a5", "")
    s = s.strip()
    sign = ""
    if s and s[0] in "+-":
        sign, s = s[0], s[1:]
    s = s.lstrip("0") or "0"
    if s.endswith(".0") and s.count(".") == 1:
        s = s[:-2]
    return sign + s


def extract_free_answer(text: str, strict: bool = False) -> str | None:
    """Extract the final numeric answer (GSM8K-style) from a model response.

    Checks, in order: the canonical ``#### <number>`` marker, explicit
    answer labels (English/Chinese), a ``\\boxed{<number>}`` pattern (AIME
    style), then the last number in the text. In *strict* mode only
    explicit answer markers are accepted (no last-number fallback), so a
    truncated reasoning trace cannot produce a bogus answer.
    """
    if not text:
        return None
    clean = re.sub(r"```(?:json)?\s*\n?", "", text)
    clean = re.sub(r"```", "", clean).strip()

    m = re.search(r"####\s*[$ \uffe5\u00a5]?\s*[+-]?\d[\d,]*(?:\.\d+)?", clean)
    if m:
        num = re.search(r"[+-]?\d[\d,]*(?:\.\d+)?", m.group(0))
        if num:
            return _normalize_number(num.group(0))

    m = re.search(r"\\boxed\{\s*([+-]?\d[\d,]*(?:\.\d+)?)\s*\}", clean)
    if m:
        return _normalize_number(m.group(1))

    for pat in (
        r"(?:final\s+answer|answer)\s*[::\uff1a]?\s*[$ \uffe5\u00a5]?\s*[+-]?\d[\d,]*(?:\.\d+)?",
        r"\u7b54\u6848\u662f\s*[::\uff1a]?\s*[$ \uffe5\u00a5]?\s*[+-]?\d[\d,]*(?:\.\d+)?",
    ):
        m = re.search(pat, clean, re.IGNORECASE)
        if m:
            num = re.search(r"[+-]?\d[\d,]*(?:\.\d+)?", m.group(0))
            if num:
                return _normalize_number(num.group(0))

    if not strict:
        nums = re.findall(r"[+-]?\d[\d,]*(?:\.\d+)?", clean)
        if nums:
            return _normalize_number(nums[-1])
    return None


_DIFF_LINE_RE = re.compile(
    r"^(diff |index |--- |\+\+\+ |@@ |[ +\-@\\]|"
    r"new file mode|deleted file mode|similarity index|"
    r"rename from|rename to|copy from|copy to|Binary files?)"
)


def extract_patch(text: str) -> str | None:
    """Extract a unified diff / code patch from a model response.

    Prefers a fenced diff/patch/python block; otherwise finds the first
    ``diff --git`` marker and keeps all diff-looking lines after it.
    Returns ``None`` when no patch-like content is found.
    """
    if not text:
        return None

    # fenced blocks first
    for block in re.findall(r"```[A-Za-z+-]*\s*\n(.*?)```", text, re.S):
        block = block.strip()
        if not block:
            continue
        if "diff --git" in block or "@@" in block or block.startswith("--- "):
            return block

    # raw unified diff: start at the first diff header
    m = re.search(r"^diff --git .*$", text, re.M)
    if m:
        lines = text[m.start():].splitlines()
        out: list[str] = []
        for line in lines:
            if _DIFF_LINE_RE.match(line) or line == r"\ No newline at end of file":
                out.append(line)
            elif out:
                break
        if out:
            return "\n".join(out).strip()
    return None


def extract_answer(
    raw_response: str,
    options: dict,
    extract_client: OpenAI | None = None,
    extract_model: str | None = None,
    is_deepseek: bool = False,
    retries: int = 3,
    strict: bool = False,
) -> str | None:
    """Unified answer extraction: model-based -> regex fallback.

    If *extract_client* is provided, ask the extraction model to return
    ``{"answer": "<letter>"}`` and parse it. On any failure, fall back
    to :func:`extract_answer_regex`.
    """
    # Fast path first: the deterministic regex extractor handles
    # well-formatted answers. Only fall back to an LLM extraction call
    # (which can hang/rate-limit and stalls the whole worker pool) when
    # regex cannot find an answer.
    regex_ans = extract_answer_regex(raw_response, options, strict=strict)
    if regex_ans is not None or extract_client is None or not raw_response.strip():
        return regex_ans

    options_list = ", ".join(options.keys())
    prompt = (
        "You are an answer extractor. Extract the answer option from the "
        'text below. Only return the answer as a JSON object: {"answer": "<option>"}, '
        f"where <option> is one of: {options_list}.\n"
        "Text:\n" + raw_response
    )
    for attempt in range(retries):
        try:
            ekwargs: dict = dict(
                model=extract_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=128,
                timeout=30,
            )
            if is_deepseek:
                ekwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            else:
                ekwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            resp = extract_client.chat.completions.create(**ekwargs)
            extraction_raw = resp.choices[0].message.content.strip()
            data = json.loads(extraction_raw)
            answer = str(data.get("answer", data.get("answer_idx", ""))).strip().upper()
            if answer in options:
                return answer
        except Exception:
            if attempt < retries - 1:
                time.sleep(2)
    return extract_answer_regex(raw_response, options)
