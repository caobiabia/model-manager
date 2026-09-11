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
# 医疗评测数据在 eval/data_medical/ (H200 仓库内自托管, 不再依赖父目录)
_EVAL_DIR = Path(__file__).resolve().parent
_CSP_DEV = _EVAL_DIR.parent
_WORKSPACE = _CSP_DEV.parent
_CSP_ROOT = _WORKSPACE.parent

for _p in (_CSP_DEV, _WORKSPACE):  # model_config 现位于仓库内 (H200)
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from model_config import MODELS, get_model_by_name, get_base_url  # noqa: E402

# -- data / output paths --------------------------------------------------
BENCHMARK_DATA_DIR = _EVAL_DIR / "data_medical"
SUBSET_DATA_FILE = _EVAL_DIR / "test_subset.jsonl"
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


def load_processed_ids(progress_file: str) -> set[str]:
    """Load already-processed question realidx for resume support.

    Returns the raw line strings. Data-file realidx may be str (hash-based
    benches) or int (index-based benches); return the original text so
    callers can match against ``str(realidx)`` regardless of type.
    """
    if not os.path.exists(progress_file):
        return set()
    with open(progress_file, "r") as f:
        return set(line.strip() for line in f if line.strip())


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


def get_extract_model() -> tuple[OpenAI, str, bool]:
    """Return ``(client, model_name, is_deepseek)`` for answer extraction.

    TEMP: switched from the remote Qwen3.8-27B instance
    (http://114.55.210.21:26008/v1) to the local vLLM instance
    Qwen3.6-35B-A3B (127.0.0.1:26001) in non-reasoning mode
    (``chat_template_kwargs.enable_thinking=False``, hence is_deepseek=False).
    Restore base_url/model_name below when the remote instance is back.
    The extraction model is mandatory: MCQ answers must be extracted by an
    LLM extractor, so a missing local instance raises an error instead of
    silently falling back to regex.
    """
    base_url = "http://127.0.0.1:26001/v1"
    model_name = "Qwen/Qwen3.6-35B-A3B"
    return OpenAI(api_key="not-needed", base_url=base_url), model_name, False


# -- model calling ---------------------------------------------------------

_server_max_tokens_cache: dict[str, int | None] = {}
_server_max_tokens_lock = Lock()


def _server_max_tokens(client: OpenAI) -> int | None:
    """Total context length of the serving engine (cached per client URL).

    vLLM reports ``max_model_len`` on each entry of the OpenAI-compatible
    ``/models`` listing and rejects requests with ``max_tokens`` above it
    (HTTP 400). Returns ``None`` for servers/SDK versions that don't expose
    the field, in which case no capping is applied.
    """
    key = str(client.base_url)
    with _server_max_tokens_lock:
        if key in _server_max_tokens_cache:
            return _server_max_tokens_cache[key]
    value: int | None = None
    try:
        for m in client.models.list():
            if getattr(m, "id", None) and getattr(m, "max_model_len", None):
                value = int(m.max_model_len)
                break
    except Exception:
        value = None
    with _server_max_tokens_lock:
        _server_max_tokens_cache[key] = value
    return value


def resolve_effort(spec: dict | None, value) -> object | None:
    """Map a requested thinking-effort (label or raw number) through a model's
    `thinking` spec (see model_config.py) to its wire value.

    Returns None when the model declares no capability, no value was requested,
    or the value is invalid for that model — callers then fall back to the
    model-side default.
    """
    if value is None or not isinstance(spec, dict):
        return None
    efforts = spec.get("efforts") or {}
    if isinstance(value, str) and value in efforts:
        return efforts[value]
    numeric = spec.get("numeric")
    if numeric is not None:
        try:
            n = int(value)
        except (TypeError, ValueError):
            return None
        if numeric.get("min", 0) <= n <= numeric.get("max", 10**9):
            return n
    return None


def call_model(
    client: OpenAI,
    model_name: str,
    messages: list[dict],
    mode: str,
    is_deepseek: bool,
    max_tokens: int | None = None,
    max_retries: int = 3,
    temperature: float | None = None,
    reasoning_effort=None,
    thinking_spec: dict | None = None,
) -> tuple[str, str, object | None]:
    """Call a chat model with mode-appropriate thinking settings.

    ``temperature`` defaults the request to greedy decoding (0.0); pass a
    value > 0 to sample instead. It is only applied where the mode already
    sends a temperature (cot via Deepseek APIs relies on server defaults).

    Returns ``(raw_response, reasoning, usage)``.
    """
    kwargs: dict = dict(model=model_name, messages=messages, seed=42)
    temp = 0.0 if temperature is None else temperature
    if mode == "zero_shot":
        kwargs["temperature"] = temp
        kwargs["max_tokens"] = max_tokens or 1024
        if is_deepseek:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        else:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    else:  # cot
        # Eval cap is 200k tokens for reasoning-heavy COT (avoid truncation),
        # lowered to the serving engine's context when it reports one.
        kwargs["max_tokens"] = max_tokens or 200000
        # Optional per-run thinking effort; resolved through the model's
        # `thinking` spec. None (unsupported / invalid / not requested) keeps
        # the previous behavior: model-side default for vLLM, "high" for the
        # DeepSeek API provider.
        wire = resolve_effort(thinking_spec, reasoning_effort)
        top_level = (thinking_spec or {}).get("transport") == "top_level"
        if is_deepseek:
            kwargs["reasoning_effort"] = wire if wire is not None else "high"
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        else:
            ctk: dict = {"enable_thinking": True}
            if wire is not None and not top_level:
                ctk["reasoning_effort"] = wire
            kwargs["extra_body"] = {"chat_template_kwargs": ctk}
            if wire is not None and top_level:
                kwargs["reasoning_effort"] = wire
            kwargs["temperature"] = temp

    # vLLM requires prompt + max_tokens <= max_model_len (HTTP 400).
    # Cap the request against the server's context, reserving room for the
    # prompt (estimated as at most its character count in tokens, which is
    # safe for CJK-heavy inputs) plus chat-template overhead.
    srv_max = _server_max_tokens(client)
    if srv_max:
        est_prompt = sum(len(m.get("content") or "") for m in messages)
        kwargs["max_tokens"] = min(
            kwargs["max_tokens"],
            max(256, srv_max - est_prompt - 256),
        )

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


def extract_free_answer(text: str) -> str | None:
    """Extract the final numeric answer from a model response.

    Only explicit answer formats are accepted: the canonical ``#### <number>``
    marker, a ``\\boxed{<number>}`` pattern (AIME style), or explicit answer
    labels (English/Chinese). There is no loose "last number in text" fallback:
    if the model did not answer in an expected format, ``None`` is returned and
    the question is treated as wrong.
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

    return None


# -- inline think splitting ------------------------------------------------

_THINK_FULL_RE = re.compile(r"^<think>([\s\S]*?)</think>")


def strip_think(text: str) -> tuple[str, str]:
    """Split an inline think block out of a raw model response.

    vLLM servers launched without ``--reasoning-parser`` return the model's
    thinking inside ``message.content``: either as a complete
    ``<think>...</think>`` block or (Qwen-style chat templates consume the
    opening tag in the generation prompt) as thinking text terminated by a
    bare ``</think>``. Responses with no closing ``</think>`` -- thinking
    disabled, a backend that separates reasoning, or a model that answered
    directly -- are returned unchanged.

    Returns ``(thinking, final_answer)``.
    """
    if not text:
        return "", ""
    m = _THINK_FULL_RE.match(text)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    i = text.find("</think>")
    if i >= 0:
        return text[:i].strip(), text[i + len("</think>"):].strip()
    return "", text


_MAX_EXTRACT_INPUT_CHARS = 32768
_EXTRACT_TRUNC_MARKER = "\n...[truncated]...\n"
_EXTRACT_ANSWER_MARKER_RE = re.compile(
    r"(?:The\s+correct\s+answer\s+is|Correct\s+answer\s*:?|"
    r"Answer\s*:?|Definite\s+Answer\s*:?|Final\s+answer\s*:?|"
    r"答案\s*(?:是|为|：))",
    re.IGNORECASE,
)


def _truncate_for_extract(text: str, budget: int) -> str:
    """Keep answer-relevant parts of an over-long response for extraction.

    Preserves the head (explicit answer is usually stated there), the tail
    (final conclusion), and windows around explicit answer markers found in
    the middle, so the extractor still sees the answer instead of a
    mid-reasoning cut.
    """
    if len(text) <= budget:
        return text

    head_chars = max(400, int(budget * 0.30))
    tail_chars = max(400, int(budget * 0.15))
    head = text[:head_chars]
    tail = text[-tail_chars:]
    middle = text[head_chars:len(text) - tail_chars]

    snippets = [head]
    for m in list(_EXTRACT_ANSWER_MARKER_RE.finditer(middle))[:4]:
        s = max(0, m.start() - 120)
        e = min(len(middle), m.end() + 320)
        snippets.append(middle[s:e])
    snippets.append(tail)

    condensed = _EXTRACT_TRUNC_MARKER.join(snippets)
    while len(condensed) > budget and len(snippets) > 2:
        snippets.pop(1)
        condensed = _EXTRACT_TRUNC_MARKER.join(snippets)
    return condensed[:budget]


def _build_extract_prompt(raw_response: str, options: dict) -> str:
    """Build the extraction prompt, keeping the full input under 32768 chars.

    If the model response is too long, keep the head (explicit answer is
    usually stated there) and the tail (final conclusion), so the extractor
    still sees the answer instead of a mid-reasoning cut.
    """
    options_list = ", ".join(options.keys())
    prefix = (
        "You are an answer extractor. Extract the answer option from the "
        'text below. Only return the answer as a JSON object: {"answer": "<option>"}, '
        f"where <option> is one of: {options_list}.\n"
        "Text:\n"
    )
    budget = max(1, _MAX_EXTRACT_INPUT_CHARS - len(prefix))
    text = _truncate_for_extract(raw_response, budget)
    return prefix + text


def extract_answer(
    raw_response: str,
    options: dict,
    extract_client: OpenAI,
    extract_model: str,
    is_deepseek: bool = False,
    retries: int = 3,
) -> str | None:
    """Extract the MCQ answer using the extraction model only.

    Ask the extraction model to return ``{"answer": "<letter>"}`` and parse
    it. No regex or other fallback extraction is used; returns ``None`` if
    the extraction model cannot produce a valid answer after ``retries``
    attempts (invalid/empty responses are retried).
    """
    if extract_client is None or extract_model is None:
        raise ValueError(
            "Extraction model is required for MCQ answers; got "
            f"extract_client={extract_client!r}, extract_model={extract_model!r}."
        )
    if not raw_response.strip():
        return None

    prompt = _build_extract_prompt(raw_response, options)
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
            if not extraction_raw:
                answer = ""
            else:
                try:
                    data = json.loads(extraction_raw)
                except json.JSONDecodeError:
                    data = None
                answer = (
                    str(data.get("answer", data.get("answer_idx", ""))).strip().upper()
                    if isinstance(data, dict) else ""
                )
            if answer in options:
                return answer
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(2)
    return None
