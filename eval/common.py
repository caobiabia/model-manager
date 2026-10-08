"""Eval shared utilities -- paths, data loading, client creation,
model calling, and answer extraction.

Ported from the original eval/common.py, benchmark.py, medqa_subset.py
but unified into one module so every bench uses the same call/extract path.
"""

from __future__ import annotations

import importlib
import itertools
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

# The OpenAI SDK talks to the network through httpx2 (its bundled httpx
# fork); build our timeouts and catch the exceptions its streaming path
# actually raises from the same library.
try:
    import httpx2 as httpx_backend
except ImportError:  # pragma: no cover - older SDKs use plain httpx
    import httpx as httpx_backend  # type: ignore[no-redef]

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
)

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


_jsonl_count_cache: dict[str, tuple[float, int]] = {}
_jsonl_count_lock = threading.Lock()


def count_jsonl_cached(file_path: str | Path) -> int:
    """Line count of a JSONL file, cached by (path, mtime).

    Callers that only need "how many questions does this bench have" used to
    json.loads() entire datasets (tens of MB each) on the event loop; with a
    few dozen benches that stalled the whole web UI.
    """
    path = str(file_path)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return 0
    with _jsonl_count_lock:
        hit = _jsonl_count_cache.get(path)
        if hit is not None and hit[0] == mtime:
            return hit[1]
    count = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
    except OSError:
        return 0
    with _jsonl_count_lock:
        _jsonl_count_cache[path] = (mtime, count)
    return count


# -- result-file stats -----------------------------------------------------
# A live run's result files grow into the tens of MB; parsing one with
# json.loads() blocks the caller for seconds. The web layer used to do that
# just to render "file (N entries)" and to reconstruct progress for finished
# runs -- with dozens of files it froze the whole console. Small files are
# parsed (once per mtime+size); large ones are counted from the sibling
# progress file, which gets exactly one line per persisted result.
_RESULT_PARSE_MAX_BYTES = 8 * 1024 * 1024
_result_stats_cache: dict[str, tuple[float, int, int | None, int | None]] = {}
_result_stats_lock = Lock()


def _progress_file_for(results_path: Path) -> Path:
    """progress_<model>_<bench>_<mode>.txt for <model>__<bench>__<mode>.json."""
    stem = results_path.name[: -len(".json")] if results_path.name.endswith(".json") else results_path.name
    parts = stem.split("__")
    if len(parts) != 3:
        return results_path.with_name(f"progress_{stem}.txt")
    model, bench, mode = parts
    return results_path.with_name(f"progress_{model}_{bench}_{mode}.txt")


def result_stats_cached(results_path: str | Path) -> tuple[int, int | None]:
    """Return ``(processed, correct)`` for a results file, cheaply.

    ``correct`` is None when the file is too large to parse for counting --
    callers should show "unknown" rather than a wrong number.
    """
    results_path = Path(results_path)
    try:
        st = results_path.stat()
    except OSError:
        return 0, None
    key = str(results_path)
    with _result_stats_lock:
        hit = _result_stats_cache.get(key)
    if hit is not None and hit[0] == st.st_mtime and hit[1] == st.st_size:
        return hit[2], hit[3]
    count: int | None = None
    correct: int | None = None
    if st.st_size <= _RESULT_PARSE_MAX_BYTES:
        try:
            data = json.loads(results_path.read_text(encoding="utf-8"))
            seen: set = set()
            unique = [
                x for x in data
                if x.get("realidx") not in seen and not seen.add(x.get("realidx"))
            ]
            count = len(unique)
            correct = sum(1 for x in unique if x.get("correct") is True)
        except Exception:
            count = None
    if count is None:
        count = count_jsonl_cached(_progress_file_for(results_path))
    with _result_stats_lock:
        _result_stats_cache[key] = (st.st_mtime, st.st_size, count, correct)
    return count, correct


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


# -- in-flight request control ---------------------------------------------
#
# Every completion is streamed and every live stream is registered, for two
# reasons:
#
#   1. A long CoT generation cannot be bounded by one "whole request"
#      timeout. With the SDK defaults (no explicit timeout + 2 SDK-level
#      retries) stacked on call_model's own retries, a single question could
#      burn ~90 minutes of 600s timeouts while the server kept generating.
#      Streaming makes the read timeout bound the *gap between tokens*
#      instead of the whole generation.
#   2. Stopping must abort the generation on the *server*, not merely stop
#      waiting for it. Closing the HTTP stream makes vLLM's chat route cancel
#      its handler (``with_cancellation``) and ``AsyncLLM.generate()`` abort
#      the request on the engine, so no orphaned generation keeps burning
#      GPU. Without this, a stopped run left dozens of zombie requests
#      generating for 45+ minutes.
#
# Aborts come in two scopes: ``None`` (the whole run, used by the stop
# signal) and a bench scope ``"model_key|bench_id"`` (used when one bench
# circuit-breaks).

_READ_TIMEOUT = 600.0
# vLLM streams one SSE chunk (one HTTP chunk) per generated token by default.
# At the eval's ~190 concurrent streams that is ~10k chunks/s, and the Python
# cost of parsing that chunked encoding (httpcore2/h11 + json) alone saturates
# the GIL of the process -- which is also the web console, so the UI froze
# while an eval ran. Coalescing this many tokens per chunk cuts that ~N-fold;
# the transcript, usage and stop behaviour are unchanged.
_STREAM_INTERVAL = 32
# read= bounds the gap between streamed tokens; connect/write/pool stay short
# so a dead endpoint fails fast instead of hanging the worker.
_CLIENT_TIMEOUT = httpx_backend.Timeout(
    connect=10.0, read=_READ_TIMEOUT, write=60.0, pool=10.0,
)

_abort_lock = Lock()
_abort_all = threading.Event()
_abort_scopes: dict[str, threading.Event] = {}
_live_streams: dict[str, dict[int, object]] = {}
_stream_ids = itertools.count()

# Servers that reject ``stream_options`` (usage on the final chunk); cleared
# on the first 400 so we keep the transcript instead of failing the question.
_stream_options_ok: dict[str, bool] = {}
# Same idea for the per-request ``stream_interval`` (not part of the OpenAI
# schema; only vLLM honours it).
_stream_interval_ok: dict[str, bool] = {}


class RequestAborted(Exception):
    """The run (or the bench scope) was aborted while this request was live.

    Callers must treat the question as *not attempted*: nothing is persisted,
    so a later resume retries it.
    """


class GenerationTimeout(Exception):
    """Stream stalled: no token arrived for _READ_TIMEOUT seconds.

    Not scored either -- a fabricated 0 would be indistinguishable from a
    wrong answer. The runner leaves the question pending for a retry.
    """


class ConnectionFailure(Exception):
    """Transport failure (connect timeout, dropped connection).

    Retryable, unlike a stalled generation: the endpoint is the problem,
    not the answer length.
    """


def abort_requested(scope: str | None = None) -> bool:
    # Called once per streamed chunk, so read the scope map without the lock
    # (a dict read is atomic under the GIL; writers hold the lock).
    if _abort_all.is_set():
        return True
    if scope is None:
        return False
    ev = _abort_scopes.get(scope)
    return ev is not None and ev.is_set()


def request_abort(scope: str | None = None) -> None:
    """Abort in-flight requests and refuse new ones.

    ``scope=None`` aborts the whole run; a bench scope aborts only that
    bench. Live streams of the affected scopes are closed, which is what
    makes the serving engine drop the requests.
    """
    if scope is None:
        _abort_all.set()
        with _abort_lock:
            scopes = list(_live_streams.keys())
    else:
        with _abort_lock:
            _abort_scopes.setdefault(scope, threading.Event()).set()
            scopes = [scope]
    closing: list[object] = []
    with _abort_lock:
        for s in scopes:
            closing.extend(_live_streams.pop(s, {}).values())
    for stream in closing:
        try:
            stream.close()
        except Exception:
            pass


def clear_abort(scope: str | None = None) -> None:
    """Re-arm requests after a stop (called at run/bench start)."""
    if scope is None:
        _abort_all.clear()
        with _abort_lock:
            _abort_scopes.clear()
        return
    with _abort_lock:
        _abort_scopes.pop(scope, None)


def _register_stream(scope: str | None, stream) -> int | None:
    if scope is None:
        return None
    sid = next(_stream_ids)
    with _abort_lock:
        _live_streams.setdefault(scope, {})[sid] = stream
    return sid


def _unregister_stream(scope: str | None, sid: int | None) -> None:
    if scope is None or sid is None:
        return
    with _abort_lock:
        live = _live_streams.get(scope)
        if not live:
            return
        live.pop(sid, None)
        if not live:
            _live_streams.pop(scope, None)


def is_deterministic_failure(err: BaseException) -> bool:
    """4xx client errors (bad prompt, context overflow) repeat forever.

    Those are persisted as a real wrong result so the run can move on;
    everything else (timeout, connection loss) stays retryable.
    """
    return isinstance(err, APIStatusError) and 400 <= err.status_code < 500


# -- client / model resolution --------------------------------------------

def make_client(base_url: str, api_key: str = "not-needed") -> OpenAI:
    """OpenAI client with an explicit, streaming-friendly timeout policy.

    ``max_retries=0``: retries are decided once in call_model. SDK-level
    retries multiply with them (3x3 attempts of 600s each), and every retry
    regenerates the whole answer from scratch.
    """
    return OpenAI(
        api_key=api_key, base_url=base_url,
        timeout=_CLIENT_TIMEOUT, max_retries=0,
    )


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
    return make_client(base_url, api_key), cfg["served_model_name"], is_deepseek


_extract_client: OpenAI | None = None
_extract_target: tuple[str, str, str, bool] | None = None
_extract_client_lock = Lock()

# Fallback when ``model_config.EXTRACT_MODEL_KEY`` is empty: the local vLLM
# instance of Qwen3.6-35B-A3B (thinking disabled via chat_template_kwargs).
_LOCAL_EXTRACT_BASE_URL = "http://127.0.0.1:26001/v1"
_LOCAL_EXTRACT_MODEL = "Qwen/Qwen3.6-35B-A3B"


def _extract_config_entry() -> dict | None:
    """The MODELS entry named by ``model_config.EXTRACT_MODEL_KEY`` (or None).

    ``model_config`` is reloaded first, mirroring ``app._cfg()``: the knob is
    read once per run instead of at import time, so editing the config takes
    effect without restarting the console.
    """
    import model_config

    try:
        importlib.reload(model_config)
    except Exception:  # 配置写坏了也要能用缓存里的旧模块, 由下面的校验兜底
        pass
    key = (getattr(model_config, "EXTRACT_MODEL_KEY", "") or "").strip()
    if not key:
        return None
    entry = model_config.MODELS.get(key)
    if not isinstance(entry, dict):
        raise RuntimeError(
            f"model_config.EXTRACT_MODEL_KEY={key!r} is not in MODELS; "
            "check model_config.py / models_user.json"
        )
    return entry


def _resolve_extract_target() -> tuple[str, str, str, bool]:
    """Resolve ``(base_url, api_key, model_name, is_deepseek)`` for extraction."""
    cfg = _extract_config_entry()
    if cfg is None:
        return _LOCAL_EXTRACT_BASE_URL, "not-needed", _LOCAL_EXTRACT_MODEL, False
    is_deepseek = cfg.get("provider") == "deepseek"
    if is_deepseek and not cfg.get("base_url"):
        raise RuntimeError(
            f"Extractor {cfg.get('display_name', cfg.get('served_model_name'))!r} "
            "is provider=deepseek but declares no base_url"
        )
    api_key = cfg.get("api_key", "not-needed") if is_deepseek else "not-needed"
    return get_base_url(cfg), api_key, cfg["served_model_name"], is_deepseek


def _verify_extract_endpoint(
    base_url: str, api_key: str, model_name: str, is_deepseek: bool,
) -> None:
    """Probe the extractor once, with the exact shape extraction requests use.

    MCQ answers are extracted by this model *only*, and an unreachable
    extractor used to be invisible: generation still succeeded, so every
    question was persisted as an ordinary wrong answer and the bench reported
    ~0% instead of an error. Failing here turns that into a loud, immediate
    run error.
    """
    kwargs: dict = {
        "model": model_name,
        "messages": [{"role": "user", "content": "ping"}],
        "temperature": 0.0,
        "max_tokens": 1,
        "timeout": 30,
        "extra_body": (
            {"thinking": {"type": "disabled"}} if is_deepseek
            else {"chat_template_kwargs": {"enable_thinking": False}}
        ),
    }
    try:
        make_client(base_url, api_key).chat.completions.create(**kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Answer-extraction model unreachable: {base_url} "
            f"(model={model_name!r}): {type(e).__name__}: {e}. "
            "MCQ benches (ceval / mmlu_std / gpqa_diamond / medical MCQ) and the "
            "inverse_ifeval judge need it -- point model_config.EXTRACT_MODEL_KEY "
            "at a reachable model, or start the local extractor on 26001."
        ) from e


def get_extract_model() -> tuple[OpenAI, str, bool]:
    """Return ``(client, model_name, is_deepseek)`` for answer extraction.

    The endpoint comes from ``model_config.EXTRACT_MODEL_KEY`` -- a remote
    ``provider="deepseek"`` entry is used as-is (its base_url and api_key), an
    empty knob falls back to the local 127.0.0.1:26001 instance. The client is
    cached per resolved target: it used to be rebuilt per call, which churned
    a fresh connection pool (and TLS/TCP handshake) for every question.
    """
    global _extract_client, _extract_target
    target = _resolve_extract_target()
    base_url, api_key, model_name, is_deepseek = target
    with _extract_client_lock:
        if _extract_client is not None and _extract_target == target:
            return _extract_client, model_name, is_deepseek
    _verify_extract_endpoint(base_url, api_key, model_name, is_deepseek)
    client = make_client(base_url, api_key)
    with _extract_client_lock:
        _extract_client, _extract_target = client, target
    return client, model_name, is_deepseek


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


def _stream_completion(
    client: OpenAI,
    kwargs: dict,
    scope: str | None,
) -> tuple[str, str, object | None]:
    """Run one streamed completion via the SDK's raw streaming API.

    The raw SSE lines are parsed here (one json.loads per chunk) instead of
    letting the SDK build a pydantic model per chunk: with ~200 concurrent
    streams at tens of chunks/second that per-chunk cost held the GIL almost
    continuously and starved every HTTP endpoint of the very same process
    (the console UI froze while an eval ran).

    Raises ``RequestAborted`` when the run/bench abort fires,
    ``GenerationTimeout`` when the stream stalls, and ``ConnectionFailure``
    for transport errors; ``APIStatusError`` propagates for non-2xx replies.
    """
    content: list[str] = []
    reasoning: list[str] = []
    usage = None
    with client.chat.completions.with_streaming_response.create(**kwargs) as stream:
        sid = _register_stream(scope, stream)
        try:
            if abort_requested(scope):
                raise RequestAborted("aborted before first token")
            for line in stream.iter_lines():
                if abort_requested(scope):
                    raise RequestAborted("aborted mid-stream")
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    chunk = json.loads(payload)
                except ValueError:
                    continue
                u = chunk.get("usage")
                if u:
                    usage = SimpleNamespace(
                        prompt_tokens=u.get("prompt_tokens", 0),
                        completion_tokens=u.get("completion_tokens", 0),
                    )
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    content.append(piece)
                rpiece = delta.get("reasoning_content") or delta.get("reasoning")
                if rpiece:
                    reasoning.append(rpiece)
        except (RequestAborted, GenerationTimeout):
            raise
        except httpx_backend.ConnectTimeout as e:
            if abort_requested(scope):
                raise RequestAborted(f"aborted mid-stream: {e}") from e
            raise ConnectionFailure(f"connect timeout: {e}") from e
        except httpx_backend.TimeoutException as e:
            if abort_requested(scope):
                raise RequestAborted(f"aborted mid-stream: {e}") from e
            # Read timeout = the engine stopped producing tokens; re-sending
            # the same generation would only burn more GPU.
            raise GenerationTimeout(str(e)) from e
        except httpx_backend.TransportError as e:
            if abort_requested(scope):
                raise RequestAborted(f"aborted mid-stream: {e}") from e
            raise ConnectionFailure(str(e)) from e
        except Exception as e:
            # Closing the stream from the aborting thread surfaces here as a
            # transport error; report it as the abort it actually is.
            if abort_requested(scope):
                raise RequestAborted(f"aborted mid-stream: {e}") from e
            raise
        finally:
            _unregister_stream(scope, sid)
            try:
                stream.close()
            except Exception:
                pass
    return "".join(content).strip(), "".join(reasoning), usage


def call_model(
    client: OpenAI,
    model_name: str,
    messages: list[dict],
    mode: str,
    is_deepseek: bool,
    max_tokens: int | None = None,
    max_retries: int = 1,
    temperature: float | None = None,
    reasoning_effort=None,
    thinking_spec: dict | None = None,
    scope: str | None = None,
) -> tuple[str, str, object | None]:
    """Call a chat model with mode-appropriate thinking settings.

    ``temperature`` defaults the request to greedy decoding (0.0); pass a
    value > 0 to sample instead. It is only applied where the mode already
    sends a temperature (cot via Deepseek APIs relies on server defaults).

    Streaming + retry policy (see the "in-flight request control" section):
      * timeouts are never retried: a generation that outruns the serving
        capacity would be regenerated from scratch, wasting the very GPU
        time that made it slow;
      * only connection failures are retried, at most ``max_retries`` times;
      * ``scope`` ties the request to a bench so that a bench-scoped abort
        can close it (``None`` = reachable only by the whole-run abort).

    Returns ``(raw_response, reasoning, usage)``.
    """
    if abort_requested(scope):
        raise RequestAborted("aborted before request start")

    kwargs: dict = dict(
        model=model_name, messages=messages, seed=42, stream=True,
    )
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

    if _stream_options_ok.get(str(client.base_url), True):
        kwargs["stream_options"] = {"include_usage": True}
    if _stream_interval_ok.get(str(client.base_url), True):
        kwargs.setdefault("extra_body", {})["stream_interval"] = _STREAM_INTERVAL

    attempts = max_retries + 1
    for attempt in range(attempts):
        try:
            return _stream_completion(client, kwargs, scope)
        except APIStatusError as e:
            if (e.status_code == 400 and "stream_options" in kwargs
                    and "stream_options" in str(e)):
                # Engine rejects per-stream usage: drop it and keep going.
                kwargs.pop("stream_options", None)
                _stream_options_ok[str(client.base_url)] = False
                continue
            if (e.status_code == 400
                    and "stream_interval" in kwargs.get("extra_body", {})):
                # Endpoint without per-request stream_interval (non-vLLM):
                # drop it and retry, we just lose the chunk coalescing.
                kwargs["extra_body"].pop("stream_interval", None)
                _stream_interval_ok[str(client.base_url)] = False
                continue
            raise
        except (RequestAborted, GenerationTimeout):
            raise
        except (APIConnectionError, ConnectionFailure) as e:  # transport
            if attempt + 1 >= attempts:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("call_model: retry loop fell through")


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
        if abort_requested():
            return None
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
