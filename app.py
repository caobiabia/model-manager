#!/usr/bin/env python3
"""FastAPI model management & chat — foreground vLLM launch with live log streaming.
Auto-detects already-running models via port scanning."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import importlib
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

_workspace = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_workspace))
import model_config  # noqa: E402   # type: ignore[import-untyped]

APP_DIR = Path(__file__).resolve().parent
VENV_PYTHON = str(_workspace / ".venv" / "bin" / "python")
DESCRIPTION_FILE = APP_DIR / "model_descriptions.json"


def _cfg():
    """Reload model_config from disk so config edits apply live without a
    server restart. model_config.py lives in the parent workspace, outside
    uvicorn's reload watch dir, so we refresh on demand here instead."""
    importlib.reload(model_config)
    return model_config


def _load_descriptions() -> dict[str, str]:
    """Load per-model Chinese descriptions (kept outside shared model_config)."""
    try:
        if DESCRIPTION_FILE.exists():
            data = json.loads(DESCRIPTION_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): str(v).strip() for k, v in data.items() if str(v).strip()}
    except Exception:
        pass
    return {}


def _save_description(key: str, description: str) -> None:
    """Persist one model description atomically in the dev worktree."""
    with _desc_lock:
        data = _load_descriptions()
        description = description.strip()
        if description:
            data[key] = description
        else:
            data.pop(key, None)
        tmp = DESCRIPTION_FILE.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, DESCRIPTION_FILE)


USER_MODELS_FILE = getattr(model_config, "_USER_MODELS_FILE", APP_DIR / "models_user.json")


def _load_user_models() -> dict[str, dict]:
    """Load the user-editable model layer (models_user.json)."""
    cfg = _cfg()
    if hasattr(cfg, "load_user_models"):
        return cfg.load_user_models()
    return {}


def _save_user_models(data: dict[str, dict]) -> None:
    """Persist the whole user model layer atomically, then hot-reload."""
    USER_MODELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = USER_MODELS_FILE.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, USER_MODELS_FILE)
    _cfg()  # reload so MODELS reflects the change immediately


app = FastAPI(title="Model Console")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")

# Diagnosis hook: `kill -USR1 <worker-pid>` dumps every thread's Python stack
# into server.log. This process shares its GIL with the in-process eval
# runner, so when the UI goes unresponsive this is the only way to see which
# frame is blocking the event loop (ptrace tools like py-spy need root here).
try:
    import faulthandler

    faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
except Exception:  # pragma: no cover - diagnostic only
    pass

# Shorter GIL switch interval: with a big eval running in this process
# (hundreds of worker threads) the default 5ms quantum lets runner threads
# hold the GIL in long stretches, which showed up as multi-second latencies
# on every HTTP endpoint.
sys.setswitchinterval(0.001)

_procs: dict[str, subprocess.Popen] = {}
_log_queues: dict[str, Queue] = {}
_start_times: dict[str, str] = {}
_log_history: dict[str, list[str]] = {}  # accumulated log lines
_locks: dict[str, asyncio.Lock] = {}     # per-model lock for launch / stop
# Shared worker pool for *short* blocking probes (port/vLLM/GPU checks,
# launch/stop helpers). Long-lived waiters must never park here: the SSE
# log/eval streams used to hold a worker for ~1s per poll and starved the
# pool for everything else.
_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=16)
# SSE streams poll their queue at this interval (non-blocking); it is the
# delivery latency of log lines / eval events, not a resource cost.
SSE_POLL_INTERVAL = 0.05


class StreamHub:
    """Fan out one producer queue to every SSE client.

    The eval runner (and each vLLM log reader) pushes into a single queue.
    Each browser tab used to read that queue directly, so tabs *competed* for
    events: every event went to exactly one tab, and with several tabs open a
    given tab saw only a random slice of the progress -- the visible tab could
    sit at 0/total while the run was progressing fine. One pump thread now
    reads the source and copies each item into a per-connection queue, keeping
    a bounded replay buffer so a freshly opened tab starts with recent history
    instead of an empty view.
    """

    def __init__(self, source: Queue, replay: int = 400):
        self._source = source
        self._replay: deque = deque(maxlen=replay)
        self._subs: set[Queue] = set()
        self._finished = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def subscribe(self) -> Queue:
        q: Queue = Queue(maxsize=4000)
        with self._lock:
            for item in self._replay:
                try:
                    q.put_nowait(item)
                except Full:
                    break
            if self._finished:
                # Never leave a late subscriber hanging: hand it the EOF
                # sentinel immediately.
                try:
                    q.put_nowait(None)
                except Full:
                    pass
                return q
            self._subs.add(q)
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._pump, daemon=True, name="sse-hub",
                )
                self._thread.start()
        return q

    def unsubscribe(self, q: Queue) -> None:
        with self._lock:
            self._subs.discard(q)

    def _pump(self) -> None:
        while True:
            item = self._source.get()  # blocking; the hub's own thread
            with self._lock:
                self._replay.append(item)
                if item is None:  # EOF sentinel of both streams
                    self._finished = True
                subs = list(self._subs)
                if self._finished:
                    self._subs.clear()
            for sub in subs:
                try:
                    sub.put_nowait(item)
                except Full:
                    pass
            if self._finished:
                return


_log_hubs: dict[str, StreamHub] = {}
_desc_lock = threading.Lock()
_GPU_CACHE_TTL = 3.0  # seconds; /api/gpus is polled every 2s from the frontend
_gpu_cache: tuple[float, list[dict[str, Any]]] | None = None
_gpu_cache_lock = threading.Lock()


# ——— port / GPU scanning ————

def _port_open(port: int, timeout: float = 1.0) -> bool:
    s: socket.socket | None = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(("127.0.0.1", port))
        return True
    except Exception:
        return False
    finally:
        if s is not None:
            s.close()


def _probe_vllm(port: int) -> list[str]:
    """Return list of model IDs served on this port."""
    try:
        import httpx  # noqa: PLC0415
        with httpx.Client(timeout=3) as c:
            r = c.get(f"http://127.0.0.1:{port}/v1/models")
            if r.status_code == 200:
                return [m["id"] for m in r.json().get("data", [])]
    except Exception:
        pass
    return []


_probe_cache: dict[str, tuple[float, list[str]]] = {}
_probe_cache_lock = threading.Lock()
_PROBE_CACHE_TTL = 30.0  # seconds


def _probe_base_url(base_url: str) -> list[str]:
    """Return model IDs on a (possibly remote) OpenAI-compatible endpoint.

    Entries carrying an explicit base_url are served on another host, so
    localhost port scanning never sees them alive; probe /v1/models instead.

    Cached for _PROBE_CACHE_TTL: a dead remote endpoint costs a 3s read
    timeout, and the frontend polls /api/models from every open tab -- that
    3s used to be paid on every call, occupying a shared pool worker each
    time (which is what made the model dropdown hang)."""
    now = time.time()
    with _probe_cache_lock:
        hit = _probe_cache.get(base_url)
        if hit is not None and now - hit[0] < _PROBE_CACHE_TTL:
            return hit[1]
    found: list[str] = []
    try:
        import httpx  # noqa: PLC0415
        with httpx.Client(timeout=3) as c:
            r = c.get(f"{base_url.rstrip('/')}/models")
            if r.status_code == 200:
                found = [m["id"] for m in r.json().get("data", [])]
    except Exception:
        pass
    with _probe_cache_lock:
        _probe_cache[base_url] = (now, found)
    return found


def scan_all_ports() -> dict[str, dict]:
    """Scan every configured port; return {key: {port, alive, models}}."""
    result: dict[str, dict] = {}
    for key, cfg in _cfg().MODELS.items():
        if cfg.get("provider") == "deepseek":
            continue
        port = cfg.get("port")
        if not port:
            continue
        if _port_open(port):
            result[key] = {"port": port, "alive": True, "models": _probe_vllm(port)}
    return result


def _gpu_processes() -> dict[int, list[str]]:
    """Return {gpu_index: [process_name, ...]} from nvidia-smi pmon."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "pmon", "-c", "1", "-s", "u"],
            capture_output=True, text=True, timeout=10,
        )
        # Parse header: # gpu pid type sm mem enc dec command
        result: dict[int, list[str]] = {}
        for line in r.stdout.strip().split("\n"):
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                gpu = int(parts[0])
                cmd = parts[-1] if len(parts) > 1 else "?"
                if gpu not in result:
                    result[gpu] = []
                if cmd not in result[gpu]:
                    result[gpu].append(cmd)
        return result
    except Exception:
        return {}


def _match_model_cmdline(cmd: str) -> str | None:
    """Match a process command line to a configured model by its port."""
    cmd_l = cmd.lower()
    for key, cfg in _cfg().MODELS.items():
        if cfg.get("provider") == "deepseek":
            continue
        port = cfg.get("port")
        if port and (f"--port {port}" in cmd_l or f"--port={port}" in cmd_l):
            return key
    return None


def _gpu_process_details() -> dict[int, list[dict]]:
    """Return {gpu_index: [{pid, used_mb, command, model_key?, ...}]}."""
    try:
        def _run(args: list[str]) -> str:
            r = subprocess.run(args, capture_output=True, text=True, timeout=10)
            return r.stdout

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            uuid_out = pool.submit(_run, [
                "nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits",
            ]).result(timeout=12)
            apps_out = pool.submit(_run, [
                "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ]).result(timeout=12)

        uuid2idx: dict[str, int] = {}
        for line in uuid_out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[0].isdigit():
                uuid2idx[parts[1]] = int(parts[0])

        apps: list[tuple[str, int, int]] = []
        for line in apps_out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3 and parts[1].isdigit():
                used = int(parts[2]) if parts[2].isdigit() else 0
                apps.append((parts[0], int(parts[1]), used))

        cmdlines: dict[int, str] = {}
        pids = [pid for _, pid, _ in apps]
        if pids:
            out = _run(["ps", "-o", "pid=,args=", "-p", ",".join(map(str, pids))])
            for line in out.splitlines():
                line = line.strip()
                if not line:
                    continue
                sp = line.split(None, 1)
                if sp and sp[0].isdigit():
                    cmdlines[int(sp[0])] = sp[1] if len(sp) > 1 else ""

        pid2model: dict[int, str] = {}
        for key, proc in _procs.items():
            if proc.poll() is None:
                pid2model[proc.pid] = key

        def _parent_cmdline(pid: int) -> tuple[int, str]:
            try:
                with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as f:
                    data = f.read()
                rpar = data.rfind(")")
                fields = data[rpar + 2:].split()
                ppid = int(fields[1]) if len(fields) > 1 else 0
                with open(f"/proc/{ppid}/cmdline", "rb") as f:
                    cmd = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
                return ppid, cmd
            except Exception:
                return 0, ""

        for pid, cmd in cmdlines.items():
            if pid not in pid2model:
                key = _match_model_cmdline(cmd)
                if key:
                    pid2model[pid] = key
                else:
                    # vLLM GPU processes often show up as "VLLM::EngineCore";
                    # walk up to the parent whose cmdline names the port/model.
                    cur = pid
                    for _ in range(4):
                        ppid, pcmd = _parent_cmdline(cur)
                        if not ppid:
                            break
                        pkey = _match_model_cmdline(pcmd)
                        if pkey:
                            pid2model[pid] = pkey
                            break
                        cur = ppid

        result: dict[int, list[dict]] = {}
        for uuid, pid, used_mb in apps:
            idx = uuid2idx.get(uuid)
            if idx is None:
                continue
            entry: dict[str, Any] = {
                "pid": pid,
                "used_mb": used_mb,
                "command": cmdlines.get(pid, "")[:80],
            }
            key = pid2model.get(pid)
            if key:
                cfg = _cfg().MODELS.get(key, {})
                entry["model_key"] = key
                entry["display_name"] = cfg.get("display_name", key)
                entry["port"] = cfg.get("port")
            result.setdefault(idx, []).append(entry)
        return result
    except Exception:
        return {}


# ——— API models ————

class LaunchRequest(BaseModel):
    gpu: int | None = None
    gpu_memory_utilization: float = 0.8
    max_model_len: int | None = None
    max_num_seqs: int = 1024
    max_lora_rank: int = 64
    enforce_eager: bool = False


class DescriptionUpdate(BaseModel):
    description: str = ""


class ModelConfigWrite(BaseModel):
    key: str
    config: dict[str, Any]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model_key: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = True
    enable_thinking: bool = True
    # 思考强度: 档位 label (如 "low") 或原始数值 (如 37); None = 不传, 用模型默认
    # 合法值域由模型配置条目的 thinking 字段声明 (model_config.py)
    reasoning_effort: str | int | None = None


# ——— helpers ————

def _get_lock(key: str) -> asyncio.Lock:
    if key not in _locks:
        _locks[key] = asyncio.Lock()
    return _locks[key]


_FLAG_LIKE_ARGS: frozenset[str] = frozenset({
    "trust-remote-code", "async-scheduling", "language-model-only",
    "enable-prefix-caching",
    "enforce-eager",
})


def _build_vllm_cmd(cfg: dict, gpu: int, gpu_memory_utilization: float,
                     max_model_len: int, max_num_seqs: int,
                     max_lora_rank: int = 64,
                     enforce_eager: bool = False) -> list[str]:
    """Build vLLM CLI from model-config vllm_args, merged with request overrides."""
    # Determine model path: use base_model_path for LoRA, otherwise model_path
    model_src = cfg.get("base_model_path") or cfg.get("model_path")
    if model_src is None:
        raise ValueError("model config must have 'model_path' or 'base_model_path'")
    cmd = [
        str(_workspace / ".venv" / "bin" / "vllm"), "serve",
        str(_workspace / model_src),
    ]

    # Start with DEFAULT_VLLM_ARGS, then overlay model-specific vllm_args
    merged: dict[str, Any] = dict(_cfg().DEFAULT_VLLM_ARGS)
    merged.update(cfg.get("vllm_args", {}))

    # Request-body overrides
    merged["gpu-memory-utilization"] = gpu_memory_utilization
    merged["max-model-len"] = max_model_len
    merged["max-num-seqs"] = max_num_seqs
    merged["enforce-eager"] = enforce_eager

    # Ensure host / port / served-model-name
    merged["host"] = "0.0.0.0"
    merged["port"] = str(cfg["port"])
    merged["served-model-name"] = cfg["served_model_name"]

    # LoRA support: when lora_path is set, enable LoRA with the adapter
    lora_path = cfg.get("lora_path")
    if lora_path:
        merged["enable-lora"] = True
        lora_rank = cfg.get("max_lora_rank", max_lora_rank)
        merged["max-lora-rank"] = str(lora_rank)

    for k, v in merged.items():
        flag = "--" + k
        if k in _FLAG_LIKE_ARGS:
            if v:
                cmd.append(flag)
        elif v is True:
            cmd.append(flag)
        else:
            cmd.extend([flag, str(v)])

    # Append lora-modules after the main flag loop (keeps name=path intact)
    if lora_path:
        lora_name = cfg["served_model_name"]
        lora_abs = str(Path(lora_path)) if Path(lora_path).is_absolute() else str(_workspace / lora_path)
        cmd.extend(["--lora-modules", f"{lora_name}={lora_abs}"])

    return cmd


def _cleanup_model_state(key: str) -> None:
    """Remove all tracked state for a model key."""
    proc = _procs.pop(key, None)
    _log_queues.pop(key, None)
    _start_times.pop(key, None)
    _log_history.pop(key, None)
    if proc is not None and proc.poll() is None:
        try:
            _kill_pid_tree(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass


# ——— log reader thread ————

def _reader(key: str, pipe, q: Queue) -> None:
    """Read lines from subprocess stdout pipe; put them on the queue + history."""
    history = _log_history.setdefault(key, [])
    try:
        for line in iter(pipe.readline, ""):
            q.put(line)
            history.append(line)
    except (UnicodeDecodeError, OSError):
        # vLLM may emit non-UTF-8 bytes; skip the malformed line
        pass
    finally:
        try:
            pipe.close()
        except OSError:
            pass
        q.put(None)  # sentinel: EOF


# ——— /api/scan ————

@app.get("/api/scan")
async def scan_models() -> dict[str, Any]:
    """Scan all ports and GPUs to detect running models."""
    loop = asyncio.get_running_loop()
    ports = await loop.run_in_executor(_thread_pool, scan_all_ports)
    gpus = await loop.run_in_executor(_thread_pool, _gpu_processes)
    for key, info in ports.items():
        if info["alive"] and key not in _procs:
            _start_times.setdefault(key, "detected externally")
    return {"ports": ports, "gpu_processes": gpus}


# ——— /api/models ————

@app.get("/api/models")
async def list_models() -> list[dict[str, Any]]:
    loop = asyncio.get_running_loop()

    async def _check_port(p: int) -> bool:
        return await loop.run_in_executor(_thread_pool, _port_open, p)

    async def _probe(p: int) -> list[str]:
        return await loop.run_in_executor(_thread_pool, _probe_vllm, p)

    async def _probe_remote(u: str) -> list[str]:
        return await loop.run_in_executor(_thread_pool, _probe_base_url, u)

    port_alive: dict[str, bool] = {}
    probe_futures: dict[str, list[str]] = {}

    async def _probe_one(key: str, cfg: dict) -> tuple[str, bool, list[str]]:
        if cfg.get("base_url"):
            found = await _probe_remote(cfg["base_url"])
            return key, bool(found), found
        p = cfg.get("port")
        if not p:
            return key, False, []
        if not await _check_port(p):
            return key, False, []
        return key, True, await _probe(p)

    # Probe every configured model concurrently: the loop used to await each
    # model in turn, so one dead remote endpoint (3s timeout) delayed the
    # whole list.
    results = await asyncio.gather(
        *(_probe_one(k, c) for k, c in _cfg().MODELS.items()),
    )
    for key, alive, found in results:
        port_alive[key] = alive
        if found:
            probe_futures[key] = found

    descriptions = _load_descriptions()
    result: list[dict[str, Any]] = []
    for key in sorted(_cfg().MODELS.keys()):
        cfg = _cfg().MODELS[key]
        proc = _procs.get(key)
        has_proc = proc is not None and proc.poll() is None
        has_port = port_alive.get(key, False)
        alive = has_proc or has_port

        entry: dict[str, Any] = {
            "key": key,
            "display_name": cfg.get("display_name", key),
            "served_model_name": cfg.get("served_model_name", key),
            "provider": cfg.get("provider", "vllm"),
            "port": cfg.get("port"),
            "gpu": cfg.get("gpu"),
            "max_model_len": cfg.get("vllm_args", {}).get(
                "max-model-len", _cfg().DEFAULT_VLLM_ARGS.get("max-model-len", 262144)),
            "running": alive,
            "description": descriptions.get(key, ""),
        }
        if cfg.get("lora_path"):
            entry["lora_path"] = cfg["lora_path"]
            entry["base_model_path"] = cfg.get("base_model_path")
            entry["max_lora_rank"] = cfg.get("max_lora_rank", 64)
        if alive:
            if has_proc:
                entry["pid"] = proc.pid  # type: ignore[union-attr]
            elif has_port:
                entry["models_on_port"] = probe_futures.get(key, [])
            entry["started_at"] = _start_times.get(key, "")
        thinking = cfg.get("thinking")
        if isinstance(thinking, dict):
            entry["thinking"] = thinking  # 前端据此渲染思考强度控件
        result.append(entry)
    return result


@app.get("/api/model-descriptions")
async def get_model_descriptions() -> dict[str, str]:
    return _load_descriptions()


@app.post("/api/model-descriptions/{model_key}")
async def update_model_description(model_key: str, body: DescriptionUpdate) -> dict[str, Any]:
    if model_key not in _cfg().MODELS:
        raise HTTPException(404, f"Unknown model: {model_key}")
    description = body.description.strip()
    _save_description(model_key, description)
    return {"ok": True, "key": model_key, "description": description}


# ——— user model config (write through to model_config.MODELS) ———

_VALID_KEYS = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")


@app.get("/api/model-config/{model_key}")
async def get_model_config(model_key: str) -> dict[str, Any]:
    cfg = _cfg().MODELS.get(model_key)
    if cfg is None:
        raise HTTPException(404, f"Unknown model: {model_key}")
    user = _load_user_models()
    return {
        "key": model_key,
        "config": cfg,
        "is_user_defined": model_key in user,
    }


@app.post("/api/model-config")
async def write_model_config(body: ModelConfigWrite) -> dict[str, Any]:
    key = body.key.strip()
    if not key:
        raise HTTPException(400, "模型 key 不能为空")
    if not _VALID_KEYS.match(key):
        raise HTTPException(400, "模型 key 只能包含字母、数字、下划线、点和中划线，且不能以数字/符号开头")
    config = body.config or {}
    if not isinstance(config, dict):
        raise HTTPException(400, "config 必须是对象")
    if not config.get("display_name"):
        raise HTTPException(400, "缺少 display_name（展示名称）")
    if not config.get("served_model_name"):
        raise HTTPException(400, "缺少 served_model_name")

    with _desc_lock:  # serialize config writes
        user = _load_user_models()
        user[key] = config
        _save_user_models(user)
    return {"ok": True, "key": key, "is_user_defined": True}


@app.delete("/api/model-config/{model_key}")
async def delete_model_config(model_key: str) -> dict[str, Any]:
    with _desc_lock:
        user = _load_user_models()
        if model_key not in user:
            raise HTTPException(404, f"未找到用户级模型配置: {model_key}")
        del user[model_key]
        _save_user_models(user)
    restored = model_key in _cfg().MODELS
    return {"ok": True, "key": model_key, "restored_to_builtin": restored}


# ——— launch ————

@app.post("/api/launch/{model_key}")
async def launch_model(model_key: str, body: LaunchRequest = LaunchRequest()):
    if model_key not in _cfg().MODELS:
        raise HTTPException(404, f"Unknown model: {model_key}")
    cfg = _cfg().MODELS[model_key]
    if cfg.get("provider") == "deepseek":
        raise HTTPException(400, f"{model_key} is remote API, no vLLM")

    lock = _get_lock(model_key)
    async with lock:
        port = cfg["port"]

        # 1) already tracked by our process?
        proc = _procs.get(model_key)
        if proc is not None:
            if proc.poll() is None:
                return {
                    "ok": True,
                    "message": f"Already running (pid={proc.pid})",
                    "pid": proc.pid,
                }
            # Process died — clean up stale state before re-launching
            _cleanup_model_state(model_key)

        # 2) port already open? (launched outside our app)
        loop = asyncio.get_running_loop()
        if await loop.run_in_executor(_thread_pool, _port_open, port):
            models = await loop.run_in_executor(_thread_pool, _probe_vllm, port)
            return {
                "ok": True,
                "message": f"Port {port} already in use",
                "port": port,
                "models_on_port": models,
                "already_running": True,
            }

        # 3) GPU check: warn if GPU has compute processes
        gpu = body.gpu if body.gpu is not None else cfg.get("gpu", 0)
        gpu_procs = await loop.run_in_executor(_thread_pool, _gpu_processes)
        gpu_warning: str | None = None
        if gpu in gpu_procs:
            procs_str = ", ".join(gpu_procs[gpu])
            gpu_warning = f"GPU {gpu} has running processes: {procs_str}"

        # Build CLI from config (not hardcoded flags!)
        default_ml = cfg.get("vllm_args", {}).get(
            "max-model-len", _cfg().DEFAULT_VLLM_ARGS.get("max-model-len", 262144))
        max_len = body.max_model_len or default_ml

        cmd = _build_vllm_cmd(
            cfg, gpu, body.gpu_memory_utilization, max_len, body.max_num_seqs,
            max_lora_rank=body.max_lora_rank,
            enforce_eager=body.enforce_eager,
        )

        env = os.environ.copy()
        venv_bin = str(_workspace / ".venv" / "bin")
        env["PATH"] = venv_bin + ":" + env.get("PATH", "")
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env,
                start_new_session=True,
            )
        except FileNotFoundError:
            raise HTTPException(500, "vLLM binary not found in .venv")
        except OSError as exc:
            raise HTTPException(500, f"Failed to start vLLM: {exc}")

        _procs[model_key] = proc
        _start_times[model_key] = datetime.now(timezone.utc).isoformat()
        q: Queue[Any] = Queue()
        _log_queues[model_key] = q
        # A relaunch gets a fresh queue: drop the hub bound to the old one,
        # otherwise SSE clients would keep reading the dead queue.
        _log_hubs.pop(model_key, None)
        threading.Thread(
            target=_reader, args=(model_key, proc.stdout, q), daemon=True,
        ).start()

        resp: dict[str, Any] = {
            "ok": True, "key": model_key, "pid": proc.pid,
            "gpu": gpu, "port": port,
        }
        if gpu_warning:
            resp["warning"] = gpu_warning
        return resp


# ——— stop ————

@app.post("/api/stop/{model_key}")
async def stop_model(model_key: str):
    if model_key not in _cfg().MODELS:
        raise HTTPException(404, f"Unknown model: {model_key}")

    lock = _get_lock(model_key)
    async with lock:
        proc = _procs.pop(model_key, None)
        _log_queues.pop(model_key, None)
        _log_hubs.pop(model_key, None)
        _start_times.pop(model_key, None)
        _log_history.pop(model_key, None)

        if proc is None:
            # Not tracked — try to kill by port using ss (safer than lsof)
            port = _cfg().MODELS[model_key].get("port")
            if port and _port_open(port):
                killed = await _kill_by_port(port)
                if killed:
                    return {"ok": True, "message": f"Killed by port {port}"}
            return {"ok": False, "message": "Not running"}

        # Graceful terminate → force kill after timeout
        try:
            if proc.poll() is None:
                await asyncio.get_running_loop().run_in_executor(
                    _thread_pool, _stop_pid_tree_sync, proc.pid, 15,
                )
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
        except ProcessLookupError:
            pass

    return {"ok": True, "message": f"Model {model_key} stopped"}


def _descendant_pids(root: int) -> list[int]:
    """Return all descendant PIDs of *root* using ps."""
    try:
        r = subprocess.run(
            ["ps", "-eo", "pid=,ppid="], capture_output=True, text=True, timeout=5,
        )
        children: dict[int, list[int]] = {}
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                children.setdefault(int(parts[1]), []).append(int(parts[0]))
        out: list[int] = []
        stack = [root]
        while stack:
            p = stack.pop()
            for c in children.get(p, []):
                out.append(c)
                stack.append(c)
        return out
    except Exception:
        return []


def _pid_alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as f:
            data = f.read()
        # state is the field right after the ")" of the comm field
        state = data[data.rfind(")") + 2:].split()[0]
        if state == "Z":
            return False
    except (ProcessLookupError, FileNotFoundError, IndexError):
        return False
    except PermissionError:
        pass
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _kill_pid_tree(root: int, sig: int) -> None:
    """Send *sig* to a PID and all of its descendants."""
    pids = [root] + _descendant_pids(root)
    # If the process is its own group leader, signal the whole group too so
    # reparented children (e.g. VLLM::EngineCore) don't survive.
    try:
        if os.getpgid(root) == root:
            os.killpg(root, sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    for pid in pids:
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


def _stop_pid_tree_sync(root: int, grace: float = 5.0) -> None:
    """SIGTERM a process tree, wait, then SIGKILL anything still alive."""
    targets = [root] + _descendant_pids(root)
    _kill_pid_tree(root, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if not any(_pid_alive(p) for p in targets):
            return
        time.sleep(0.2)
    _kill_pid_tree(root, signal.SIGKILL)
    for pid in targets:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _kill_by_port_sync(port: int) -> bool:
    """Find the PID listening on *port* and stop its whole process tree."""
    try:
        r = subprocess.run(
            ["ss", "-tlnp"], capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return False

    import re
    for line in r.stdout.split("\n"):
        if f":{port}" not in line:
            continue
        # Extract pid from "pid=<N>" or "pid=<N>,fd=..."
        m = re.search(r"pid=(\d+)", line)
        if m:
            _stop_pid_tree_sync(int(m.group(1)), grace=5)
            return True
    return False


async def _kill_by_port(port: int) -> bool:
    """Kill the process listening on *port* (including EngineCore children)."""
    return await asyncio.get_running_loop().run_in_executor(
        _thread_pool, _kill_by_port_sync, port,
    )


# ——— live log SSE ————

@app.get("/api/log/{model_key}/history")
async def get_log_history(model_key: str):
    hist = _log_history.get(model_key)
    if hist is None:
        return {"lines": []}
    return {"lines": list(hist)}


@app.get("/api/log/{model_key}")
async def stream_log(model_key: str, request: Request):
    src = _log_queues.get(model_key)
    if src is None:
        raise HTTPException(404, "No log stream")
    hub = _log_hubs.get(model_key)
    if hub is None or hub._source is not src:
        hub = StreamHub(src)
        _log_hubs[model_key] = hub
    q = hub.subscribe()

    async def gen():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    # Non-blocking: each connection has its own queue, and
                    # blocking here (old code: run_in_executor(q.get) with a
                    # 1s timeout) parked a shared pool worker per poll.
                    line = q.get_nowait()
                except Empty:
                    await asyncio.sleep(SSE_POLL_INTERVAL)
                    continue
                except Exception:
                    break
                if line is None:
                    yield "data: [EOF]\n\n"
                    break
                yield f"data: {json.dumps({'text': line})}\n\n"
        finally:
            hub.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream")


# ——— GPUs ————

def _query_gpu_status() -> list[dict[str, Any]]:
    """Run the nvidia-smi summary and per-process queries concurrently."""
    def _main_query() -> str:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,memory.used,memory.free,"
             "utilization.gpu,temperature.gpu,power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        main_fut = pool.submit(_main_query)
        proc_fut = pool.submit(_gpu_process_details)
        out = main_fut.result(timeout=12)
        process_map = proc_fut.result(timeout=12)

    gpus: list[dict[str, Any]] = []
    for line in out.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue

        def _num(v: str) -> float | None:
            try:
                return float(v)
            except ValueError:
                return None

        idx = int(parts[0])
        gpus.append({
            "index": idx,
            "name": parts[1],
            "total_mb": int(parts[2]),
            "used_mb": int(parts[3]),
            "free_mb": int(parts[4]),
            "utilization_percent": _num(parts[5]),
            "temperature_c": _num(parts[6]),
            "power_w": _num(parts[7]),
            "power_limit_w": _num(parts[8]),
            "processes": process_map.get(idx, []),
        })
    return gpus


@app.get("/api/gpus")
async def gpu_status() -> list[dict[str, Any]]:
    global _gpu_cache
    now = time.monotonic()
    with _gpu_cache_lock:
        if _gpu_cache is not None and now - _gpu_cache[0] < _GPU_CACHE_TTL:
            return _gpu_cache[1]
    try:
        loop = asyncio.get_running_loop()
        gpus = await loop.run_in_executor(_thread_pool, _query_gpu_status)
    except Exception as e:
        raise HTTPException(500, f"GPU query failed: {e}")
    with _gpu_cache_lock:
        _gpu_cache = (time.monotonic(), gpus)
    return gpus


# ——— health ————

@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


# ——— chat ————

def _model_is_running(model_key: str) -> bool:
    """Return True if the vLLM model is running (tracked proc or endpoint alive)."""
    proc = _procs.get(model_key)
    if proc is not None and proc.poll() is None:
        return True
    cfg = _cfg().MODELS.get(model_key)
    if cfg and cfg.get("base_url"):
        return bool(_probe_base_url(cfg["base_url"]))
    if cfg and cfg.get("port"):
        return _port_open(cfg["port"])
    return False


def _thinking_kwargs(cfg: dict, enable_thinking: bool,
                     reasoning_effort: str | int | None) -> tuple[dict, dict]:
    """Build upstream thinking params from the model's `thinking` spec.

    Returns (chat_template_kwargs, extra_top_level_fields). reasoning_effort=None
    means "don't send the param, use the model default". Raises HTTPException(400)
    when the model declares no effort capability or the value is out of range.
    """
    # thinking / enable_thinking 双 key 同值下发: 不同模型的 chat template 变量名不一
    # 致, DeepSeek-V4.1 文档明确两 key 同发必须一致 — 同值总满足
    ctk: dict[str, Any] = {"enable_thinking": enable_thinking,
                           "thinking": enable_thinking}
    extra: dict[str, Any] = {}
    if reasoning_effort is None or not enable_thinking:
        return ctk, extra

    spec = cfg.get("thinking")
    if not isinstance(spec, dict):
        raise HTTPException(
            400, f"模型 {cfg.get('display_name', '?')} 不支持思考强度调节")

    efforts: dict = spec.get("efforts") or {}
    numeric: dict | None = spec.get("numeric")
    wire: Any = None
    if isinstance(reasoning_effort, str) and reasoning_effort in efforts:
        wire = efforts[reasoning_effort]
    elif numeric is not None:
        try:
            n = int(reasoning_effort)
        except (TypeError, ValueError):
            n = None
        if n is not None and numeric.get("min", 0) <= n <= numeric.get("max", 10**9):
            wire = n
    if wire is None:
        allowed = ", ".join(f"{k}({v})" if k != str(v) else k
                            for k, v in efforts.items())
        if numeric:
            allowed += f", 或 {numeric.get('min')}-{numeric.get('max')} 的整数"
        raise HTTPException(
            400,
            f"无效的思考强度 '{reasoning_effort}'。可选: {allowed or '无'}")

    if spec.get("transport", "chat_template_kwargs") == "top_level":
        extra["reasoning_effort"] = wire
    else:
        ctk["reasoning_effort"] = wire
    return ctk, extra


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if req.model_key not in _cfg().MODELS:
        raise HTTPException(404, f"Unknown model: {req.model_key}")
    cfg = _cfg().MODELS[req.model_key]
    provider = cfg.get("provider", "vllm")

    if provider == "vllm":
        if not _model_is_running(req.model_key):
            raise HTTPException(
                503,
                f"Model '{req.model_key}' is not running. Launch it first.",
            )
        base_url = cfg.get("base_url") or f"http://localhost:{cfg['port']}/v1"
        api_key = "not-needed"
    else:
        base_url = cfg.get("base_url", "")
        api_key = cfg.get("api_key", "")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    ctk, extra = _thinking_kwargs(cfg, req.enable_thinking, req.reasoning_effort)

    if req.stream:
        return StreamingResponse(
            _stream(base_url, api_key, cfg["served_model_name"],
                    messages, req.temperature, req.max_tokens,
                    chat_template_kwargs=ctk, extra_body=extra),
            media_type="text/event-stream",
        )
    else:
        import httpx  # noqa: PLC0415
        headers: dict[str, str] = {}
        if api_key and api_key != "not-needed":
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(
                    f"{base_url}/chat/completions", headers=headers,
                    json={
                        "model": cfg["served_model_name"], "messages": messages,
                        "temperature": req.temperature, "max_tokens": req.max_tokens,
                        "stream": False,
                        "chat_template_kwargs": ctk, **extra,
                    })
                if resp.status_code != 200:
                    raise HTTPException(resp.status_code, resp.text[:500])
                return resp.json()
        except HTTPException:
            raise
        except httpx.ConnectError:
            port = cfg.get("port", "?")
            raise HTTPException(503, f"Model {req.model_key} not running (port {port})")
        except Exception as e:
            raise HTTPException(500, f"Chat error: {e}")


async def _stream(base_url: str, api_key: str, model: str,
                  messages: list[dict], temperature: float, max_tokens: int,
                  chat_template_kwargs: dict | None = None,
                  extra_body: dict | None = None):
    import httpx  # noqa: PLC0415
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "not-needed":
        headers["Authorization"] = f"Bearer {api_key}"
    body: dict[str, Any] = {
        "model": model, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens,
        "stream": True,
    }
    if chat_template_kwargs:
        body["chat_template_kwargs"] = chat_template_kwargs
    if extra_body:
        body.update(extra_body)
    try:
        async with httpx.AsyncClient(timeout=600) as client:
            async with client.stream(
                "POST", f"{base_url}/chat/completions", headers=headers,
                json=body,
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    yield f"data: {json.dumps({'error': body.decode(errors='replace')})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        yield f"{line}\n\n"
    except httpx.ConnectError:
        yield f"data: {json.dumps({'error': 'Model not reachable — is it running?'})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as exc:
        yield f"data: {json.dumps({'error': f'Stream error: {exc}'})}\n\n"
        yield "data: [DONE]\n\n"


# ——— eval ———
# Ensure the local csp_dev/eval/ package is found before the
# workspace-level eval/ symlink (which points to the old eval code).
sys.path.insert(0, str(APP_DIR))
from eval import list_benches as _eval_list_benches  # noqa: E402
from eval.common import (  # noqa: E402
    count_jsonl_cached,
    result_stats_cached,
)
from eval.runner import EvalRunner, create_run_id, list_runs as _eval_list_runs  # noqa: E402
from eval import leaderboard as _eval_lb  # noqa: E402

# In-memory eval run tracking: run_id -> {"runner": EvalRunner, "status": str}
_eval_runs: dict[str, dict[str, Any]] = {}


class EvalRunRequest(BaseModel):
    model_keys: list[str]
    bench_ids: list[str]
    mode: str = "cot"  # "zero_shot" or "cot"
    workers: int = 10
    limit: int = 0
    no_resume: bool = False
    max_model_len: int | None = None  # None = auto from each model's config
    temperature: float | None = None  # None = greedy decoding (0.0)
    # 可选思考强度 (档位 label 或数值), 按各模型 thinking spec 解析;
    # None = 不发参数, 与历史跑分行为一致
    reasoning_effort: str | int | None = None


@app.get("/api/eval/benches")
async def eval_list_benches() -> list[dict[str, Any]]:
    """List all registered evaluation benches with metadata."""
    benches = []
    for b in _eval_list_benches():
        benches.append({
            "id": b.id,
            "name": b.name,
            "language": b.language,
            "split": b.split,
            "description": b.description,
            "format": b.format,
            "scorable": b.scorable,
            "data_file": str(b.data_file),
            "exists": Path(b.data_file).exists(),
        })
    return benches


@app.post("/api/eval/run")
async def eval_start_run(req: EvalRunRequest) -> dict[str, Any]:
    """Start an evaluation run in a background thread."""
    if not req.model_keys:
        raise HTTPException(400, "At least one model is required")
    if not req.bench_ids:
        raise HTTPException(400, "At least one bench is required")
    if req.mode not in ("zero_shot", "cot"):
        raise HTTPException(400, f"Invalid mode: {req.mode}")
    if req.temperature is not None and not (0.0 <= req.temperature <= 2.0):
        raise HTTPException(400, f"Invalid temperature: {req.temperature} (must be 0.0–2.0)")

    cfg = _cfg()
    for key in req.model_keys:
        if key not in cfg.MODELS:
            raise HTTPException(404, f"Unknown model: {key}")
        # Refuse to eval a vLLM model that isn't running
        model_cfg = cfg.MODELS[key]
        if model_cfg.get("provider", "vllm") != "deepseek":
            if not _model_is_running(key):
                raise HTTPException(
                    503,
                    f"Model '{key}' is not running. Launch it first.",
                )

    run_id = create_run_id()
    runner = EvalRunner(
        run_id=run_id,
        model_keys=req.model_keys,
        bench_ids=req.bench_ids,
        mode=req.mode,
        workers=req.workers,
        limit=req.limit,
        no_resume=req.no_resume,
        max_model_len=req.max_model_len,
        temperature=req.temperature,
        reasoning_effort=req.reasoning_effort,
    )
    _eval_runs[run_id] = {"runner": runner, "status": "running"}

    def _run():
        try:
            runner.run()
            # runner.run() returns normally even after a stop signal. Derive
            # the true end state from the runner itself: a user stop sets
            # stopped without the stop endpoint ever touching _eval_runs,
            # which previously let a stopped run be mis-labeled "completed"
            # in memory and then surface that way in the UI. (A bench that
            # keeps failing circuit-breaks on its own -- its unanswered
            # questions stay unscored so a resume retries them -- and does
            # not abort the rest of the run.)
            if _eval_runs[run_id]["status"] == "running":
                if runner.stopped:
                    _eval_runs[run_id]["status"] = "stopped"
                elif runner.error:
                    _eval_runs[run_id]["status"] = f"error: {runner.error}"
                else:
                    _eval_runs[run_id]["status"] = "completed"
        except Exception as e:
            _eval_runs[run_id]["status"] = f"error: {e}"

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {"ok": True, "run_id": run_id}


@app.get("/api/eval/run/{run_id}/stream")
async def eval_stream(run_id: str, request: Request):
    """SSE stream of eval progress events."""
    entry = _eval_runs.get(run_id)
    if entry is None:
        raise HTTPException(404, f"Unknown run: {run_id}")
    runner: EvalRunner = entry["runner"]
    hub = entry.get("hub")
    if hub is None or hub._source is not runner.queue:
        hub = StreamHub(runner.queue)
        entry["hub"] = hub
    q = hub.subscribe()

    async def gen():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    # Non-blocking, and per-connection: every open tab now
                    # sees every event instead of competing for one queue.
                    event = q.get_nowait()
                except Empty:
                    if entry["status"] != "running":
                        yield "data: [EOF]\n\n"
                        break
                    await asyncio.sleep(SSE_POLL_INTERVAL)
                    continue
                except Exception:
                    break
                if event is None:
                    yield "data: [EOF]\n\n"
                    break
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            hub.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/eval/run/{run_id}/stop")
async def eval_stop_run(run_id: str):
    """Signal a running eval to stop."""
    entry = _eval_runs.get(run_id)
    if entry is None:
        raise HTTPException(404, f"Unknown run: {run_id}")
    runner: EvalRunner = entry["runner"]
    runner.stop()
    entry["status"] = "stopped"
    return {
        "ok": True,
        "message": "Stop signal sent; in-flight generations aborted",
    }


def _progress_from_disk(run_id: str) -> dict | None:
    """Reconstruct progress for a run from its results/progress files."""
    output_dir = Path(__file__).resolve().parent / "eval" / "output" / run_id
    if not output_dir.exists():
        return None
    config_file = output_dir / "config.json"
    if not config_file.exists():
        return None
    config = json.loads(config_file.read_text(encoding="utf-8"))
    model_keys = config.get("model_keys", [])
    bench_ids = config.get("bench_ids", [])
    mode = config.get("mode", "cot")

    from eval.benches import get_bench

    progress = {}
    for mk in model_keys:
        mc = _cfg().MODELS.get(mk, {})
        model_short = mc.get("served_model_name", mk).split("/")[-1]
        for bid in bench_ids:
            bench = get_bench(bid)
            bench_name = bench.name if bench else bid
            total = 0
            if bench and Path(bench.data_file).exists():
                # Cached line count: parsing a 60MB JSONL just to know how
                # many questions it holds stalled the event loop on every
                # history/progress refresh.
                total = count_jsonl_cached(bench.data_file)

            # Count completed questions from the results file (cheaply)
            results_file = output_dir / f"{model_short}__{bid}__{mode}.json"
            done = 0
            correct = None
            if results_file.exists():
                done, correct = result_stats_cached(results_file)

            if bench is not None and not bench.scorable:
                acc = None
            elif correct is None:
                # Could not be counted without parsing a huge file: show
                # "—" instead of a wrong number.
                acc = None
            else:
                acc = round(correct / done * 100, 1) if done > 0 else 0
            key = f"{mk}|{bid}"
            progress[key] = {
                "model_key": mk,
                "bench_id": bid,
                "bench_name": bench_name,
                "total": total,
                "done": done,
                "correct": correct,
                "accuracy": acc,
                "status": "done" if (total > 0 and done >= total) else "interrupted",
            }
    return progress


@app.get("/api/eval/run/{run_id}/progress")
async def eval_get_progress(run_id: str):
    """Get current progress snapshot.

    For active runs, prefer the runner's in-memory progress dict and fall
    back to reconstructing from disk (covers runs started before a server
    reload or whose in-memory snapshot is not yet populated).
    """
    entry = _eval_runs.get(run_id)
    if entry is not None:
        runner: EvalRunner = entry["runner"]
        mem = runner.get_progress()
        if mem:
            return {"run_id": run_id, "progress": mem}
        progress = _progress_from_disk(run_id)
        if progress is not None:
            return {"run_id": run_id, "progress": progress}
        return {"run_id": run_id, "progress": {}}

    progress = _progress_from_disk(run_id)
    if progress is None:
        raise HTTPException(404, f"Unknown run: {run_id}")
    return {"run_id": run_id, "progress": progress}


@app.post("/api/eval/run/{run_id}/resume")
async def eval_resume_run(run_id: str):
    """Resume an interrupted run -- reuses the same run_id so progress
    files and partial results are picked up automatically."""
    safe = Path(run_id).name
    if safe != run_id:
        raise HTTPException(404, f"Unknown run: {run_id}")
    output_dir = Path(__file__).resolve().parent / "eval" / "output" / run_id
    config_file = output_dir / "config.json"
    if not config_file.exists():
        raise HTTPException(404, f"Run not found: {run_id}")
    if run_id in _eval_runs and _eval_runs[run_id]["status"] == "running":
        raise HTTPException(400, "Run is already active")

    config = json.loads(config_file.read_text(encoding="utf-8"))
    runner = EvalRunner(
        run_id=run_id,
        model_keys=config["model_keys"],
        bench_ids=config["bench_ids"],
        mode=config.get("mode", "cot"),
        workers=config.get("workers", 10),
        limit=config.get("limit", 0),
        no_resume=False,
        max_model_len=config.get("max_model_len"),
        temperature=config.get("temperature"),
        reasoning_effort=config.get("reasoning_effort"),
    )
    # Initialize progress from disk BEFORE putting in _eval_runs,
    # so the GET /progress endpoint returns correct data immediately
    runner._init_progress_from_disk()
    _eval_runs[run_id] = {"runner": runner, "status": "running"}

    def _run():
        try:
            runner.run()
            # runner.run() returns normally even after a stop signal. Derive
            # the true end state from the runner itself: a user stop sets
            # stopped without the stop endpoint ever touching _eval_runs,
            # which previously let a stopped run be mis-labeled "completed"
            # in memory and then surface that way in the UI. (A bench that
            # keeps failing circuit-breaks on its own -- its unanswered
            # questions stay unscored so a resume retries them -- and does
            # not abort the rest of the run.)
            if _eval_runs[run_id]["status"] == "running":
                if runner.stopped:
                    _eval_runs[run_id]["status"] = "stopped"
                elif runner.error:
                    _eval_runs[run_id]["status"] = f"error: {runner.error}"
                else:
                    _eval_runs[run_id]["status"] = "completed"
        except Exception as e:
            _eval_runs[run_id]["status"] = f"error: {e}"

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return {"ok": True, "run_id": run_id}


_runs_scan_cache: tuple[float, list[dict[str, Any]]] | None = None
_runs_scan_lock = threading.Lock()
_RUNS_SCAN_TTL = 5.0  # seconds


def _eval_list_runs_cached() -> list[dict[str, Any]]:
    """Disk scan of eval runs, cached for _RUNS_SCAN_TTL.

    Reading + parsing every summary.json is synchronous disk work on the
    event loop; the frontend polls the run list from every open tab every
    5s, so without the cache a big history (the 84-pair summary alone is
    ~60KB) was re-read constantly and delayed unrelated endpoints.
    """
    global _runs_scan_cache
    now = time.time()
    with _runs_scan_lock:
        if _runs_scan_cache is not None and now - _runs_scan_cache[0] < _RUNS_SCAN_TTL:
            return _runs_scan_cache[1]
    runs = _eval_list_runs()
    with _runs_scan_lock:
        _runs_scan_cache = (now, runs)
    return runs


@app.get("/api/eval/runs")
async def eval_list_runs() -> list[dict[str, Any]]:
    """List all eval runs (from disk + in-memory)."""
    # Shallow copies: the caller mutates per-run dicts (status merge) and
    # must not corrupt the cached snapshot.
    disk_runs = [dict(r) for r in _eval_list_runs_cached()]
    for run in disk_runs:
        rid = run.get("run_id", "")
        if rid in _eval_runs:
            run["status"] = _eval_runs[rid]["status"]
        elif "status" not in run:
            run["status"] = "unknown"
    disk_ids = {r.get("run_id") for r in disk_runs}
    # Snapshot + deferred pops: mutating _eval_runs mid-iteration raises
    # RuntimeError, and runner threads also write to it concurrently.
    for rid, entry in list(_eval_runs.items()):
        if rid not in disk_ids:
            # Only show in-memory runs that are still actively running;
            # skip phantom entries whose output dirs were deleted.
            if entry["status"] == "running":
                disk_runs.append({"run_id": rid, "status": entry["status"]})
            else:
                _eval_runs.pop(rid, None)
    disk_runs.sort(key=lambda r: r.get("run_id", ""), reverse=True)
    return disk_runs


@app.get("/api/eval/run/{run_id}")
async def eval_get_run(run_id: str) -> dict[str, Any]:
    """Get full details of a specific eval run (config + summary)."""
    output_dir = Path(__file__).resolve().parent / "eval" / "output" / run_id
    if not output_dir.exists():
        entry = _eval_runs.get(run_id)
        if entry is None:
            raise HTTPException(404, f"Unknown run: {run_id}")
        return {"run_id": run_id, "status": entry["status"], "results": []}

    result: dict[str, Any] = {"run_id": run_id}
    config_file = output_dir / "config.json"
    if config_file.exists():
        result["config"] = json.loads(config_file.read_text(encoding="utf-8"))
    summary_file = output_dir / "summary.json"
    if summary_file.exists():
        try:
            result["summary"] = json.loads(summary_file.read_text(encoding="utf-8"))
            result["status"] = result["summary"].get("status", "completed")
        except (json.JSONDecodeError, OSError):
            # Empty/corrupt summary (e.g. the final write failed when the
            # disk was full) -- fall back to the no-summary path instead
            # of returning a 500 that breaks the frontend.
            result["summary_error"] = (
                f"summary.json exists but is not readable "
                f"({summary_file.stat().st_size} bytes)"
            )
            summary_file = None
    else:
        summary_file = None
    if summary_file is None:
        entry = _eval_runs.get(run_id)
        if entry:
            result["status"] = entry["status"]
        else:
            json_files = list(output_dir.glob("*.json"))
            result["status"] = "interrupted" if json_files else "pending"
            result["partial_files"] = [f.name for f in json_files]

    detail_files = []
    for f in sorted(output_dir.glob("*__*.json")):
        if f.name in ("config.json", "summary.json"):
            continue
        # Cheap count -- this used to json.loads every result file, which
        # froze the whole console (minutes of blocked event loop) once a
        # live run's files grew into the tens of MB.
        count, _correct = result_stats_cached(f)
        detail_files.append({"file": f.name, "count": count})
    result["detail_files"] = detail_files
    return result


@app.get("/api/eval/run/{run_id}/detail/{filename}")
async def eval_get_detail(run_id: str, filename: str):
    """Get the full per-(model,bench) results JSON for a run."""
    output_dir = Path(__file__).resolve().parent / "eval" / "output" / run_id
    safe_name = Path(filename).name
    detail_file = output_dir / safe_name
    if not detail_file.exists() or not safe_name.endswith(".json"):
        raise HTTPException(404, f"File not found: {filename}")
    return json.loads(detail_file.read_text(encoding="utf-8"))


@app.delete("/api/eval/run/{run_id}")
async def eval_delete_run(run_id: str):
    """Delete an eval run and its output files."""
    output_dir = Path(__file__).resolve().parent / "eval" / "output" / run_id
    safe = Path(run_id).name
    if safe != run_id:
        raise HTTPException(404, f"Unknown run: {run_id}")
    import shutil
    if output_dir.exists():
        shutil.rmtree(output_dir)
    _eval_runs.pop(run_id, None)
    return {"ok": True, "run_id": run_id}


# -- leaderboard --

@app.get("/api/eval/leaderboard")
async def eval_leaderboard() -> dict[str, Any]:
    return _eval_lb.get_leaderboard()


@app.post("/api/eval/leaderboard/import/{run_id}")
async def eval_lb_import(run_id: str) -> dict[str, Any]:
    output_dir = Path(__file__).resolve().parent / "eval" / "output" / run_id
    if not output_dir.exists():
        raise HTTPException(404, f"Unknown run: {run_id}")
    return _eval_lb.import_run(run_id, output_dir)


@app.delete("/api/eval/leaderboard/{entry_id}")
async def eval_lb_delete(entry_id: str):
    ok = _eval_lb.delete_entry(entry_id)
    if not ok:
        raise HTTPException(404, "Entry not found")
    return {"ok": True}


@app.delete("/api/eval/leaderboard")
async def eval_lb_clear():
    n = _eval_lb.clear_all()
    return {"ok": True, "deleted": n}


class LbExportGroup(BaseModel):
    category: str
    benches: list[str] = []
    models: list[str] = []


class LbExportRequest(BaseModel):
    groups: list[LbExportGroup]
    avg_mode: str = "simple"
    max_model_len: int | None = None


@app.post("/api/eval/leaderboard/export")
async def eval_lb_export(req: LbExportRequest) -> Response:
    """Export selected leaderboard groups (category / bench / model) to xlsx."""
    if req.avg_mode not in ("simple", "weighted"):
        raise HTTPException(400, "avg_mode 只能是 simple 或 weighted")
    groups = [
        {"category": g.category, "benches": g.benches, "models": g.models}
        for g in req.groups
    ]
    data = _eval_lb.export_to_xlsx(
        groups, avg_mode=req.avg_mode, max_model_len=req.max_model_len
    )
    if not data:
        raise HTTPException(
            400,
            "没有可导出的数据：请至少选择一个榜单组及其中的 bench 和模型",
        )
    filename = f"leaderboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return Response(
        data,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )




# ——— frontend ————

@app.get("/")
async def index():
    return HTMLResponse((APP_DIR / "templates" / "index.html").read_text())


if __name__ == "__main__":
    import uvicorn
    # reload watches app + templates. Config edits apply live via _cfg()
    # above, so model_config.py is excluded — editing it used to trigger a
    # graceful reload that then hung forever waiting on the frontend's
    # never-closing SSE streams (logs/chat), taking port 27000 down.
    # timeout_graceful_shutdown caps that wait for any remaining real reloads.
    uvicorn.run(
        "app:app", host="0.0.0.0", port=27000, reload=True,
        reload_excludes=["model_config.py"],
        timeout_graceful_shutdown=5,
    )
