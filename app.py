#!/usr/bin/env python3
"""FastAPI model management & chat — foreground vLLM launch with live log streaming.
Auto-detects already-running models via port scanning."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import importlib
import os
import signal
import socket
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

_workspace = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_workspace))
import model_config  # noqa: E402   # type: ignore[import-untyped]

APP_DIR = Path(__file__).resolve().parent
VENV_PYTHON = str(_workspace / ".venv" / "bin" / "python")


def _cfg():
    """Reload model_config from disk so config edits apply live without a
    server restart. model_config.py lives in the parent workspace, outside
    uvicorn's reload watch dir, so we refresh on demand here instead."""
    importlib.reload(model_config)
    return model_config

app = FastAPI(title="CSP Model Manager")

_procs: dict[str, subprocess.Popen] = {}
_log_queues: dict[str, Queue] = {}
_start_times: dict[str, str] = {}
_log_history: dict[str, list[str]] = {}  # accumulated log lines
_locks: dict[str, asyncio.Lock] = {}     # per-model lock for launch / stop
_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)


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


# ——— API models ————

class LaunchRequest(BaseModel):
    gpu: int | None = None
    gpu_memory_utilization: float = 0.8
    max_model_len: int | None = None
    max_num_seqs: int = 256
    max_lora_rank: int = 64


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


# ——— helpers ————

def _get_lock(key: str) -> asyncio.Lock:
    if key not in _locks:
        _locks[key] = asyncio.Lock()
    return _locks[key]


_FLAG_LIKE_ARGS: frozenset[str] = frozenset({
    "trust-remote-code", "async-scheduling", "language-model-only",
    "enable-prefix-caching",
})


def _build_vllm_cmd(cfg: dict, gpu: int, gpu_memory_utilization: float,
                     max_model_len: int, max_num_seqs: int,
                     max_lora_rank: int = 64) -> list[str]:
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
            proc.terminate()
        except ProcessLookupError:
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

    port_alive: dict[str, bool] = {}
    probe_futures: dict[str, asyncio.Task[list[str]]] = {}
    for key, cfg in _cfg().MODELS.items():
        p = cfg.get("port")
        if p:
            port_alive[key] = await _check_port(p)
            if port_alive[key]:
                probe_futures[key] = asyncio.ensure_future(_probe(p))

    for key in probe_futures:
        probe_futures[key] = await probe_futures[key]

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
            "running": alive,
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
        result.append(entry)
    return result


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
        default_ml = cfg.get("vllm_args", {}).get("max-model-len", 12000)
        max_len = body.max_model_len or default_ml

        cmd = _build_vllm_cmd(
            cfg, gpu, body.gpu_memory_utilization, max_len, body.max_num_seqs,
            max_lora_rank=body.max_lora_rank,
        )

        env = os.environ.copy()
        venv_bin = str(_workspace / ".venv" / "bin")
        env["PATH"] = venv_bin + ":" + env.get("PATH", "")
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env,
            )
        except FileNotFoundError:
            raise HTTPException(500, "vLLM binary not found in .venv")
        except OSError as exc:
            raise HTTPException(500, f"Failed to start vLLM: {exc}")

        _procs[model_key] = proc
        _start_times[model_key] = datetime.now(timezone.utc).isoformat()
        q: Queue[Any] = Queue()
        _log_queues[model_key] = q
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
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        except ProcessLookupError:
            pass

    return {"ok": True, "message": f"Model {model_key} stopped"}


async def _kill_by_port(port: int) -> bool:
    """Kill the process listening on *port* using ss + kill. Returns True on success."""
    try:
        r = await asyncio.get_running_loop().run_in_executor(
            _thread_pool,
            lambda: subprocess.run(
                ["ss", "-tlnp"], capture_output=True, text=True, timeout=5,
            ),
        )
    except Exception:
        return False

    # Parse ss output for "LISTEN ... :<port> ... pid=<N>"
    for line in r.stdout.split("\n"):
        if f":{port}" not in line:
            continue
        # Extract pid from "pid=<N>" or "pid=<N>,fd=..."
        import re
        m = re.search(r"pid=(\d+)", line)
        if m:
            try:
                os.kill(int(m.group(1)), signal.SIGTERM)
                return True
            except (ProcessLookupError, PermissionError):
                pass
    return False


# ——— live log SSE ————

@app.get("/api/log/{model_key}/history")
async def get_log_history(model_key: str):
    hist = _log_history.get(model_key)
    if hist is None:
        return {"lines": []}
    return {"lines": list(hist)}


@app.get("/api/log/{model_key}")
async def stream_log(model_key: str, request: Request):
    q = _log_queues.get(model_key)
    if q is None:
        raise HTTPException(404, "No log stream")

    async def gen():
        loop = asyncio.get_running_loop()
        while True:
            if await request.is_disconnected():
                break
            try:
                # Poll with 1 s timeout so we can detect client disconnect
                line = await loop.run_in_executor(
                    _thread_pool, lambda: q.get(timeout=1),
                )
            except Empty:
                continue
            except Exception:
                break
            if line is None:
                yield "data: [EOF]\n\n"
                break
            yield f"data: {json.dumps({'text': line})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# ——— GPUs ————

@app.get("/api/gpus")
async def gpu_status() -> list[dict[str, Any]]:
    try:
        loop = asyncio.get_running_loop()
        r = await loop.run_in_executor(
            _thread_pool,
            lambda: subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,name,memory.total,memory.used,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            ),
        )
        gpus: list[dict[str, Any]] = []
        for line in r.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append({
                    "index": int(parts[0]), "name": parts[1],
                    "total_mb": int(parts[2]), "used_mb": int(parts[3]),
                    "free_mb": int(parts[4]),
                })
        return gpus
    except Exception as e:
        raise HTTPException(500, f"GPU query failed: {e}")


# ——— health ————

@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


# ——— chat ————

def _model_is_running(model_key: str) -> bool:
    """Return True if the vLLM model is running (tracked proc or port open)."""
    proc = _procs.get(model_key)
    if proc is not None and proc.poll() is None:
        return True
    cfg = _cfg().MODELS.get(model_key)
    if cfg and cfg.get("port"):
        return _port_open(cfg["port"])
    return False


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
        base_url = f"http://localhost:{cfg['port']}/v1"
        api_key = "not-needed"
    else:
        base_url = cfg.get("base_url", "")
        api_key = cfg.get("api_key", "")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    if req.stream:
        return StreamingResponse(
            _stream(base_url, api_key, cfg["served_model_name"],
                    messages, req.temperature, req.max_tokens,
                    enable_thinking=req.enable_thinking),
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
                        "chat_template_kwargs": {"enable_thinking": req.enable_thinking},
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
                  enable_thinking: bool = True):
    import httpx  # noqa: PLC0415
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "not-needed":
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with httpx.AsyncClient(timeout=600) as client:
            async with client.stream(
                "POST", f"{base_url}/chat/completions", headers=headers,
                json={
                    "model": model, "messages": messages,
                    "temperature": temperature, "max_tokens": max_tokens,
                    "stream": True,
                    "chat_template_kwargs": {"enable_thinking": enable_thinking},
                },
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


# ——— frontend ————

@app.get("/")
async def index():
    return HTMLResponse((APP_DIR / "templates" / "index.html").read_text())


if __name__ == "__main__":
    import uvicorn
    # reload watches csp_dev/ (app + template edits). Config edits apply
    # live via _cfg() above, so the workspace dir is intentionally NOT
    # watched (it holds training checkpoints/log churn that would spam
    # reloads and interrupt in-flight chat requests).
    uvicorn.run("app:app", host="0.0.0.0", port=27000, reload=True)
