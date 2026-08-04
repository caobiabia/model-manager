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
import time
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


app = FastAPI(title="CSP Model Manager")

_procs: dict[str, subprocess.Popen] = {}
_log_queues: dict[str, Queue] = {}
_start_times: dict[str, str] = {}
_log_history: dict[str, list[str]] = {}  # accumulated log lines
_locks: dict[str, asyncio.Lock] = {}     # per-model lock for launch / stop
_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
_desc_lock = threading.Lock()


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

        uuid2idx: dict[str, int] = {}
        for line in _run([
            "nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits",
        ]).strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[0].isdigit():
                uuid2idx[parts[1]] = int(parts[0])

        apps: list[tuple[str, int, int]] = []
        for line in _run([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ]).strip().splitlines():
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
        default_ml = cfg.get("vllm_args", {}).get("max-model-len", 16384)
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
                 "--query-gpu=index,name,memory.total,memory.used,memory.free,"
                 "utilization.gpu,temperature.gpu,power.draw,power.limit",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            ),
        )
        process_map = await loop.run_in_executor(_thread_pool, _gpu_process_details)
        gpus: list[dict[str, Any]] = []
        for line in r.stdout.strip().split("\n"):
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


# ——— eval ———
# Ensure the local csp_dev/eval/ package is found before the
# workspace-level eval/ symlink (which points to the old eval code).
sys.path.insert(0, str(APP_DIR))
from eval import list_benches as _eval_list_benches  # noqa: E402
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
    )
    _eval_runs[run_id] = {"runner": runner, "status": "running"}

    def _run():
        try:
            runner.run()
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
    q = runner.queue

    async def gen():
        loop = asyncio.get_running_loop()
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await loop.run_in_executor(
                    _thread_pool, lambda: q.get(timeout=1),
                )
            except Empty:
                if entry["status"] != "running":
                    yield "data: [EOF]\n\n"
                    break
                continue
            except Exception:
                break
            if event is None:
                yield "data: [EOF]\n\n"
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

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
    return {"ok": True, "message": "Stop signal sent"}


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
    from eval.common import load_jsonl

    progress = {}
    for mk in model_keys:
        mc = _cfg().MODELS.get(mk, {})
        model_short = mc.get("served_model_name", mk).split("/")[-1]
        for bid in bench_ids:
            bench = get_bench(bid)
            bench_name = bench.name if bench else bid
            total = 0
            if bench and Path(bench.data_file).exists():
                total = len(load_jsonl(bench.data_file))

            # Count completed questions from the results file
            results_file = output_dir / f"{model_short}__{bid}__{mode}.json"
            done = 0
            correct = 0
            if results_file.exists():
                try:
                    results = json.loads(results_file.read_text(encoding="utf-8"))
                    # deduplicate by realidx
                    seen_ids = set()
                    unique = []
                    for r in results:
                        rid = r.get("realidx")
                        if rid not in seen_ids:
                            seen_ids.add(rid)
                            unique.append(r)
                    done = len(unique)
                    correct = sum(1 for r in unique if r.get("correct") is True)
                except (json.JSONDecodeError, Exception):
                    # results file may be corrupted (interrupted write)
                    # fall back to counting progress file entries
                    import re
                    pf = output_dir / f"progress_{model_short}_{bid}_{mode}.txt"
                    if pf.exists():
                        done = len(set(int(x) for x in pf.read_text().split() if x.isdigit()))
                    else:
                        done = 0
                    correct = 0

            if bench is not None and not bench.scorable:
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
    )
    # Initialize progress from disk BEFORE putting in _eval_runs,
    # so the GET /progress endpoint returns correct data immediately
    runner._init_progress_from_disk()
    _eval_runs[run_id] = {"runner": runner, "status": "running"}

    def _run():
        try:
            runner.run()
            _eval_runs[run_id]["status"] = "completed"
        except Exception as e:
            _eval_runs[run_id]["status"] = f"error: {e}"

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return {"ok": True, "run_id": run_id}


@app.get("/api/eval/runs")
async def eval_list_runs() -> list[dict[str, Any]]:
    """List all eval runs (from disk + in-memory)."""
    disk_runs = _eval_list_runs()
    for run in disk_runs:
        rid = run.get("run_id", "")
        if rid in _eval_runs:
            run["status"] = _eval_runs[rid]["status"]
        elif "status" not in run:
            run["status"] = "unknown"
    disk_ids = {r.get("run_id") for r in disk_runs}
    for rid, entry in _eval_runs.items():
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
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            detail_files.append({
                "file": f.name,
                "count": len(data) if isinstance(data, list) else 0,
            })
        except Exception:
            pass
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
