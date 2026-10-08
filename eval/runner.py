"""Unified evaluation runner.

Processes the cartesian product of *models x benches* sequentially.
Within each (model, bench) pair, questions are answered concurrently via
a thread pool with **rolling submission** (only workers+1 futures alive
at any time). This makes stop responsive: setting the stop flag prevents
new futures from being submitted and in-flight ones check the flag.

Resume: if a results file already exists for a (model, bench, mode) triple,
already-processed question indices are skipped.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Lock
from typing import Any

from eval.common import (
    EVAL_OUTPUT_DIR,
    GenerationTimeout,
    RequestAborted,
    abort_requested,
    call_model,
    clear_abort,
    create_client,
    extract_answer,
    extract_free_answer,
    get_extract_model,
    is_deterministic_failure,
    request_abort,
    save_progress,
    strip_think,
)
from eval.benches import get_bench
from model_config import get_model_by_name

# Circuit-break a *bench* after this many consecutive infrastructure failures
# (timeouts / connection loss -- the endpoint is unreachable or every request
# stalls). The bench stops, its unprocessed questions stay unanswered on disk
# so a later resume retries them, and the run keeps going. It deliberately
# does NOT touch sibling benches: a long-CoT bench tripping its breaker used
# to zero out a model's whole MMLU/C-Eval run that was progressing fine.
MAX_CONSECUTIVE_ERRORS = 5

class EvalRunner:
    """Run one evaluation pass across models x benches."""

    def __init__(
        self,
        run_id: str,
        model_keys: list[str],
        bench_ids: list[str],
        mode: str = "cot",
        workers: int = 10,
        limit: int = 0,
        no_resume: bool = False,
        max_model_len: int | None = None,
        temperature: float | None = None,
        reasoning_effort=None,
    ):
        self.run_id = run_id
        self.model_keys = model_keys
        self.bench_ids = bench_ids
        self.mode = mode
        self.workers = workers
        self.limit = limit
        self.no_resume = no_resume
        self.max_model_len = max_model_len  # optional override; None = auto per model
        self.temperature = temperature  # None = greedy decoding (0.0)
        # Optional thinking-effort request (label or number, per each model's
        # `thinking` spec); None = model-side default, preserving old behavior
        self.reasoning_effort = reasoning_effort

        self.queue: Queue[dict | None] = Queue()
        self._stop = threading.Event()

        self.output_dir = EVAL_OUTPUT_DIR / run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: dict[tuple[str, str], dict] = {}
        self.error: str | None = None
        # live progress snapshot, updated alongside queue events
        self.progress: dict[str, dict] = {}
        self._progress_lock = Lock()
        # (model_key, bench_id) -> reason, set when that bench circuit-breaks
        # (MAX_CONSECUTIVE_ERRORS consecutive infrastructure failures). Only
        # the failing bench stops; siblings keep running.
        self._dead_benches: dict[tuple[str, str], str] = {}
        self._dead_lock = Lock()

    def _mark_bench_dead(self, model_key: str, bench_id: str, reason: str) -> None:
        """Circuit-break one bench (first reason wins)."""
        with self._dead_lock:
            self._dead_benches.setdefault((model_key, bench_id), reason)

    def _bench_dead_reason(self, model_key: str, bench_id: str) -> str | None:
        with self._dead_lock:
            return self._dead_benches.get((model_key, bench_id))

    # -- public API -------------------------------------------------------

    def stop(self) -> None:
        """Stop the run *and* abort what is already generating.

        Setting the flag alone only discards results: the abandoned HTTP
        requests kept running on the engine (dozens of them generating for
        45+ minutes after a stop). Closing their streams makes vLLM abort
        the requests server-side.
        """
        self._stop.set()
        request_abort()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def model_max_len(self, model_key: str) -> int | None:
        """Context length for a model: explicit override if set, else the
        value configured in vllm_args (auto, plan A). Falls back to the
        launch default 16384 when the model doesn't declare one.

        API-variant configs (provider="deepseek" with base_url) don't carry a
        local vllm_args, so we inherit the context length from the local vLLM
        config that serves the same model (same served_model_name). Otherwise a
        262144 model would be recorded as the 16384 default."""
        if self.max_model_len is not None:
            return self.max_model_len
        cfg = get_model_by_name(model_key)
        if cfg:
            ml = cfg.get("vllm_args", {}).get("max-model-len")
            if ml:
                return ml
            # No max-model-len on this config: inherit from a sibling config
            # serving the same model that declares one.
            from model_config import MODELS
            served = cfg.get("served_model_name")
            if served:
                for other in MODELS.values():
                    if other is cfg:
                        continue
                    if other.get("served_model_name") == served:
                        ml = other.get("vllm_args", {}).get("max-model-len")
                        if ml:
                            return ml
            return 16384
        return None

    def _init_progress_from_disk(self) -> None:
        """Populate self.progress from existing files before starting.

        Called by the resume endpoint so get_progress() returns correct
        data immediately, before any questions are processed.
        """
        from eval.benches import get_bench
        from eval.common import (
            count_jsonl_cached,
            load_processed_ids,
            result_stats_cached,
        )

        for model_key in self.model_keys:
            cfg = get_model_by_name(model_key)
            if cfg is None:
                continue
            model_short = cfg.get("served_model_name", model_key).split("/")[-1]
            for bench_id in self.bench_ids:
                bench = get_bench(bench_id)
                if bench is None:
                    continue
                pf = str(self.output_dir / f"progress_{model_short}_{bench_id}_{self.mode}.txt")
                rf = self.output_dir / f"{model_short}__{bench_id}__{self.mode}.json"
                processed_ids = load_processed_ids(pf)
                total = 0
                if Path(bench.data_file).exists():
                    total = count_jsonl_cached(bench.data_file)
                done = min(len(processed_ids), total) if total > 0 else len(processed_ids)
                correct = 0
                if rf.exists():
                    # Cheap stats: never json.loads a multi-MB result file
                    # just to rebuild the initial progress display.
                    n, c = result_stats_cached(rf)
                    done = max(done, min(n, total)) if total > 0 else n
                    if c is not None:
                        correct = c
                if getattr(bench, "scorable", True):
                    acc = round(correct / done * 100, 1) if done > 0 else 0
                else:
                    acc = None
                key = f"{model_key}|{bench_id}"
                with self._progress_lock:
                    self.progress[key] = {
                        "model_key": model_key, "model": model_key, "bench_id": bench_id,
                        "bench_name": bench.name, "total": total,
                        "done": done, "correct": correct,
                        "accuracy": acc,
                        "status": "done" if (total > 0 and done >= total) else "interrupted",
                    }

    def run(self) -> None:
        """Main entry -- call from a background thread."""
        # Re-arm request submission: a stop of a previous run leaves the
        # global abort latched, and this runner may be a resume.
        clear_abort()
        try:
            self.queue.put({
                "type": "start",
                "run_id": self.run_id,
                "mode": self.mode,
                "temperature": self.temperature,
                "reasoning_effort": self.reasoning_effort,
                "model_keys": self.model_keys,
                "bench_ids": self.bench_ids,
                "total_pairs": len(self.model_keys) * len(self.bench_ids),
            })

            self._save_config()

            # MCQ answer extraction is model-only: resolve (and probe) the
            # extractor up front so an unreachable one fails the run right
            # away instead of scoring every MCQ answer as wrong. Runs made
            # only of free/follow benches don't need it at all -- the
            # inverse_ifeval judge resolves it lazily, when its bench runs.
            needs_extract = any(
                getattr(bench, "format", "mcq") == "mcq"
                for bench in (get_bench(b) for b in self.bench_ids)
                if bench is not None
            )
            if needs_extract:
                ext_client, ext_model, ext_deepseek = get_extract_model()
            else:
                ext_client, ext_model, ext_deepseek = None, None, False

            # Run all models in parallel -- each model gets its own
            # thread, and within each model all benches run concurrently.
            # This works well when models are on different GPUs.
            model_threads = []
            for model_key in self.model_keys:
                if self.stopped:
                    break
                t = threading.Thread(
                    target=self._run_model,
                    args=(model_key, ext_client, ext_model, ext_deepseek),
                )
                model_threads.append(t)
                t.start()
            for t in model_threads:
                t.join()

            self._save_summary()
            if self.error:
                self.queue.put({
                    "type": "error", "run_id": self.run_id,
                    "message": self.error,
                })
            else:
                self.queue.put({
                    "type": "done",
                    "run_id": self.run_id,
                    "results": [
                        {**v, "model_key": k[0], "bench_id": k[1]}
                        for k, v in self.results.items()
                    ],
                })
        except Exception as e:
            # Record the failure on the runner *and* on disk: self.error is
            # what the app reports as the run status, and without the summary
            # the run looked "completed" (with zero results) after a refresh,
            # so a failed run -- e.g. an unreachable answer extractor -- was
            # indistinguishable from a finished one.
            self.error = str(e)
            self._save_summary()
            self.queue.put({"type": "error", "run_id": self.run_id, "message": str(e)})
        finally:
            self.queue.put(None)  # sentinel for SSE EOF

    # -- per-model --------------------------------------------------------

    def _run_model(
        self,
        model_key: str,
        ext_client,
        ext_model: str,
        ext_deepseek: bool,
    ) -> None:
        cfg = get_model_by_name(model_key)
        if cfg is None:
            self.queue.put({
                "type": "skip", "model_key": model_key,
                "reason": f"Unknown model: {model_key}",
            })
            return

        client, model_name, is_deepseek = create_client(cfg)

        self.queue.put({
            "type": "model_start",
            "model_key": model_key,
            "model_name": cfg.get("display_name", model_key),
            "bench_ids": self.bench_ids,
        })

        # Run all benches concurrently, sharing a single per-model worker
        # budget. To avoid a lone trailing bench stalling at a tiny fixed
        # concurrency, the budget is *redistributed* whenever a bench
        # finishes: every still-running bench grows its concurrency, so the
        # last remaining bench (often the largest) eventually gets most of
        # the worker budget instead of being stuck at workers//n_benches.
        n_benches = len(self.bench_ids)
        alive = set(self.bench_ids)
        share = max(1, self.workers)
        quota = {
            bid: max(1, share // n_benches)
            for bid in alive
        }
        cap_lock = Lock()

        def get_quota(bid: str) -> int:
            with cap_lock:
                return quota.get(bid, 1)

        def reallocate(done_bid: str) -> None:
            with cap_lock:
                alive.discard(done_bid)
                quota.pop(done_bid, None)
                n = len(alive)
                if n == 0:
                    return
                base = share // n
                extra = share % n
                for i, bid in enumerate(sorted(alive)):
                    quota[bid] = max(1, base + (1 if i < extra else 0))

        threads = []
        for bench_id in self.bench_ids:
            if self.stopped:
                break
            t = threading.Thread(
                target=self._run_bench,
                args=(model_key, bench_id, client, model_name,
                      is_deepseek, ext_client, ext_model, ext_deepseek,
                      get_quota, reallocate, cfg.get("thinking")),
            )
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        # All benches of this model are done (or aborted): drop the HTTP
        # connections instead of leaving them around for the GC.
        try:
            client.close()
        except Exception:
            pass

    # -- per-(model, bench) ----------------------------------------------

    def _run_bench(
        self,
        model_key: str,
        bench_id: str,
        client,
        model_name: str,
        is_deepseek: bool,
        ext_client,
        ext_model: str | None,
        ext_deepseek: bool,
        get_quota,
        reallocate,
        thinking_spec: dict | None = None,
    ) -> None:
        bench = get_bench(bench_id)
        if bench is None:
            self.queue.put({"type": "skip", "model_key": model_key,
                            "bench_id": bench_id, "reason": "Unknown bench"})
            reallocate(bench_id)
            return

        if not Path(bench.data_file).exists():
            self.queue.put({
                "type": "skip", "model_key": model_key, "bench_id": bench_id,
                "reason": f"Data file not found: {bench.data_file}",
            })
            reallocate(bench_id)
            return

        questions = bench.load_questions()
        total = len(questions)

        model_short = model_name.split("/")[-1]
        results_file = str(
            self.output_dir / f"{model_short}__{bench_id}__{self.mode}.json"
        )
        progress_file = str(
            self.output_dir / f"progress_{model_short}_{bench_id}_{self.mode}.txt"
        )

        # Bench-scoped request control: lets this bench abort its own
        # in-flight requests (circuit-break) without touching siblings, and
        # re-arms requests when a previously aborted bench is resumed.
        scope = f"{model_key}|{bench_id}"
        clear_abort(scope)

        if self.no_resume:
            for f in [results_file, progress_file]:
                if os.path.exists(f):
                    os.remove(f)

        results = []
        if os.path.exists(results_file):
            try:
                with open(results_file, "r", encoding="utf-8") as f:
                    results = json.load(f)
                # deduplicate by realidx
                seen = set()
                deduped = []
                for r in results:
                    rid = r.get("realidx")
                    if rid not in seen:
                        seen.add(rid)
                        deduped.append(r)
                results = deduped
            except (json.JSONDecodeError, Exception):
                # corrupted file (interrupted write) -- start fresh
                results = []

        # The results file is authoritative for what has actually been
        # persisted. A realidx that is checkpointed in the progress file but
        # has no saved result (e.g. an in-flight future discarded when the
        # run was stopped, or a result lost in an unflushed batch when the
        # process died) must be RETRIED, not skipped -- otherwise the run
        # can never reach total/total.
        processed_ids = {str(r.get("realidx")) for r in results}

        to_process: list[tuple[int, dict]] = []
        for i, q in enumerate(questions):
            realidx = q.get("realidx", i)
            if str(realidx) in processed_ids:
                continue
            if self.limit > 0 and len(to_process) >= self.limit:
                break
            to_process.append((i, q))

        initial_done = min(len(processed_ids), total) if total > 0 else len(processed_ids)
        initial_correct = sum(1 for r in results if r.get("correct") is True)
        initial_acc = (
            round(initial_correct / initial_done * 100, 2)
            if initial_done else None
        )
        pk = f"{model_key}|{bench_id}"
        with self._progress_lock:
            self.progress[pk] = {
                "model_key": model_key, "model": model_key,
                "bench_id": bench_id, "bench_name": bench.name,
                "total": total, "done": initial_done,
                "correct": initial_correct, "accuracy": initial_acc,
                "status": "running",
            }

        self.queue.put({
            "type": "pair_start",
            "model_key": model_key, "bench_id": bench_id,
            "bench_name": bench.name, "total": total,
            "done": initial_done,
            "resume_skipped": total - len(to_process),
        })

        if not to_process:
            # All already processed (resume)
            correct = sum(1 for r in results if r.get("correct") is True)
            n = len(results)
            acc = round(correct / n * 100, 2) if n else None
            self.results[(model_key, bench_id)] = {
                "model_key": model_key, "bench_id": bench_id,
                "bench_name": bench.name, "mode": self.mode,
                "max_model_len": self.model_max_len(model_key),
                "processed": len(results), "correct": correct,
                "accuracy": acc, "time_elapsed": 0,
                "results_file": results_file,
                "aborted": False, "unreachable": False,
                "unattempted": 0, "failed_pending": 0,
            }
            with self._progress_lock:
                pk = f"{model_key}|{bench_id}"
                if pk in self.progress:
                    self.progress[pk].update({
                        "done": len(results), "correct": correct,
                        "accuracy": acc, "status": "done",
                    })
            self.queue.put({
                "type": "pair_done", "model_key": model_key,
                "bench_id": bench_id, "bench_name": bench.name,
                "processed": len(results), "correct": correct,
                "accuracy": acc, "time": 0,
                "aborted": False,
            })
            reallocate(bench_id)
            return

        completed = len(processed_ids)
        start_time = time.time()
        consecutive_errors = 0
        aborted = False
        # Set when this bench circuit-broke. Its unanswered questions are
        # left out of the results file, so a later resume retries them.
        unreachable = False
        unreachable_error = ""
        # Questions whose request failed for infra reasons (timeout /
        # connection loss): attempted but deliberately not scored, because a
        # fabricated 0 is indistinguishable from a wrong answer.
        failed_pending = 0

        # Per-(model, bench) persistence. Each bench writes only its own
        # results list/file, and only from this thread, so no shared global
        # lock is needed (the old self._write_lock serialized every bench's
        # disk write). Batched writes avoid doing an O(n^2) full rewrite+sorted
        # json.dump on every single question.
        write_lock = Lock()
        flush_every = 50
        pending = 0
        correct_count = initial_correct

        def flush_results() -> None:
            nonlocal pending
            if not results:
                return
            with write_lock:
                results.sort(key=lambda x: x.get("realidx", 0))
                # Atomic write: a crash mid-rewrite must not leave a
                # truncated results file (which would make resume lose the
                # whole bench).
                tmp_file = results_file + ".tmp"
                with open(tmp_file, "w", encoding="utf-8") as f:
                    # Compact JSON: indent=2 roughly doubles both the dump
                    # time (GIL held, web UI starves) and the file size of a
                    # 14k-question bench, for readability nobody reads.
                    json.dump(results, f, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_file, results_file)
            pending = 0

        def process_one(idx: int, problem: dict) -> dict | None:
            if self._stop.is_set() or abort_requested(scope):
                return None
            realidx = problem.get("realidx", idx)
            question = problem.get("question", "")
            options = problem.get("options", {})
            answer_idx = problem.get("answer_idx", "")
            is_free = getattr(bench, "format", "mcq") == "free"
            is_follow = getattr(bench, "format", "mcq") == "follow"
            gold = (
                str(problem.get("answer", answer_idx)).strip()
                if is_free else answer_idx
            )
            if not question:
                return None
            if not is_free and not is_follow and not options:
                return None

            messages = bench.build_messages(problem, self.mode)
            q_start = time.time()

            try:
                raw, reasoning, usage = call_model(
                    client, model_name, messages, self.mode, is_deepseek,
                    temperature=self.temperature,
                    reasoning_effort=self.reasoning_effort,
                    thinking_spec=thinking_spec,
                    scope=scope,
                )
            except RequestAborted:
                # Run/bench aborted: drop the question entirely so it stays
                # retryable instead of being recorded as answered.
                return None
            except GenerationTimeout as e:
                return {
                    "realidx": realidx, "pending": True,
                    "error": f"generation timeout: {e}",
                }
            except Exception as e:
                if is_deterministic_failure(e):
                    # 4xx (e.g. prompt exceeds the context): repeats forever,
                    # so record it as a real wrong answer to make progress.
                    return {
                        "realidx": realidx,
                        "error": f"request rejected: {e}", "correct": False,
                    }
                return {"realidx": realidx, "pending": True, "error": str(e)}

            # vLLM without --reasoning-parser returns the thinking inline in
            # the content; pull it out so extractors/verifiers only see the
            # final answer (it is kept in `reasoning` for inspection).
            emb_reason, raw = strip_think(raw)
            reasoning = reasoning or emb_reason

            correct = None
            correct_loose = None
            if is_follow:
                # No single extractable answer: correctness comes from the
                # per-bench scorer (IFEval/IFBench verifier or Inverse IFEval
                # LLM-judge), applied to the stripped final answer.
                pred = None
                scorer = getattr(bench, "scorer", None)
                if scorer is not None:
                    correct = bool(scorer(raw, problem))
                # Official IFEval reports a second, looser prompt-level metric;
                # record it alongside strict for ifeval/ifbench benches.
                scorer_loose = getattr(bench, "scorer_loose", None)
                if scorer_loose is not None:
                    correct_loose = bool(scorer_loose(raw, problem))
            elif is_free:
                pred = extract_free_answer(raw)
            else:
                pred = extract_answer(
                    raw, options, ext_client, ext_model, ext_deepseek,
                )

            if not is_follow and pred is not None:
                correct = (pred == gold)

            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            elapsed = time.time() - q_start

            result = {
                "realidx": realidx,
                "question": question,
                "options": options,
                "answer_idx": gold,
                "predicted_answer": pred or "",
                "correct": correct,
                "correct_loose": correct_loose,
                "raw_response": raw,
                "reasoning": reasoning,
                "token_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                },
                "time_elapsed": elapsed,
            }
            return result

        # -- rolling submission driven by a dynamic concurrency quota --
        # This makes stop responsive: new futures aren't submitted after
        # the stop flag is set, and process_one checks it at entry.
        # We manage the executor manually (not via ``with``) so that on
        # stop we can shutdown(wait=False) instead of blocking on
        # in-flight API calls that may take 30-60s each.
        # The executor's max_workers is sized to the *worst-case* quota
        # (threads are created lazily); the live concurrency is bounded by
        # get_quota(), which grows as sibling benches finish. So a trailing
        # bench picks up the freed workers instead of staying at a small
        # fixed number.
        executor = ThreadPoolExecutor(max_workers=max(1, self.workers))
        future_map: dict = {}
        next_idx = 0

        def _submit_more(quota: int) -> None:
            """Submit up to ``quota`` in-flight futures."""
            nonlocal next_idx
            while (not self.stopped and next_idx < len(to_process)
                   and len(future_map) < quota):
                idx, q = to_process[next_idx]
                future_map[executor.submit(process_one, idx, q)] = (
                    idx, q.get("realidx", idx))
                next_idx += 1

        # Pre-fill the pool with the current quota. A bench that circuit-broke
        # in an earlier run (resume) submits nothing.
        dead_reason = self._bench_dead_reason(model_key, bench_id)
        if dead_reason:
            aborted = True
            unreachable = True
            unreachable_error = dead_reason
        else:
            _submit_more(get_quota(bench_id))

        while future_map:
            # Wait for at least one future to complete (poll every 2s)
            done_set, _ = concurrent.futures.wait(
                future_map.keys(),
                timeout=2.0,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

            # Nothing completed within timeout — check stop
            if not done_set:
                if self.stopped:
                    break
                continue

            for future in done_set:
                i, realidx = future_map.pop(future)

                # Refill up to the (possibly grown) quota (never after an abort)
                if not self.stopped and not aborted:
                    _submit_more(get_quota(bench_id))

                try:
                    result = future.result()
                except Exception as e:
                    if is_deterministic_failure(e):
                        result = {
                            "realidx": realidx,
                            "error": f"request rejected: {e}", "correct": False,
                        }
                    else:
                        result = {
                            "realidx": realidx, "pending": True,
                            "error": str(e),
                        }
                completed += 1

                if result is None:
                    # Discarded (stop / abort): stays retryable on resume.
                    pass
                elif result.get("pending"):
                    # Infra failure: attempted but NOT scored. Nothing is
                    # written, so a resume retries the question instead of
                    # inheriting a fabricated zero.
                    failed_pending += 1
                    consecutive_errors += 1
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        unreachable_error = (
                            f"Bench '{bench_id}' stopped after "
                            f"{MAX_CONSECUTIVE_ERRORS} consecutive request "
                            f"failures. Last error: "
                            f"{result.get('error', 'unknown')}"
                        )
                        self._mark_bench_dead(
                            model_key, bench_id, unreachable_error,
                        )
                        # Close this bench's in-flight streams: the server
                        # then aborts those generations instead of finishing
                        # answers nobody will read.
                        request_abort(scope)
                        aborted = True
                        unreachable = True
                else:
                    results.append(result)
                    if result.get("correct") is True:
                        correct_count += 1
                    pending += 1
                    if pending >= flush_every:
                        flush_results()
                    # Checkpoint only what was actually persisted. A None
                    # result (question discarded at stop time) must stay
                    # retryable on resume.
                    save_progress(progress_file, realidx)
                    consecutive_errors = 0

                processed_count = len(results)
                acc = (
                    round(correct_count / processed_count * 100, 1)
                    if processed_count > 0 else None
                )

                elapsed = time.time() - start_time
                with self._progress_lock:
                    self.progress[pk].update({
                        "done": completed, "total": total,
                        "correct": correct_count,
                        "accuracy": acc,
                        "status": "running",
                    })
                self.queue.put({
                    "type": "progress",
                    "model_key": model_key, "bench_id": bench_id,
                    "done": completed, "total": total,
                    "correct": correct_count,
                    "accuracy": acc,
                    "elapsed": round(elapsed, 0),
                })

            if aborted:
                break

        # Shutdown: if stopped or this bench circuit-broke, don't wait for
        # in-flight API calls. request_abort() has already closed their
        # streams (so the server stops generating), which lets the threads
        # unwind immediately instead of holding the connection until the
        # read timeout.
        dropping = self.stopped or unreachable
        if dropping:
            request_abort(scope)
        executor.shutdown(wait=not dropping, cancel_futures=dropping)

        # Persist any buffered results so the on-disk file is complete.
        # Nothing is fabricated here: questions this bench never answered
        # (stop, or circuit-break) are simply absent, so a resume retries
        # exactly those.
        flush_results()

        elapsed = time.time() - start_time
        processed_count = len(results)
        unattempted = max(0, total - processed_count)
        accuracy = (
            round(correct_count / processed_count * 100, 2)
            if processed_count > 0 else None
        )
        correct_loose_count = sum(
            1 for r in results if r.get("correct_loose") is True
        )
        accuracy_loose = (
            round(correct_loose_count / processed_count * 100, 2)
            if processed_count > 0 else None
        )
        total_prompt = sum(
            r.get("token_usage", {}).get("prompt_tokens", 0) for r in results
        )
        total_completion = sum(
            r.get("token_usage", {}).get("completion_tokens", 0) for r in results
        )

        summary = {
            "model_key": model_key,
            "bench_id": bench_id,
            "bench_name": bench.name,
            "mode": self.mode,
            "max_model_len": self.model_max_len(model_key),
            "processed": processed_count,
            "correct": correct_count,
            "accuracy": accuracy,
            "correct_loose": correct_loose_count,
            "accuracy_loose": accuracy_loose,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "time_elapsed": round(elapsed, 0),
            "results_file": results_file,
            "aborted": aborted,
            "unreachable": unreachable,
            # Questions the bench never scored (stop / circuit-break /
            # per-question infra failures). A resume picks up exactly these.
            "unattempted": unattempted,
            "failed_pending": failed_pending,
        }
        if unreachable_error:
            summary["error"] = unreachable_error
        self.results[(model_key, bench_id)] = summary
        incomplete = processed_count < total
        with self._progress_lock:
            self.progress[pk].update({
                "done": processed_count, "total": total,
                "correct": correct_count, "accuracy": accuracy,
                "status": "interrupted" if incomplete else "done",
            })

        self.queue.put({
            "type": "pair_done",
            "model_key": model_key, "bench_id": bench_id,
            "bench_name": bench.name,
            "processed": processed_count, "correct": correct_count,
            "total": total, "unattempted": unattempted,
            "accuracy": accuracy, "time": round(elapsed, 0),
            "aborted": aborted, "stopped": self.stopped,
            "unreachable": unreachable, "error": unreachable_error or None,
        })

        # This bench is done; hand its worker budget back so a still-running
        # sibling bench can grow its concurrency (dynamic tail reallocation).
        reallocate(bench_id)

    def get_progress(self):
        # Return current progress snapshot for all (model, bench) pairs.
        with self._progress_lock:
            return {k: dict(v) for k, v in self.progress.items()}

    # -- persistence ------------------------------------------------------

    def _save_config(self) -> None:
        config = {
            "run_id": self.run_id,
            "model_keys": self.model_keys,
            "bench_ids": self.bench_ids,
            "mode": self.mode,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
            "max_model_len": self.max_model_len,
            "workers": self.workers,
            "limit": self.limit,
            "no_resume": self.no_resume,
            "started_at": datetime.now().isoformat(),
        }
        with open(self.output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def _save_summary(self) -> None:
        started_at = ""
        config_file = self.output_dir / "config.json"
        if config_file.exists():
            try:
                started_at = json.loads(config_file.read_text()).get("started_at", "")
            except Exception:
                pass
        summary = {
            "run_id": self.run_id,
            "mode": self.mode,
            "max_model_len": self.max_model_len,
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(),
            "status": "aborted" if self.stopped else ("error" if self.error else "completed"),
            "results": [
                {**v, "model_key": k[0], "bench_id": k[1]}
                for k, v in self.results.items()
            ],
        }
        if self.error:
            summary["error"] = self.error
        # Atomic write: never leave a 0-byte summary behind if the disk is
        # full or the process dies mid-write. A failed summary write is
        # reported on stderr but must not turn a completed run into an error.
        tmp_file = self.output_dir / "summary.json.tmp"
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_file, self.output_dir / "summary.json")
        except OSError as e:
            try:
                tmp_file.unlink(missing_ok=True)
            except OSError:
                pass
            print(f"WARNING: failed to write summary.json: {e}", file=sys.stderr)


# -- helpers for the API layer --------------------------------------------

def create_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def list_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not EVAL_OUTPUT_DIR.exists():
        return runs
    for entry in sorted(EVAL_OUTPUT_DIR.iterdir(), reverse=True):
        if not entry.is_dir():
            continue
        meta: dict[str, Any] = {"run_id": entry.name}
        config_file = entry / "config.json"
        if config_file.exists():
            try:
                meta["config"] = json.loads(config_file.read_text())
            except Exception:
                pass
        summary_file = entry / "summary.json"
        if summary_file.exists():
            try:
                summary = json.loads(summary_file.read_text())
                meta["status"] = summary.get("status", "completed")
                meta["summary"] = summary
            except Exception:
                pass
        else:
            has_progress = any(
                f.name.startswith("progress_") for f in entry.iterdir()
            )
            # "running" means there are progress files but no summary --
            # since the server may have restarted, this is really
            # "interrupted" unless an in-memory entry says it's active.
            meta["status"] = "interrupted" if has_progress else "completed"
        runs.append(meta)
    return runs
