"""Leaderboard data management -- persist and query eval results.

Stores entries in a single JSON file (eval/leaderboard.json). Each entry
is one (model, bench, mode) score. Importing a run reads its summary.json
and upserts entries, replacing old scores for the same triple.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from eval.common import _EVAL_DIR
from eval.benches import get_bench

LB_FILE = _EVAL_DIR / "leaderboard.json"


def _load() -> list[dict[str, Any]]:
    if not LB_FILE.exists():
        return []
    try:
        return json.loads(LB_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save(entries: list[dict[str, Any]]) -> None:
    LB_FILE.parent.mkdir(parents=True, exist_ok=True)
    LB_FILE.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


def _bench_category(bench_id: str) -> str:
    """Determine the leaderboard category from a bench id."""
    bench = get_bench(bench_id)
    if bench:
        split = bench.split or ""
        if split == "test_hard":
            return "hard"
        if split == "test":
            return "full"
        if split.startswith("sampled10"):
            return "s10"
        if split == "general":
            return "general"
        if split == "general_subset":
            return "general_subset"
        return "subset"
    # fallback: infer from id
    if bench_id.endswith("_full"):
        return "full"
    if any(s in bench_id for s in ("_en_hard", "_zh_hard", "_en_easy", "_zh_easy")):
        return "s10"
    if bench_id == "medqa_cn":
        return "subset"
    return "hard"


def import_run(
    run_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Import all results from a run into the leaderboard.

    Replaces existing entries with the same (model_key, bench_id, mode).
    Returns a summary of what was imported.
    """
    summary_file = output_dir / "summary.json"
    config_file = output_dir / "config.json"
    if not summary_file.exists():
        return {"ok": False, "error": "Run has no summary (not completed?)"}

    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    config = {}
    if config_file.exists():
        config = json.loads(config_file.read_text(encoding="utf-8"))

    mode = summary.get("mode", config.get("mode", "?"))
    results = summary.get("results", [])
    if not results:
        return {"ok": False, "error": "Run has no results"}

    entries = _load()
    imported = 0
    replaced = 0

    for r in results:
        model_key = r.get("model_key", "")
        bench_id = r.get("bench_id", "")
        bench_name = r.get("bench_name", bench_id)
        category = _bench_category(bench_id)

        # look up model display name
        from model_config import get_model_by_name
        mc = get_model_by_name(model_key)
        model_name = mc.get("display_name", model_key) if mc else model_key

        entry = {
            "id": str(uuid.uuid4())[:8],
            "model_key": model_key,
            "model_name": model_name,
            "bench_id": bench_id,
            "bench_name": bench_name,
            "category": category,
            "mode": mode,
            "accuracy": r.get("accuracy", 0),
            "correct": r.get("correct", 0),
            "processed": r.get("processed", 0),
            "run_id": run_id,
            "imported_at": datetime.now().isoformat(),
        }

        # upsert: replace existing entry with same (model_key, bench_id, mode)
        found = False
        for i, e in enumerate(entries):
            if (
                e.get("model_key") == model_key
                and e.get("bench_id") == bench_id
                and e.get("mode") == mode
            ):
                entries[i] = entry
                found = True
                replaced += 1
                break
        if not found:
            entries.append(entry)
            imported += 1

    _save(entries)
    return {
        "ok": True,
        "imported": imported,
        "replaced": replaced,
        "total_entries": len(entries),
    }


def get_leaderboard() -> dict[str, Any]:
    """Return leaderboard entries grouped by category."""
    entries = _load()
    categories: dict[str, list[dict]] = {
        "hard": [],
        "full": [],
        "general": [],
        "general_subset": [],
        "s10": [],
        "subset": [],
    }
    for e in entries:
        cat = e.get("category", "hard")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(e)
    return {
        "categories": {
            "hard": "Hard (test_hard)",
            "full": "Test (test)",
            "general": "\u901a\u7528 Bench (MMLU/GSM8K/C-Eval/GPQA/AIME)",
            "general_subset": "\u901a\u7528\u5b50\u96c6 (MMLU/GSM8K/C-Eval)",
            "s10": "Sampled 10%",
            "subset": "MedQA \u4e2d\u6587\u5b50\u96c6",
        },
        "entries": categories,
        "total": len(entries),
    }


def delete_entry(entry_id: str) -> bool:
    entries = _load()
    new = [e for e in entries if e.get("id") != entry_id]
    if len(new) == len(entries):
        return False
    _save(new)
    return True


def clear_all() -> int:
    count = len(_load())
    _save([])
    return count
