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
MEDBENCH_FILE = _EVAL_DIR / "medbench.json"


def get_medbench() -> dict[str, Any]:
    """MedBench 官方榜单快照（API 榜单 + 自测榜单）。

    与内部 eval 条目无关: 这份数据来自 MedBench 官方榜单, 不随评测导入变化,
    直接编辑 eval/medbench.json 即可更新。
    """
    empty: dict[str, Any] = {"api": [], "self_test": [], "sub_benches": []}
    if not MEDBENCH_FILE.exists():
        return empty
    try:
        data = json.loads(MEDBENCH_FILE.read_text(encoding="utf-8"))
    except Exception:
        return empty
    if not isinstance(data, dict):
        return empty
    for key, default in empty.items():
        data.setdefault(key, default)
    return data


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
        if split == "general_follow":
            return "general_follow"
        if split == "general_knowledge":
            return "general_knowledge"
        if split == "general_reasoning":
            return "general_reasoning"
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
    # 思考强度 (档位 label 或数值); None = 未指定/模型不支持, 与旧条目等价
    effort = config.get("reasoning_effort", summary.get("reasoning_effort"))
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
            "reasoning_effort": effort,
            "max_model_len": r.get("max_model_len"),
            "accuracy": r.get("accuracy", 0),
            "correct": r.get("correct", 0),
            "processed": r.get("processed", 0),
            "accuracy_loose": r.get("accuracy_loose"),
            "correct_loose": r.get("correct_loose", 0),
            "run_id": run_id,
            "imported_at": datetime.now().isoformat(),
        }

        # upsert: replace existing entry with same
        # (model_key, bench_id, mode, reasoning_effort, max_model_len) — runs at
        # different thinking efforts stay separate entries
        found = False
        for i, e in enumerate(entries):
            if (
                e.get("model_key") == model_key
                and e.get("bench_id") == bench_id
                and e.get("mode") == mode
                and e.get("reasoning_effort") == entry.get("reasoning_effort")
                and e.get("max_model_len") == entry.get("max_model_len")
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
    """Return leaderboard entries grouped by category.

    Categories shown to users are: hard, full, general_follow,
    general_knowledge, general_reasoning.
    The removed categories (s10 "Sampled 10%", medqa_cn "subset" and
    general_subset "通用子集") are deliberately excluded from the response so
    the UI no longer renders them; their stored entries remain in
    leaderboard.json and newly imported runs keep getting categorised the same
    way, so dropping a category from the UI is reversible. `total` still
    reflects the full stored count.
    """
    entries = _load()
    SHOWN = (
        "hard", "full",
        "general_follow", "general_knowledge", "general_reasoning",
    )
    categories: dict[str, list[dict]] = {c: [] for c in SHOWN}
    for e in entries:
        cat = e.get("category", "hard")
        if cat in categories:
            categories[cat].append(e)
    return {
        "categories": {
            "hard": "MedicalAgentsBench 高难子集 (test_hard)",
            "full": "MedicalAgentsBench 全量测试集 (test)",
            "general_follow": "指令遵循 (IFEval/IFBench/Inverse IFEval)",
            "general_knowledge": "世界知识 (MMLU/GSM8K/C-Eval)",
            "general_reasoning": "推理能力 (GPQA Diamond/AIME 2026)",
        },
        "entries": categories,
        "total": len(entries),
        "medbench": get_medbench(),
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


def _entry_len(e: dict) -> int | None:
    v = e.get("max_model_len")
    return v if isinstance(v, int) and v > 0 else 16384


def export_to_xlsx(
        groups: list[dict[str, Any]],
        avg_mode: str = "simple",
        max_model_len: int | None = None,
) -> bytes:
    """Build an xlsx workbook for selected leaderboard groups.

    Each group dict supports:
      category: leaderboard category key (e.g. "hard", "general")
      benches:  optional list of bench_ids; empty means all in the category
      models:   optional list of model_keys; empty means all in the category

    avg_mode is "simple" (arithmetic mean) or "weighted" (weighted by the
    processed question count of each bench; entries without a valid processed
    count fall back to a weight of 1).

    Returns the workbook as bytes, or b"" when nothing matches.
    """
    from io import BytesIO

    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    labels = get_leaderboard()["categories"]
    entries = _load()
    wb = Workbook()
    wb.remove(wb.active)

    thin = Side(style="thin", color="D9D9D9")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="2F5B9E")
    best_font = Font(bold=True, color="1E7B34")
    center = Alignment(horizontal="center", vertical="center")

    def sanitize_sheet(title: str) -> str:
        for ch in '[]:*?/\\':
            title = title.replace(ch, "-")
        title = title.strip() or "leaderboard"
        return title[:31]

    used_titles: set[str] = set()

    for group in groups:
        cat = group.get("category", "")
        wanted_benches = group.get("benches") or None
        wanted_models = group.get("models") or None
        items = [e for e in entries if e.get("category") == cat]
        if max_model_len is not None:
            items = [e for e in items if _entry_len(e) == max_model_len]
        if wanted_benches:
            wanted = set(wanted_benches)
            items = [e for e in items if e.get("bench_id") in wanted]
        if wanted_models:
            wanted = set(wanted_models)
            items = [e for e in items if e.get("model_key") in wanted]
        if not items:
            continue

        bench_order: list[str] = []
        for e in items:
            bid = e.get("bench_id", "")
            if bid not in bench_order:
                bench_order.append(bid)
        # 行身份 = (model_key, 思考强度): 同一模型不同强度的成绩分行展示
        model_order: list[tuple[str, Any]] = []
        for e in items:
            mk = (e.get("model_key", ""), e.get("reasoning_effort"))
            if mk not in model_order:
                model_order.append(mk)

        avg_cache: dict[tuple[str, Any], float | None] = {}

        def avg_acc(mk: tuple[str, Any]) -> float | None:
            if mk not in avg_cache:
                es = [
                    e for e in items
                    if (e.get("model_key"), e.get("reasoning_effort")) == mk
                    and isinstance(e.get("accuracy"), (int, float))
                ]
                if not es:
                    avg_cache[mk] = None
                elif avg_mode == "weighted":
                    wsum = 0.0
                    wcnt = 0.0
                    for e in es:
                        w = e.get("processed")
                        if not isinstance(w, (int, float)) or w <= 0:
                            w = 1
                        wsum += e["accuracy"] * w
                        wcnt += w
                    avg_cache[mk] = wsum / wcnt
                else:
                    avg_cache[mk] = sum(e["accuracy"] for e in es) / len(es)
            return avg_cache[mk]

        model_order.sort(
            key=lambda mk: avg_acc(mk) if avg_acc(mk) is not None else -1.0,
            reverse=True,
        )

        bench_best: dict[str, float | None] = {}
        for bid in bench_order:
            vals = [
                e["accuracy"] for e in items
                if e.get("bench_id") == bid and isinstance(e.get("accuracy"), (int, float))
            ]
            bench_best[bid] = max(vals) if vals else None

        # 表名去掉 "(test_hard)" 这类 split 后缀, 免得撞上 Excel 的 31 字符上限被截断
        short_label = labels.get(cat, cat).split(" (")[0]
        base_title = sanitize_sheet(short_label)
        if max_model_len is not None:
            base_title = sanitize_sheet(
                f"{short_label}-{(max_model_len / 1024):.0f}k"
            )
        title = base_title
        n = 2
        while title in used_titles:
            suffix = f"-{n}"
            title = base_title[: 31 - len(suffix)] + suffix
            n += 1
        used_titles.add(title)

        ws = wb.create_sheet(title=title)
        headers = ["模型", "Avg"]
        for bid in bench_order:
            e = next((e for e in items if e.get("bench_id") == bid), None)
            headers.append(e.get("bench_name", bid) if e else bid)
        ws.append(headers)
        for ci in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=ci)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = center
            cell.border = border

        def first_entry(mk: tuple[str, Any], bid: str):
            return next(
                (e for e in items
                 if (e.get("model_key"), e.get("reasoning_effort")) == mk
                 and e.get("bench_id") == bid),
                None,
            )

        for mk in model_order:
            e0 = first_entry(mk, bench_order[0]) if bench_order else None
            name = e0.get("model_name", mk[0]) if e0 else mk[0]
            if mk[1] is not None:
                name = f"{name} (强度 {mk[1]})"
            row = [name, avg_acc(mk)]
            for bid in bench_order:
                e = first_entry(mk, bid)
                acc = e.get("accuracy") if e and isinstance(e.get("accuracy"), (int, float)) else None
                row.append(acc)
            ws.append(row)
            r = ws.max_row
            ws.cell(row=r, column=1).font = Font(bold=True)
            avg_cell = ws.cell(row=r, column=2)
            if isinstance(avg_cell.value, (int, float)):
                avg_cell.number_format = '0.0"%"'
            for ci, bid in enumerate(bench_order, start=3):
                cell = ws.cell(row=r, column=ci)
                if isinstance(cell.value, (int, float)):
                    cell.number_format = '0.0"%"'
                    if (
                        bench_best[bid] is not None
                        and cell.value == bench_best[bid]
                        and cell.value > 0
                    ):
                        cell.font = best_font
            for ci in range(1, len(headers) + 1):
                ws.cell(row=r, column=ci).border = border

        ws.column_dimensions["A"].width = 34
        ws.column_dimensions["B"].width = 10
        for ci in range(3, len(headers) + 1):
            ws.column_dimensions[get_column_letter(ci)].width = 16
        ws.freeze_panes = "C2"

    if not wb.sheetnames:
        return b""
    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()
