#!/usr/bin/env python3
"""Extract wrong answers from eval runs and produce statistics.

A *wrong answer* is a question whose recorded ``correct`` flag is not True.
Two kinds are tracked separately:

* ``no_answer``  -- the answer extractor produced nothing usable (empty,
  ``null`` or a letter outside the option set): refusal / format drift
* ``wrong_choice`` -- a valid option was chosen, but not the gold one

Usage
-----
    python eval/wrong_answers.py extract \\
        --run-id 20260910_003815 \\
        --model-key qwen36_35b_sft_med_v72_ep3 \\
        --out eval/analysis/v72ep3

    python eval/wrong_answers.py compare \\
        --a eval/analysis/v72ep3 --b eval/analysis/v4ep3 \\
        --out eval/analysis/compare

``extract`` writes into ``--out``:

    wrongs_<bench>.jsonl   per-bench wrong items (full raw_response included)
    wrongs_all.jsonl       all wrong items, sorted by bench + realidx
    unjudged_all.jsonl     items with correct == null (extraction never scored)
    index_all.jsonl        compact per-item index (correctness/prediction)
    stats.json             machine-readable statistics
    REPORT.md              human-readable report (Chinese)

``compare`` writes into ``--out``:

    COMPARE.md             contingency table (both wrong / only A / only B / both right)
    wrongs_both.jsonl      compact lists of questions by outcome
    wrongs_only_a.jsonl
    wrongs_only_b.jsonl
    right_b_wrong_a.jsonl  A wrong & B right, enriched: question + options + gold
                           + both models' answer letter and full response/reasoning
    right_a_wrong_b.jsonl  the mirror case
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from eval.common import BENCHMARK_DATA_DIR, EVAL_OUTPUT_DIR  # noqa: E402
from model_config import MODELS  # noqa: E402

# MedicalAgentsBench datasets (bench id == dataset dir name)
MED_DATASETS = [
    "medqa", "pubmedqa", "medmcqa", "mmlu", "mmlu-pro",
    "medbullets", "afrimedqa", "medexqa", "medxpertqa-r", "medxpertqa-u",
]

_LETTER_RE = re.compile(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])")


# -- loading helpers -------------------------------------------------------

def served_prefix(model_key: str) -> str:
    """Result files are named after served_model_name; fall back to the key."""
    cfg = MODELS.get(model_key)
    if cfg and cfg.get("served_model_name"):
        return cfg["served_model_name"]
    return model_key


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_items(path: Path) -> list[dict]:
    data = load_json(path)
    return data if isinstance(data, list) else data.get("results", data.get("items", []))


def hard_realidx(dataset: str) -> set[str]:
    """realidx values of a dataset's test_hard split (for hard/full linkage)."""
    path = BENCHMARK_DATA_DIR / dataset / "test_hard.jsonl"
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.add(str(json.loads(line).get("realidx")))
    return out


def find_bench_files(run_ids: str | list[str], model_key: str,
                     bench_ids: list[str]) -> dict[str, Path]:
    """Locate each bench's result file, searching the runs in order.

    Accepts several run ids so a model whose hard split and full split live in
    different runs (e.g. separate max-effort runs) can be extracted at once.
    """
    if isinstance(run_ids, str):
        run_ids = [r.strip() for r in run_ids.split(",") if r.strip()]
    prefix = served_prefix(model_key)
    found: dict[str, Path] = {}
    for run_id in run_ids:
        run_dir = EVAL_OUTPUT_DIR / run_id
        if not run_dir.exists():
            raise SystemExit(f"run dir not found: {run_dir}")
        for bid in bench_ids:
            if bid in found:
                continue
            hits = sorted(run_dir.glob(f"{prefix}__{bid}__*.json"))
            if hits:
                found[bid] = hits[0]
    return found


# -- statistics ------------------------------------------------------------

def _is_no_answer(item: dict) -> bool:
    pred = item.get("predicted_answer")
    if pred is None:
        return True
    pred = str(pred).strip()
    if not pred:
        return True
    return pred not in (item.get("options") or {})


def _mentions(text: str, letter: str) -> bool:
    return bool(text) and letter in set(_LETTER_RE.findall(text))


def analyse_bench(bench_id: str, items: list[dict]) -> dict:
    """Per-bench statistics over one result file."""
    total = len(items)
    correct = sum(1 for i in items if i.get("correct") is True)
    unjudged = sum(1 for i in items if i.get("correct") is None)
    wrong_items = [i for i in items if i.get("correct") is not True]
    no_answer = [i for i in wrong_items if _is_no_answer(i)]
    wrong_choice = [i for i in wrong_items if not _is_no_answer(i)]

    def _mean_tokens(group: list[dict]) -> float:
        vals = [
            (i.get("token_usage") or {}).get("completion_tokens") or 0
            for i in group
        ]
        return round(sum(vals) / len(vals), 1) if vals else 0.0

    gold_letters = Counter(str(i.get("answer_idx")) for i in wrong_choice)
    pred_letters = Counter(str(i.get("predicted_answer")) for i in wrong_choice)
    confusion = Counter(
        f"{i.get('answer_idx')}->{i.get('predicted_answer')}" for i in wrong_choice
    )
    gold_mentioned = sum(
        1 for i in wrong_choice
        if _mentions(str(i.get("raw_response") or ""), str(i.get("answer_idx")))
    )

    return {
        "bench_id": bench_id,
        "total": total,
        "correct": correct,
        "wrong": len(wrong_items),
        "unjudged": unjudged,
        "scored": total - unjudged,
        "accuracy": round(100.0 * correct / total, 2) if total else None,
        "accuracy_scored": (
            round(100.0 * correct / (total - unjudged), 2)
            if total - unjudged else None
        ),
        "no_answer": len(no_answer),
        "wrong_choice": len(wrong_choice),
        "gold_letters_at_miss": dict(gold_letters.most_common()),
        "pred_letters_at_miss": dict(pred_letters.most_common()),
        "confusion_top": dict(confusion.most_common(8)),
        "gold_letter_mentioned_in_response": gold_mentioned,
        "mean_completion_tokens_correct": _mean_tokens(
            [i for i in items if i.get("correct") is True]
        ),
        "mean_completion_tokens_wrong": _mean_tokens(wrong_items),
        "wrong_items": wrong_items,
    }


def extract(run_id: str, model_key: str, bench_ids: list[str], out_dir: Path) -> dict:
    run_ids = [r.strip() for r in run_id.split(",") if r.strip()]
    files = find_bench_files(run_ids, model_key, bench_ids)
    missing = [b for b in bench_ids if b not in files]
    if not files:
        raise SystemExit(
            f"no result files for model '{model_key}' in run {run_id} "
            f"(looked for prefix '{served_prefix(model_key)}')"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = (MODELS.get(model_key) or {}).get("display_name", model_key)

    stats: dict = {
        "run_id": run_id,
        "run_ids": run_ids,
        "model_key": model_key,
        "model_name": model_name,
        "benches": {},
        "missing_benches": missing,
    }
    all_wrongs: list[dict] = []
    all_unjudged: list[dict] = []
    index_rows: list[dict] = []

    hards = {ds: hard_realidx(ds) for ds in MED_DATASETS}

    for bid, path in sorted(files.items()):
        items = load_items(path)
        base_ds = bid[:-5] if bid.endswith("_full") else bid
        hard_set = hards.get(base_ds, set())
        is_full_bench = bid.endswith("_full")

        for i in items:
            ridx = str(i.get("realidx"))
            row = {
                "bench_id": bid,
                "is_full": is_full_bench,
                "in_hard_subset": bool(ridx in hard_set),
                "realidx": ridx,
                "question": i.get("question"),
                "options": i.get("options"),
                "answer_idx": i.get("answer_idx"),
                "predicted_answer": i.get("predicted_answer"),
                "correct": i.get("correct"),
                "correct_loose": i.get("correct_loose"),
                "no_answer": _is_no_answer(i),
                "completion_tokens": (i.get("token_usage") or {}).get("completion_tokens"),
                "time_elapsed": i.get("time_elapsed"),
                "reasoning": i.get("reasoning"),
                "raw_response": i.get("raw_response"),
                "model_name": model_name,
                "model_key": model_key,
                "run_id": run_id,
            }
            index_rows.append({k: row[k] for k in (
                "bench_id", "is_full", "in_hard_subset", "realidx", "answer_idx",
                "predicted_answer", "correct", "no_answer", "completion_tokens",
            )})
            if i.get("correct") is not True:
                all_wrongs.append(row)
                if i.get("correct") is None:
                    all_unjudged.append(row)

        st = analyse_bench(bid, items)
        st.pop("wrong_items")
        stats["benches"][bid] = st

        bench_wrongs = [r for r in all_wrongs if r["bench_id"] == bid]
        with open(out_dir / f"wrongs_{bid}.jsonl", "w", encoding="utf-8") as f:
            for r in bench_wrongs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # aggregate over all extracted benches
    grand = {
        "total": sum(b["total"] for b in stats["benches"].values()),
        "correct": sum(b["correct"] for b in stats["benches"].values()),
        "wrong": sum(b["wrong"] for b in stats["benches"].values()),
        "unjudged": sum(b["unjudged"] for b in stats["benches"].values()),
        "no_answer": sum(b["no_answer"] for b in stats["benches"].values()),
    }
    grand["accuracy"] = (
        round(100.0 * grand["correct"] / grand["total"], 2) if grand["total"] else None
    )
    stats["grand_total"] = grand

    with open(out_dir / "wrongs_all.jsonl", "w", encoding="utf-8") as f:
        for r in all_wrongs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "unjudged_all.jsonl", "w", encoding="utf-8") as f:
        for r in all_unjudged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "index_all.jsonl", "w", encoding="utf-8") as f:
        for r in index_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats["hard_vs_full"] = _hard_vs_full(index_rows)
    (out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "REPORT.md").write_text(render_report(stats), encoding="utf-8")
    return stats


def _hard_vs_full(index_rows: list[dict]) -> dict:
    """For questions in both the hard file and the full file, compare the two
    recorded answers (same model, same prompt -> determinism check)."""
    hard_pred: dict[tuple[str, str], dict] = {}
    full_pred: dict[tuple[str, str], dict] = {}
    for r in index_rows:
        key = (r["bench_id"].replace("_full", ""), r["realidx"])
        (full_pred if r["is_full"] else hard_pred)[key] = r
    shared = sorted(set(hard_pred) & set(full_pred))
    same_pred = sum(
        1 for k in shared
        if hard_pred[k]["predicted_answer"] == full_pred[k]["predicted_answer"]
    )
    same_correct = sum(
        1 for k in shared if hard_pred[k]["correct"] == full_pred[k]["correct"]
    )
    out = {
        "shared_questions": len(shared),
        "same_prediction": same_pred,
        "same_correctness": same_correct,
        "per_dataset": {},
    }
    per = defaultdict(lambda: {"n": 0, "same_pred": 0, "same_correct": 0})
    for k in shared:
        e = per[k[0]]
        e["n"] += 1
        e["same_pred"] += hard_pred[k]["predicted_answer"] == full_pred[k]["predicted_answer"]
        e["same_correct"] += hard_pred[k]["correct"] == full_pred[k]["correct"]
    out["per_dataset"] = {k: dict(v) for k, v in sorted(per.items())}
    return out


# -- reporting -------------------------------------------------------------

_MED_HARD_LABEL = "高难子集 (test_hard)"
_MED_FULL_LABEL = "全量测试集 (test)"


def _pct(a, b) -> str:
    return f"{100.0 * a / b:.1f}%" if b else "-"


def render_report(stats: dict) -> str:
    lines: list[str] = []
    add = lines.append
    name = stats["model_name"]
    add(f"# 错题分析 — {name}")
    add("")
    add(f"- 模型 key: `{stats['model_key']}`")
    add(f"- run: `{stats['run_id']}`")
    if stats.get("missing_benches"):
        add(f"- ⚠️ run 中缺少这些 bench 的结果文件: {', '.join(stats['missing_benches'])}")
    add("")

    for suffix, label in (("", _MED_HARD_LABEL), ("_full", _MED_FULL_LABEL)):
        rows = {b: s for b, s in stats["benches"].items()
                if (b.endswith("_full")) == bool(suffix)}
        if not rows:
            continue
        add(f"## MedicalAgentsBench {label}")
        add("")
        add("| 数据集 | 题数 | 正确 | 错误 | 未判分 | 准确率 | 未作答/解析失败 | 误选 | 错题中最常丢的答案 | 错题中最常选的答案 |")
        add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
        for bid, s in sorted(rows.items()):
            gold = ", ".join(f"{k}({v})" for k, v in list(s["gold_letters_at_miss"].items())[:3]) or "-"
            pred = ", ".join(f"{k}({v})" for k, v in list(s["pred_letters_at_miss"].items())[:3]) or "-"
            acc = f"{s['accuracy']:.1f}%" if s["accuracy"] is not None else "-"
            add(f"| {bid} | {s['total']} | {s['correct']} | {s['wrong']} | {s['unjudged']} "
                f"| {acc} | {s['no_answer']} | {s['wrong_choice']} | {gold} | {pred} |")
        tot = {k: sum(s[k] for s in rows.values()) for k in
               ("total", "correct", "wrong", "unjudged", "no_answer", "wrong_choice")}
        gold = Counter()
        pred = Counter()
        for s in rows.values():
            gold.update(s["gold_letters_at_miss"])
            pred.update(s["pred_letters_at_miss"])
        add(f"| **合计** | {tot['total']} | {tot['correct']} | {tot['wrong']} | "
            f"{tot['unjudged']} | {_pct(tot['correct'], tot['total'])} | "
            f"{tot['no_answer']} | {tot['wrong_choice']} | "
            f"{', '.join(f'{k}({v})' for k, v in gold.most_common(3))} | "
            f"{', '.join(f'{k}({v})' for k, v in pred.most_common(3))} |")
        add("")

        mt_c = [s["mean_completion_tokens_correct"] for s in rows.values() if s["total"]]
        mt_w = [s["mean_completion_tokens_wrong"] for s in rows.values() if s["total"]]
        if mt_c and mt_w:
            add(f"平均生成长度(completion tokens): 正确题 {sum(mt_c)/len(mt_c):.0f}, "
                f"错题 {sum(mt_w)/len(mt_w):.0f}")
            add("")

    hv = stats.get("hard_vs_full") or {}
    if hv.get("shared_questions"):
        add("## 高难子集 vs 全量测试集 (同一批题)")
        add("")
        add(f"高难子集中的 {hv['shared_questions']} 道题全部也出现在全量测试集里。"
            f"两次回答一致 {hv['same_prediction']} 道 "
            f"({_pct(hv['same_prediction'], hv['shared_questions'])}), "
            f"对错判定一致 {hv['same_correctness']} 道 "
            f"({_pct(hv['same_correctness'], hv['shared_questions'])}).")
        add("")
        add("| 数据集 | 共同题数 | 答案一致 | 对错一致 |")
        add("| --- | ---: | ---: | ---: |")
        for ds, e in hv["per_dataset"].items():
            add(f"| {ds} | {e['n']} | {e['same_pred']} | {e['same_correct']} |")
        add("")

    g = stats["grand_total"]
    add("## 总览")
    add("")
    add(f"- 题数 {g['total']}, 正确 {g['correct']}, 错误 {g['wrong']} "
        f"(其中未判分 {g['unjudged']}), 总体准确率 {g['accuracy']}%")
    add(f"- 错题构成: 未作答/解析失败 {g['no_answer']} 道, 误选 {g['wrong'] - g['no_answer'] - g['unjudged']} 道")
    add("")
    return "\n".join(lines)


# -- compare ---------------------------------------------------------------

def _load_index(d: Path) -> dict[tuple[str, str], dict]:
    out = {}
    for line in (d / "index_all.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[(r["bench_id"], r["realidx"])] = r
    return out


def _load_source_items(d: Path) -> dict[tuple[str, str], dict]:
    """Reload the raw per-question records of the extraction's source run.

    ``index_all.jsonl`` only keeps compact fields, so responses of *correct*
    answers are not in the extraction dir; the run's own result files are.
    """
    s = load_json(d / "stats.json")
    files = find_bench_files(s.get("run_ids") or s["run_id"], s["model_key"],
                             list(s["benches"]))
    out: dict[tuple[str, str], dict] = {}
    for bid, path in files.items():
        for item in load_items(path):
            out[(bid, str(item.get("realidx")))] = item
    return out


def _sided_record(key: tuple[str, str], a_item: dict, b_item: dict,
                  a_name: str, b_name: str) -> dict:
    """One question with both models' answers + full responses."""
    def side(item: dict, name: str, prefix: str) -> dict:
        item = item or {}
        return {
            f"{prefix}_model": name,
            f"{prefix}_answer": item.get("predicted_answer"),
            f"{prefix}_correct": item.get("correct"),
            f"{prefix}_no_answer": _is_no_answer(item),
            f"{prefix}_completion_tokens": (item.get("token_usage") or {}).get("completion_tokens"),
            f"{prefix}_response": item.get("raw_response"),
            f"{prefix}_reasoning": item.get("reasoning"),
        }
    src = b_item or a_item or {}
    return {
        "bench_id": key[0],
        "realidx": key[1],
        "question": src.get("question"),
        "options": src.get("options"),
        "answer_idx": src.get("answer_idx"),
        **side(a_item, a_name, "a"),
        **side(b_item, b_name, "b"),
    }


def compare(a_dir: Path, b_dir: Path, out_dir: Path,
            a_label: str | None = None, b_label: str | None = None) -> dict:
    a_idx, b_idx = _load_index(a_dir), _load_index(b_dir)
    a_stats = load_json(a_dir / "stats.json")
    b_stats = load_json(b_dir / "stats.json")
    a_label = a_label or a_stats["model_name"]
    b_label = b_label or b_stats["model_name"]
    shared = sorted(set(a_idx) & set(b_idx))
    out_dir.mkdir(parents=True, exist_ok=True)

    both_wrong, only_a, only_b, both_right = [], [], [], []
    for k in shared:
        wa = a_idx[k]["correct"] is not True
        wb = b_idx[k]["correct"] is not True
        rec = {
            "bench_id": k[0], "realidx": k[1],
            "answer_idx": a_idx[k]["answer_idx"],
            "pred_a": a_idx[k]["predicted_answer"],
            "pred_b": b_idx[k]["predicted_answer"],
            "no_answer_a": a_idx[k]["no_answer"],
            "no_answer_b": b_idx[k]["no_answer"],
        }
        (both_wrong if wa and wb else only_a if wa else only_b if wb else both_right).append(rec)

    for fname, rows in (("wrongs_both.jsonl", both_wrong), ("wrongs_only_a.jsonl", only_a),
                        ("wrongs_only_b.jsonl", only_b)):
        with open(out_dir / fname, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Enriched one-sided files: B right where A is wrong (and the mirror).
    # Needs the source runs, because a correct answer's response is not part
    # of the extraction dir (which only dumps wrong ones).
    a_full, b_full = _load_source_items(a_dir), _load_source_items(b_dir)
    sides = {
        "right_b_wrong_a.jsonl": (only_a, a_full, b_full),   # A wrong, B right
        "right_a_wrong_b.jsonl": (only_b, a_full, b_full),   # B wrong, A right
    }
    for fname, (rows, af, bf) in sides.items():
        with open(out_dir / fname, "w", encoding="utf-8") as f:
            for r in rows:
                k = (r["bench_id"], r["realidx"])
                f.write(json.dumps(
                    _sided_record(k, af.get(k), bf.get(k), a_label, b_label),
                    ensure_ascii=False) + "\n")

    n = len(shared)
    stats = {
        "a": {"model_name": a_label, "stats": a_stats},
        "b": {"model_name": b_label, "stats": b_stats},
        "shared_questions": n,
        "both_wrong": len(both_wrong),
        "only_a_wrong": len(only_a),
        "only_b_wrong": len(only_b),
        "both_right": len(both_right),
        "jaccard_wrong": round(len(both_wrong) / max(1, len(both_wrong) + len(only_a) + len(only_b)), 4),
    }

    lines = [f"# 错题对比 — A: {a_label} vs B: {b_label}", "",
             f"共同出现(两个模型的 run 都覆盖)的题: {n} 道", "",
             "| 结果 | 题数 | 占比 |", "| --- | ---: | ---: |",
             f"| 两个模型都错 | {len(both_wrong)} | {_pct(len(both_wrong), n)} |",
             f"| 只有 A 错 | {len(only_a)} | {_pct(len(only_a), n)} |",
             f"| 只有 B 错 | {len(only_b)} | {_pct(len(only_b), n)} |",
             f"| 都答对 | {len(both_right)} | {_pct(len(both_right), n)} |", "",
             f"共同错误(Jaccard): {stats['jaccard_wrong']}", "",
             "明细文件：`wrongs_both.jsonl` / `wrongs_only_a.jsonl` / `wrongs_only_b.jsonl`（精简索引）；"
             "`right_b_wrong_a.jsonl`（A 错 B 对，含题干、选项、gold、双方答案字母与完整回答）与镜像 "
             "`right_a_wrong_b.jsonl`（含 B 的答案与完整回答，可直接用于分析 A 的差距）。", ""]

    per = defaultdict(lambda: Counter())
    for r in both_wrong:
        per[r["bench_id"]]["both"] += 1
    for r in only_a:
        per[r["bench_id"]]["only_a"] += 1
    for r in only_b:
        per[r["bench_id"]]["only_b"] += 1
    for r in both_right:
        per[r["bench_id"]]["right"] += 1
    lines += ["| bench | 都错 | 仅A错 | 仅B错 | 都对 |", "| --- | ---: | ---: | ---: | ---: |"]
    for bid in sorted(per):
        c = per[bid]
        lines.append(f"| {bid} | {c['both']} | {c['only_a']} | {c['only_b']} | {c['right']} |")
    lines.append("")

    (out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2)[:500000], encoding="utf-8")
    (out_dir / "COMPARE.md").write_text("\n".join(lines), encoding="utf-8")
    return stats


# -- CLI -------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ext = sub.add_parser("extract", help="dump wrong answers + stats for one run")
    p_ext.add_argument("--run-id", required=True)
    p_ext.add_argument("--model-key", required=True,
                       help="model config key (served_model_name gives the file prefix)")
    p_ext.add_argument("--benches", default="",
                       help="comma-separated bench ids; default = 10 hard + 10 full medical benches")
    p_ext.add_argument("--out", required=True, type=Path)

    p_cmp = sub.add_parser("compare", help="compare two extraction dirs")
    p_cmp.add_argument("--a", required=True, type=Path)
    p_cmp.add_argument("--b", required=True, type=Path)
    p_cmp.add_argument("--a-label", default=None)
    p_cmp.add_argument("--b-label", default=None)
    p_cmp.add_argument("--out", required=True, type=Path)

    args = ap.parse_args()
    if args.cmd == "extract":
        if args.benches:
            bench_ids = [b.strip() for b in args.benches.split(",") if b.strip()]
        else:
            bench_ids = MED_DATASETS + [f"{d}_full" for d in MED_DATASETS]
        stats = extract(args.run_id, args.model_key, bench_ids, args.out)
        g = stats["grand_total"]
        print(f"[extract] {stats['model_name']} run={stats['run_id']}")
        print(f"          题数 {g['total']} 正确 {g['correct']} 错误 {g['wrong']} "
              f"准确率 {g['accuracy']}%")
        print(f"          输出 -> {args.out}")
        if stats.get("missing_benches"):
            print(f"          ⚠️ 缺少: {', '.join(stats['missing_benches'])}")
    else:
        stats = compare(args.a, args.b, args.out, args.a_label, args.b_label)
        print(f"[compare] 共同题 {stats['shared_questions']} 都错 {stats['both_wrong']} "
              f"仅A错 {stats['only_a_wrong']} 仅B错 {stats['only_b_wrong']} "
              f"都对 {stats['both_right']}")
        print(f"          输出 -> {args.out}")


if __name__ == "__main__":
    main()
