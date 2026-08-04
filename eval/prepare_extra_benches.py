"""Download & convert extra general benchmarks into the eval JSONL schema.

Outputs land in ``eval/data_general/<Dataset>/<split>.jsonl`` with the same
record shape used by the runner:

    {"realidx": 0, "question": "...", "subject": "...",
     "options": {"A": "...", ...}, "answer_idx": "C", "answer": "..."}

* GPQA Diamond       -> hendrydong/gpqa_diamond_mc  (198 questions, 4-choice MCQ)
* AIME 2026          -> MathArena/aime_2026         (30 problems, integer answers)
* SWE-bench Verified -> princeton-nlp/SWE-bench_Verified (500 instances, patch)

Run from the repo root:

    /mntnlp/csp/workspace/.venv/bin/python eval/prepare_extra_benches.py

Notes:
* The original GPQA dataset (Idavidrein/gpqa) is gated on HuggingFace; this
  script uses a public mirror with the same Diamond split.
* SWE-bench is a "patch generation" bench: the runner stores the generated
  patch for every instance. Final PASS_TO_PASS/FAIL_TO_PASS scoring must be
  done with the official SWE-bench harness (needs Docker) -- see
  ``eval/export_swebench_predictions.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

try:
    from eval.prepare_general_benches import download, write_jsonl
except ImportError:  # running as a plain script: eval/ is on sys.path
    from prepare_general_benches import download, write_jsonl

OUT_DIR = Path(__file__).resolve().parent / "data_general"


def prepare_gpqa() -> None:
    """GPQA Diamond: 198 expert-written PhD-level science MCQs."""
    print("[1/3] GPQA Diamond (hendrydong/gpqa_diamond_mc, test)")
    target = OUT_DIR / "GPQA-Diamond" / "test.jsonl"
    if target.exists():
        print(f"  already prepared ({target.stat().st_size / 1024:.0f} KB) -- skipping")
        return
    parquet = download(
        "hendrydong/gpqa_diamond_mc",
        "data/test-00000-of-00001.parquet",
    )
    df = pd.read_parquet(parquet)

    marker_re = re.compile(r"\n\(([A-D])\)\s*")
    trailing_re = re.compile(
        r"\s*Please write your final answer.*$", flags=re.IGNORECASE | re.S
    )
    records: list[dict] = []
    failed: list[tuple[int, str]] = []

    for i, row in enumerate(df.itertuples()):
        text = row.problem
        markers = list(marker_re.finditer(text))
        # Options are the last consecutive A/B/C/D block (question text can
        # contain stray "(X)" markers).
        block_start = None
        for st in range(len(markers) - 4, -1, -1):
            seq = [m.group(1) for m in markers[st : st + 4]]
            if seq == ["A", "B", "C", "D"]:
                block_start = st
                break
        if block_start is None:
            failed.append((i, f"cannot locate 4-option block ({len(markers)} markers)"))
            continue
        block = markers[block_start : block_start + 4]
        question = text[: block[0].start()].strip()
        options: dict[str, str] = {}
        for j, m in enumerate(block):
            end = block[j + 1].start() if j + 1 < len(block) else len(text)
            value = text[m.end() : end].strip()
            if j == len(block) - 1:
                value = trailing_re.sub("", value).strip()
            options[m.group(1)] = value
        m = re.search(r"\\boxed\{([A-D])\}", row.solution)
        letter = m.group(1) if m else str(row.solution).strip().upper()
        if letter not in options:
            failed.append((i, f"answer letter {letter!r} not in options"))
            continue
        records.append({
            "realidx": i,
            "question": question,
            "subject": str(row.domain),
            "options": options,
            "answer_idx": letter,
            "answer": options[letter],
        })

    if failed:
        raise RuntimeError(f"GPQA parse failed for {len(failed)} rows: {failed[:5]}")
    if len(records) != 198:
        raise RuntimeError(f"GPQA Diamond should have 198 rows, got {len(records)}")
    write_jsonl(records, target)


def prepare_aime2026() -> None:
    """AIME 2026: 30 competition problems with integer answers (0-999)."""
    print("[2/3] AIME 2026 (MathArena/aime_2026, train split)")
    target = OUT_DIR / "AIME2026" / "test.jsonl"
    if target.exists():
        print(f"  already prepared ({target.stat().st_size / 1024:.0f} KB) -- skipping")
        return
    parquet = download("MathArena/aime_2026", "data/train-00000-of-00001.parquet")
    df = pd.read_parquet(parquet)
    if len(df) != 30:
        raise RuntimeError(f"AIME 2026 should have 30 problems, got {len(df)}")
    records: list[dict] = []
    for i, row in enumerate(df.itertuples()):
        answer = str(row.answer)
        records.append({
            "realidx": i,
            "question": str(row.problem),
            "subject": "aime2026",
            "options": {},
            "answer_idx": answer,
            "answer": answer,
            "problem_idx": int(row.problem_idx),
        })
    write_jsonl(records, target)


def prepare_swebench_verified() -> None:
    """SWE-bench Verified: 500 human-validated GitHub issue instances."""
    print("[3/3] SWE-bench Verified (princeton-nlp/SWE-bench_Verified, test)")
    target = OUT_DIR / "SWE-bench-Verified" / "test.jsonl"
    if target.exists():
        print(f"  already prepared ({target.stat().st_size / 1024:.0f} KB) -- skipping")
        return
    parquet = download(
        "princeton-nlp/SWE-bench_Verified",
        "data/test-00000-of-00001.parquet",
    )
    df = pd.read_parquet(parquet)
    if len(df) != 500:
        raise RuntimeError(f"SWE-bench Verified should have 500 instances, got {len(df)}")

    def _listish(v):
        if v is None:
            return []
        return list(v) if not isinstance(v, str) else [v]

    records: list[dict] = []
    for i, row in enumerate(df.itertuples()):
        records.append({
            "realidx": i,
            "question": str(row.problem_statement),
            "subject": str(row.repo),
            "options": {},
            "answer_idx": "",
            "answer": "",
            "instance_id": str(row.instance_id),
            "repo": str(row.repo),
            "base_commit": str(row.base_commit),
            "patch": str(row.patch),
            "test_patch": str(row.test_patch),
            "FAIL_TO_PASS": _listish(row.FAIL_TO_PASS),
            "PASS_TO_PASS": _listish(row.PASS_TO_PASS),
            "hints_text": str(row.hints_text or ""),
            "version": str(row.version or ""),
            "environment_setup_commit": str(row.environment_setup_commit or ""),
            "difficulty": str(row.difficulty or ""),
        })
    write_jsonl(records, target)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prepare_gpqa()
    prepare_aime2026()
    prepare_swebench_verified()
    print("Done. Benches are registered in eval/benches.py.")


if __name__ == "__main__":
    main()
