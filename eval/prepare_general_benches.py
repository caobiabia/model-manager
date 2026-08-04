"""Download & convert the standard general benchmarks (MMLU / GSM8K / C-Eval)
into the JSONL schema used by the eval runner.

Outputs land in ``eval/data_general/<Dataset>/<split>.jsonl`` with the same
record shape as the other benches:

    {"realidx": 0, "question": "...", "subject": "...",
     "options": {"A": "...", ...}, "answer_idx": "C", "answer": "..."}

Run from the repo root:

    /mntnlp/csp/workspace/.venv/bin/python eval/prepare_general_benches.py

Sources (HuggingFace, default endpoint is the China mirror hf-mirror.com;
override with the ``HF_ENDPOINT`` env var):

* MMLU    -> cais/mmlu          (all/test, 57 subjects, 14,042 questions)
* GSM8K   -> openai/gsm8k       (main/test, 1,319 questions, free-form math)
* C-Eval  -> ceval/ceval-exam   (test, 52 subjects, 4-option Chinese MCQ)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# Use the China mirror by default; respect an explicit HF_ENDPOINT override.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

OUT_DIR = Path(__file__).resolve().parent / "data_general"
CACHE_DIR = Path.home() / ".cache" / "csp_eval_general"


def download(repo_id: str, filename: str, retries: int = 6) -> str:
    """Download a dataset file via curl (more reliable against hf-mirror
    than the huggingface_hub client on this network) and cache it under
    ``~/.cache/csp_eval_general``."""
    target = CACHE_DIR / repo_id.replace("/", "--") / filename
    if target.exists() and target.stat().st_size > 0:
        return str(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    url = (
        f"{os.environ['HF_ENDPOINT']}/datasets/{repo_id}"
        f"/resolve/main/{filename}"
    )
    tmp = target.with_suffix(".part")
    for attempt in range(retries):
        try:
            subprocess.run(
                [
                    "curl", "-sL", "--connect-timeout", "15",
                    "--max-time", "120", "--retry", "2", "--retry-all-errors",
                    "-o", str(tmp), url,
                ],
                check=True,
            )
            if tmp.exists() and tmp.stat().st_size > 0:
                tmp.rename(target)
                return str(target)
            raise RuntimeError(f"empty download from {url}")
        except Exception as e:  # noqa: BLE001
            if attempt < retries - 1:
                print(f"    retry {attempt + 1}/{retries} after: {e}")
                time.sleep(2 * (attempt + 1))
            else:
                raise
    raise RuntimeError("unreachable")


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(records):>6} questions -> {path}")


def prepare_mmlu() -> None:
    print("[1/3] MMLU (cais/mmlu, all/test)")
    target = OUT_DIR / "MMLU" / "test.jsonl"
    if target.exists():
        print(f"  already prepared ({target.stat().st_size / 1024:.0f} KB) -- skipping")
        return
    parquet = download("cais/mmlu", "all/test-00000-of-00001.parquet")
    df = pd.read_parquet(parquet)
    records: list[dict] = []
    letters = "ABCDEFGH"
    for i, row in enumerate(df.itertuples()):
        choices = list(row.choices)
        ans = int(row.answer)
        options = {
            letters[j]: str(c) for j, c in enumerate(choices[: len(letters)])
        }
        records.append({
            "realidx": i,
            "question": row.question,
            "subject": row.subject,
            "options": options,
            "answer_idx": letters[ans],
            "answer": str(choices[ans]),
        })
    write_jsonl(records, target)


def prepare_gsm8k() -> None:
    print("[2/3] GSM8K (openai/gsm8k, main/test)")
    target = OUT_DIR / "GSM8K" / "test.jsonl"
    if target.exists():
        print(f"  already prepared ({target.stat().st_size / 1024:.0f} KB) -- skipping")
        return
    parquet = download("openai/gsm8k", "main/test-00000-of-00001.parquet")
    df = pd.read_parquet(parquet)
    records: list[dict] = []
    for i, row in enumerate(df.itertuples()):
        m = re.search(r"####\s*(.+)", row.answer)
        gold = m.group(1).strip() if m else ""
        records.append({
            "realidx": i,
            "question": row.question,
            "subject": "gsm8k",
            "options": {},          # free-form: no MCQ options
            "answer_idx": gold,     # numeric gold answer (string)
            "answer": gold,
            "reference_answer": row.answer,  # full GSM8K solution text
        })
    write_jsonl(records, target)


def prepare_ceval() -> bool:
    """Download & convert C-Eval; return True when all subjects succeeded."""
    subjects = [
        "accountant", "advanced_mathematics", "art_studies", "basic_medicine",
        "business_administration", "chinese_language_and_literature",
        "civil_servant", "clinical_medicine", "college_chemistry",
        "college_economics", "college_physics", "college_programming",
        "computer_architecture", "computer_network", "discrete_mathematics",
        "education_science", "electrical_engineer",
        "environmental_impact_assessment_engineer", "fire_engineer",
        "high_school_biology", "high_school_chemistry", "high_school_chinese",
        "high_school_geography", "high_school_history",
        "high_school_mathematics", "high_school_physics",
        "high_school_politics", "ideological_and_moral_cultivation", "law",
        "legal_professional", "logic", "mao_zedong_thought", "marxism",
        "metrology_engineer", "middle_school_biology",
        "middle_school_chemistry", "middle_school_geography",
        "middle_school_history", "middle_school_mathematics",
        "middle_school_physics", "middle_school_politics",
        "modern_chinese_history", "operating_system", "physician",
        "plant_protection", "probability_and_statistics",
        "professional_tour_guide", "sports_science", "tax_accountant",
        "teacher_qualification", "urban_and_rural_planner",
        "veterinary_medicine",
    ]
    all_ok = True
    for split, split_name in [("test", "test")]:
        records: list[dict] = []
        realidx = 0
        failed: list[str] = []
        for subject in subjects:
            try:
                parquet = download(
                    "ceval/ceval-exam",
                    f"{subject}/{split}-00000-of-00001.parquet",
                )
                df = pd.read_parquet(parquet)
            except Exception as e:  # noqa: BLE001
                # The AWS CDN behind hf-mirror is intermittently unreachable;
                # skip this file now and report it so a re-run can finish it.
                failed.append(f"{subject}/{split}: {e}")
                continue
            for row in df.itertuples():
                options = {"A": row.A, "B": row.B, "C": row.C, "D": row.D}
                letter = str(row.answer).strip().upper()
                records.append({
                    "realidx": realidx,
                    "question": row.question,
                    "subject": subject,
                    "options": options,
                    "answer_idx": letter,
                    "answer": options.get(letter, ""),
                    "explanation": getattr(row, "explanation", "") or "",
                })
                realidx += 1
        write_jsonl(records, OUT_DIR / "C-Eval" / f"{split_name}.jsonl")
        if failed:
            all_ok = False
            print(f"  WARNING: {len(failed)}/{len(subjects)} subjects failed "
                  f"for {split_name} split (re-run to retry):")
            for f in failed[:10]:
                print(f"    - {f}")
    return all_ok


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prepare_mmlu()
    prepare_gsm8k()
    print("[3/3] C-Eval (ceval/ceval-exam, test + val)")
    for pass_no in range(1, 6):
        if pass_no > 1:
            print(f"  --- retry pass {pass_no} ---")
        if prepare_ceval():
            print("Done. Benches are registered in eval/benches.py.")
            return
    print("C-Eval still incomplete after retries -- re-run this script later "
          "to finish the remaining subjects.")
    sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
