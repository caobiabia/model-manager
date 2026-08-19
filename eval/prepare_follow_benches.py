"""Download & convert the instruction-following benchmarks into the eval
JSONL schema.

Outputs land in ``eval/data_general/<Dataset>/test.jsonl`` with the same
record shape used by the runner:

    {"realidx": 0, "question": "...", "subject": "...", "options": {},
     "answer_idx": "", "answer": "", ...bench-specific fields...}

* IFEval       -> google/IFEval                  (~500 prompts, 25 instruction types)
* IFBench      -> allenai/IFBench_test           (300 samples, 58 constraint types)
* Inverse IFEval -> m-a-p/Inverse_IFEval         (1,012 questions, 8 reverse types)

The IFEval / IFBench records keep the verifier fields ``instruction_id_list``
and ``kwargs``; the Inverse IFEval records keep the ``judge_*`` fields used by
the LLM-as-a-Judge scorer.

Run from the repo root:

    /mntnlp/csp/workspace/.venv/bin/python eval/prepare_follow_benches.py

Notes:
* IFBench's ``kwargs`` store integral values as floats (e.g. ``n_start: 3.0``);
  these are coerced to ``int`` so the vendored ``repeat:repeat_span`` checker
  (which slices the response) does not fail.
* This also ensures the NLTK corpora used by the verifiers are available.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

try:
    from eval.prepare_general_benches import download, write_jsonl
except ImportError:  # running as a plain script: eval/ is on sys.path
    from prepare_general_benches import download, write_jsonl

try:
    from eval.follow import ensure_nltk_data
except ImportError:
    def ensure_nltk_data():  # pragma: no cover
        pass

OUT_DIR = Path(__file__).resolve().parent / "data_general"


def _coerce_kwargs(kwargs: list) -> list:
    """Deep-copy kwargs, turning integral floats into ints (IFBench stores
    values like ``n_start: 3.0`` which the verifier feeds into slicing)."""
    out = []
    for d in kwargs:
        if not isinstance(d, dict):
            out.append(d)
            continue
        cleaned = {}
        for k, v in d.items():
            if isinstance(v, float) and v.is_integer():
                cleaned[k] = int(v)
            elif isinstance(v, list) and all(
                isinstance(x, float) and x.is_integer() for x in v
            ):
                cleaned[k] = [int(x) for x in v]
            else:
                cleaned[k] = v
        out.append(cleaned)
    return out


def _if_verifiable_record(realidx, key, prompt, instruction_id_list, kwargs, subject):
    return {
        "realidx": realidx,
        "question": prompt,
        "subject": subject,
        "options": {},
        "answer_idx": "",
        "answer": "",
        "key": key,
        "instruction_id_list": list(instruction_id_list),
        "kwargs": _coerce_kwargs(kwargs),
    }


def prepare_ifeval() -> None:
    """google/IFEval -- ~500 prompts with 25 verifiable instruction types."""
    print("[1/3] IFEval (google/IFEval)")
    target = OUT_DIR / "IFEval" / "test.jsonl"
    if target.exists():
        print(f"  already prepared ({target.stat().st_size / 1024:.0f} KB) -- skipping")
        return
    src = download("google/IFEval", "ifeval_input_data.jsonl")
    records = []
    with open(src, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            records.append(_if_verifiable_record(
                i, r.get("key"), r.get("prompt", ""),
                r.get("instruction_id_list", []), r.get("kwargs", []),
                "ifeval",
            ))
    if not records:
        raise RuntimeError("IFEval produced no records")
    write_jsonl(records, target)


def prepare_ifbench() -> None:
    """allenai/IFBench_test -- 300 WildChat-derived samples, 58 constraints."""
    print("[2/3] IFBench (allenai/IFBench_test)")
    target = OUT_DIR / "IFBench" / "test.jsonl"
    if target.exists():
        print(f"  already prepared ({target.stat().st_size / 1024:.0f} KB) -- skipping")
        return
    parquet = download(
        "allenai/IFBench_test", "data/train-00000-of-00001.parquet"
    )
    df = pd.read_parquet(parquet)
    if len(df) != 300:
        raise RuntimeError(f"IFBench should have 300 samples, got {len(df)}")
    records = []
    for i, row in enumerate(df.itertuples()):
        kws = row.kwargs
        if isinstance(kws, dict):
            kws = [kws]
        records.append(_if_verifiable_record(
            i, str(row.key), row.prompt,
            list(row.instruction_id_list), [dict(x) for x in kws],
            "ifbench",
        ))
    write_jsonl(records, target)


def prepare_inverse_ifeval() -> None:
    """m-a-p/Inverse_IFEval -- 1,012 questions (zh/en), 8 reverse types."""
    print("[3/3] Inverse IFEval (m-a-p/Inverse_IFEval)")
    target = OUT_DIR / "InverseIFEval" / "test.jsonl"
    if target.exists():
        print(f"  already prepared ({target.stat().st_size / 1024:.0f} KB) -- skipping")
        return
    src = download("m-a-p/Inverse_IFEval", "Inverse_IFEval_Dataset.json")
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    if len(data) != 1012:
        raise RuntimeError(f"Inverse IFEval should have 1012 questions, got {len(data)}")
    records = []
    for i, r in enumerate(data):
        records.append({
            "realidx": i,
            "question": r.get("prompt", ""),
            "subject": r.get("instruction_types", ""),
            "options": {},
            "answer_idx": "",
            "answer": "",
            "language": r.get("language", ""),
            "instruction_types": r.get("instruction_types", ""),
            "response_reference": r.get("response_reference", ""),
            "judge_prompt_template": r.get("judge_prompt_template", ""),
            "judge_system_prompt": r.get("judge_system_prompt", ""),
        })
    write_jsonl(records, target)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Ensure the verifier corpora are available before registering data.
    try:
        ensure_nltk_data()
    except Exception as e:  # noqa: BLE001
        print(f"  warning: could not ensure NLTK data ({e}); verifiers may degrade")
    prepare_ifeval()
    prepare_ifbench()
    prepare_inverse_ifeval()
    print("Done. Benches are registered in eval/benches.py.")


if __name__ == "__main__":
    main()
