"""Export SWE-bench run results into official-harness predictions JSONL.

The inline eval runner only *generates* patches for SWE-bench; correctness
requires running the repo test suites with the official SWE-bench harness
(needs Docker). This script converts one eval run into the predictions file
format accepted by that harness:

    {"instance_id": "...", "model_name_or_path": "...", "model_patch": "..."}

Usage:

    /mntnlp/csp/workspace/.venv/bin/python eval/export_swebench_predictions.py \
        --run 20260804_120000 [--bench swebench_verified] [--mode cot]

Output: eval/output/<run_id>/swebench_predictions_<model>.jsonl

On a Docker-enabled machine, evaluate with e.g.:

    python -m swebench.harness.run_evaluation \
        --dataset_name princeton-nlp/SWE-bench_Verified \
        --predictions_path eval/output/<run_id>/swebench_predictions_<model>.jsonl \
        --max_workers 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from eval.common import EVAL_OUTPUT_DIR
except ImportError:  # running as a plain script: eval/ is on sys.path
    from common import EVAL_OUTPUT_DIR


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="eval run id, e.g. 20260804_120000")
    ap.add_argument("--bench", default="swebench_verified")
    ap.add_argument("--mode", default="cot", choices=["cot", "zero_shot"])
    ap.add_argument("--model", action="append", help="filter by model key (repeatable)")
    args = ap.parse_args()

    run_dir = EVAL_OUTPUT_DIR / args.run
    if not run_dir.is_dir():
        raise SystemExit(f"run dir not found: {run_dir}")

    exported = 0
    for rf in sorted(run_dir.glob(f"*__{args.bench}__{args.mode}.json")):
        model_short = rf.name.split(f"__{args.bench}__")[0]
        if args.model and model_short not in args.model:
            continue
        results = json.loads(rf.read_text(encoding="utf-8"))
        out = run_dir / f"swebench_predictions_{model_short}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for r in results:
                if not r.get("predicted_patch"):
                    continue
                f.write(json.dumps({
                    "instance_id": r.get("instance_id", r.get("realidx")),
                    "model_name_or_path": model_short,
                    "model_patch": r["predicted_patch"],
                }, ensure_ascii=False) + "\n")
        n = sum(1 for r in results if r.get("predicted_patch"))
        exported += 1
        print(f"wrote {n}/{len(results)} patches -> {out}")

    if not exported:
        raise SystemExit(
            f"no results found for bench={args.bench} mode={args.mode} in {run_dir}"
        )


if __name__ == "__main__":
    main()
