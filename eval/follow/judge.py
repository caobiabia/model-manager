"""LLM-as-a-Judge scoring for the Inverse IFEval benchmark.

Inverse IFEval (https://huggingface.co/datasets/m-a-p/Inverse_IFEval) is
scored with the paper's "LLM-as-a-Judge" protocol: each question ships a
judge **system prompt** and a **prompt template** with ``{prompt}``,
``{response_reference}`` and ``{response}`` slots. The judge returns a JSON
block ``{"answer_score": 0|1}``; a score of 1 is treated as correct.

The judge model reuses the same local DeepSeek instance used for MCQ answer
extraction (``eval.common.get_extract_model``), so no extra deployment is
needed.
"""

from __future__ import annotations

import logging
import re

from eval.common import call_model, get_extract_model

log = logging.getLogger(__name__)

_ANSWER_SCORE_RE = re.compile(r'\{\s*"answer_score"\s*:\s*(0|1)\s*\}')
_SCORE_LINE_RE = re.compile(r"【评分】\s*[:：]?\s*(\d+)\s*分")


def _parse_score(text: str) -> int | None:
    """Extract the 0/1 ``answer_score`` from a judge response."""
    if not text:
        return None
    m = _ANSWER_SCORE_RE.search(text)
    if m:
        return int(m.group(1))
    # fallback: the Chinese "【评分】：X分" marker line
    m = _SCORE_LINE_RE.search(text)
    if m:
        return int(m.group(1))
    return None


def judge_inverse_ifeval(raw_response: str, problem: dict) -> bool:
    """Judge the model's raw response against the Inverse IFEval reference.

    Returns ``True`` only when the judge awards a perfect score (1).
    Unparseable judge outputs are treated as not following (False).
    """
    if not raw_response or not raw_response.strip():
        return False

    template = problem.get("judge_prompt_template") or ""
    system = problem.get("judge_system_prompt") or ""
    ref = problem.get("response_reference") or ""
    prompt = problem.get("question") or ""
    if not template or not system:
        log.warning("Inverse IFEval record missing judge fields: %s", problem.get("realidx"))
        return False

    user_content = template.format(
        prompt=prompt,
        response_reference=ref,
        response=raw_response,
    )

    client, model, is_deepseek = get_extract_model()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    try:
        # zero_shot mode disables thinking for the judge (fast + deterministic).
        raw, _reasoning, _usage = call_model(
            client, model, messages, mode="zero_shot",
            is_deepseek=is_deepseek, max_tokens=4000,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("Inverse IFEval judge call failed: %s", e)
        return False

    score = _parse_score(raw)
    if score is None:
        log.warning("Could not parse judge score from: %.200s", raw)
        return False
    return score >= 1


# Alias used by the bench registry.
scorer_inverse_ifeval = judge_inverse_ifeval
