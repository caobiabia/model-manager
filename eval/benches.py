"""Bench registry -- flat list of all evaluation benches.

Every bench is treated identically by the runner. The only things that
differ between benches are:

* **data_file** -- where the questions live
* **language** -- ``"en"`` or ``"zh"`` (controls prompt template)
* **build_messages** -- how to format the system/user messages

To add a new bench, either append a ``Bench(...)`` call to the auto-registration
loop at the bottom, or call ``register(Bench(...))`` from your own module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from eval.common import BENCHMARK_DATA_DIR, load_jsonl


# -- prompt constants ------------------------------------------------------

_EN_ZS_SYSTEM = (
    "You are a knowledgeable medical assistant. Provide accurate answers "
    "to the medical question based on the given information."
)
_EN_ZS_USER = (
    "Given the following question and options, select the correct answer "
    "by returning only the answer index (e.g., 'A', 'B', 'C', or 'D').\n\n"
    "Question:\n{question}\n\n"
    "Options:\n{options}\n\n"
    "Reply with the answer index only."
)
_EN_COT_SYSTEM = (
    "You are a helpful medical expert. Your task is to answer multi-choice "
    "medical questions by thinking step-by-step and providing clear "
    "explanations."
)
_EN_COT_USER = (
    "Please answer this medical question by first thinking step-by-step and "
    "then choosing from the provided options. Your responses will be used for "
    "research purposes only, so please have a definite answer (e.g., 'A', "
    "'B', 'C', etc.).\n\n"
    "Question:\n{question}\n\n"
    "Options:\n{options}\n\n"
)
_ZH_SYSTEM = (
    "\u4f60\u662f\u4e00\u4e2a\u9009\u62e9\u9898\u7b54\u9898\u52a9\u624b\u3002"
    "\u7b54\u6848\u8bf7\u653e\u5230\u4e00\u4e2aJSON\u5bf9\u8c61\u4e2d\u8f93\u51fa\uff0c"
    "\u683c\u5f0f\u4e3a{\"answer\":\"\u9009\u9879\u5b57\u6bcd\"}\uff0c"
    "\u4e0d\u8981\u8f93\u51fa\u4efb\u4f55\u5176\u4ed6\u5185\u5bb9\u3002"
)
# "你是一个选择题答题助手。答案请放到一个JSON对象中输出，格式为{"answer":"选项字母"}，不要输出任何其他内容。"

# General English MCQ (standard MMLU)
_EN_GENERAL_ZS_SYSTEM = (
    "You are a helpful assistant. Answer the following multiple-choice "
    "question with the single correct option letter."
)
_EN_GENERAL_ZS_USER = (
    "Given the following question and options, select the correct answer "
    "by returning only the answer index (e.g., 'A', 'B', 'C', or 'D').\n\n"
    "Question:\n{question}\n\n"
    "Options:\n{options}\n\n"
    "Reply with the answer index only."
)
_EN_GENERAL_COT_SYSTEM = (
    "You are a helpful assistant. Answer the question by thinking "
    "step-by-step and then choosing from the provided options."
)
_EN_GENERAL_COT_USER = (
    "Please answer this question by first thinking step-by-step and then "
    "choosing from the provided options. Your responses will be used for "
    "research purposes only, so please have a definite answer (e.g., 'A', "
    "'B', 'C', etc.).\n\n"
    "Question:\n{question}\n\n"
    "Options:\n{options}\n\n"
)

# GSM8K -- free-form numeric answers (no MCQ options)
_GSM8K_ZS_SYSTEM = (
    "You are a math problem solver. Solve the word problem and return "
    "only the final numeric answer."
)
_GSM8K_ZS_USER = (
    "Solve the following math word problem. Return only the final numeric "
    "answer.\n\nProblem:\n{question}\n"
)
_GSM8K_COT_SYSTEM = (
    "You are a math problem solver. Solve the word problem step by step "
    "and end your response with the final numeric answer on its own line "
    "prefixed with '#### '."
)
_GSM8K_COT_USER = (
    "Solve the following math word problem step by step. End your response "
    "with a line like '#### 42' containing the final numeric answer.\n\n"
    "Problem:\n{question}\n"
)

# AIME -- free-form integer answers (0-999), e.g. "#### 042"
_AIME_ZS_SYSTEM = (
    "You are a math competition solver. Solve the problem and return "
    "only the final integer answer."
)
_AIME_ZS_USER = (
    "Solve the following math competition problem. Return only the final "
    "integer answer (an integer from 0 to 999).\n\n"
    "Problem:\n{question}\n"
)
_AIME_COT_SYSTEM = (
    "You are a math competition solver. Solve the problem step by step "
    "and end your response with the final integer answer."
)
_AIME_COT_USER = (
    "Solve the following math competition problem step by step. End your "
    "response with a line like '#### 042' containing the final integer "
    "answer.\n\n"
    "Problem:\n{question}\n"
)

# Instruction following (IFEval / IFBench / Inverse IFEval) -- just hand the
# raw instruction/prompt to the model; correctness is judged by a verifier.
_FOLLOW_ZS_SYSTEM = (
    "You are a helpful assistant. Follow the user's instructions carefully "
    "and respond exactly as asked."
)
_FOLLOW_ZS_USER = "{question}"
_FOLLOW_COT_SYSTEM = (
    "You are a helpful assistant. Follow the user's instructions carefully. "
    "Think step by step if useful, then produce your final response."
)
_FOLLOW_COT_USER = "{question}"

# SWE-bench -- free-form unified diff patch generation
_SWE_ZS_SYSTEM = (
    "You are an expert software engineer. Given a GitHub issue, produce "
    "a complete patch that resolves it. Output only the unified diff, "
    "with no extra explanation."
)
_SWE_ZS_USER = (
    "Repository: {repo}\nBase commit: {base_commit}\n\n"
    "Issue:\n{question}\n\n"
    "Produce a complete git diff patch that fixes this issue. Output only "
    "the patch."
)
_SWE_COT_SYSTEM = (
    "You are an expert software engineer. Think step-by-step about how to "
    "fix the GitHub issue, then output the final complete patch."
)
_SWE_COT_USER = (
    "Repository: {repo}\nBase commit: {base_commit}\n\n"
    "Issue:\n{question}\n\n"
    "Think step-by-step about the fix, then output the final patch as a "
    "unified diff inside a code block."
)


@dataclass
class Bench:
    """A single evaluation benchmark.

    Attributes:
        id:        stable identifier used in API/CLI (e.g. ``"medqa"``)
        name:      display name
        data_file: path to the JSONL questions file
        language:  ``"en"`` or ``"zh"`` -- picks the prompt family
        split:     data split label (informational, e.g. ``"test_hard"``)
        description: short human-readable description
        format:    ``"mcq"`` (letter choice, default), ``"free"``
                   (free-form numeric answer, e.g. GSM8K), or ``"follow"``
                   (instruction following -- scored by a verifier/judge).
        prompt_style: English prompt family: ``"med"`` (medical MCQ),
                      ``"general"`` (general MCQ), ``"gsm8k"`` / ``"aime"``
                      (free-form math), ``"swebench"`` (patch generation), or
                      ``"follow"`` (instruction-following benches).
        scorable: whether the runner can compute an inline accuracy.
                  False for benches whose correctness requires an external
                  harness (e.g. SWE-bench patch test execution).
        scorer:   optional callable ``scorer(raw_response, question) -> bool``
                  used to compute per-question correctness for
                  ``format=="follow"`` benches (built-in verifiers/judges).
        scorer_loose: optional second scorer (e.g. the official IFEval "loose"
                  upper-bound metric) recorded alongside ``scorer`` so both
                  official accuracies are surfaced.
    """

    id: str
    name: str
    data_file: Path
    language: str = "en"
    split: str = "test_hard"
    description: str = ""
    format: str = "mcq"
    prompt_style: str = "med"
    scorable: bool = True
    scorer: callable | None = None
    scorer_loose: callable | None = None

    def load_questions(self) -> list[dict]:
        """Load and return all questions from the data file."""
        return load_jsonl(self.data_file)

    def build_messages(self, question: dict, mode: str) -> list[dict]:
        """Build system + user messages for the model.

        For English benches, *mode* selects between the zero-shot and
        chain-of-thought prompt templates. For the Chinese bench the
        prompt is fixed (JSON-style) and *mode* only toggles thinking
        on/off (handled in :func:`eval.common.call_model`).
        """
        q_text = question.get("question", "")
        options = question.get("options", {})
        options_inline = " ".join(f"({k}) {v}" for k, v in options.items())
        options_listed = "\n".join(f"{k}. {v}" for k, v in options.items())

        if self.format == "free":
            if self.prompt_style == "aime":
                if mode == "zero_shot":
                    return [
                        {"role": "system", "content": _AIME_ZS_SYSTEM},
                        {"role": "user", "content": _AIME_ZS_USER.format(question=q_text)},
                    ]
                return [
                    {"role": "system", "content": _AIME_COT_SYSTEM},
                    {"role": "user", "content": _AIME_COT_USER.format(question=q_text)},
                ]
            if mode == "zero_shot":
                return [
                    {"role": "system", "content": _GSM8K_ZS_SYSTEM},
                    {"role": "user", "content": _GSM8K_ZS_USER.format(question=q_text)},
                ]
            return [
                {"role": "system", "content": _GSM8K_COT_SYSTEM},
                {"role": "user", "content": _GSM8K_COT_USER.format(question=q_text)},
            ]

        if self.format == "patch":
            repo = question.get("repo", "")
            base_commit = question.get("base_commit", "")
            if mode == "zero_shot":
                return [
                    {"role": "system", "content": _SWE_ZS_SYSTEM},
                    {
                        "role": "user",
                        "content": _SWE_ZS_USER.format(
                            repo=repo, base_commit=base_commit, question=q_text
                        ),
                    },
                ]
            return [
                {"role": "system", "content": _SWE_COT_SYSTEM},
                {
                    "role": "user",
                    "content": _SWE_COT_USER.format(
                        repo=repo, base_commit=base_commit, question=q_text
                    ),
                },
            ]

        if self.format == "follow":
            if mode == "zero_shot":
                return [
                    {"role": "system", "content": _FOLLOW_ZS_SYSTEM},
                    {"role": "user", "content": _FOLLOW_ZS_USER.format(question=q_text)},
                ]
            return [
                {"role": "system", "content": _FOLLOW_COT_SYSTEM},
                {"role": "user", "content": _FOLLOW_COT_USER.format(question=q_text)},
            ]

        if self.language == "zh":
            return [
                {"role": "system", "content": _ZH_SYSTEM},
                {"role": "user", "content": f"{q_text}\n{options_listed}"},
            ]

        if self.prompt_style == "general":
            if mode == "zero_shot":
                return [
                    {"role": "system", "content": _EN_GENERAL_ZS_SYSTEM},
                    {
                        "role": "user",
                        "content": _EN_GENERAL_ZS_USER.format(
                            question=q_text, options=options_inline
                        ),
                    },
                ]
            return [
                {"role": "system", "content": _EN_GENERAL_COT_SYSTEM},
                {
                    "role": "user",
                    "content": _EN_GENERAL_COT_USER.format(
                        question=q_text, options=options_inline
                    ),
                },
            ]

        # English
        if mode == "zero_shot":
            return [
                {"role": "system", "content": _EN_ZS_SYSTEM},
                {
                    "role": "user",
                    "content": _EN_ZS_USER.format(
                        question=q_text, options=options_inline
                    ),
                },
            ]
        # cot
        return [
            {"role": "system", "content": _EN_COT_SYSTEM},
            {
                "role": "user",
                "content": _EN_COT_USER.format(
                    question=q_text, options=options_inline
                ),
            },
        ]


# -- registry --------------------------------------------------------------

BENCHES: dict[str, Bench] = {}


def register(bench: Bench) -> Bench:
    """Add a bench to the global registry (id must be unique)."""
    if bench.id in BENCHES:
        raise ValueError(f"Duplicate bench id: {bench.id}")
    BENCHES[bench.id] = bench
    return bench


def get_bench(bench_id: str) -> Bench | None:
    return BENCHES.get(bench_id)


def list_benches() -> list[Bench]:
    return list(BENCHES.values())


# -- bench display names / descriptions ------------------------------------

_BENCH_NAMES = {
    "medqa": "MedQA",
    "pubmedqa": "PubMedQA",
    "medmcqa": "MedMCQA",
    "mmlu": "MMLU",
    "mmlu-pro": "MMLU-Pro",
    "medbullets": "MedBullets",
    "afrimedqa": "AfriMedQA",
    "medexqa": "MedExQA",
    "medxpertqa-r": "MedXpertQA-R",
    "medxpertqa-u": "MedXpertQA-U",
}

_BENCH_DESC = {
    "medqa": "USMLE-style medical Q&A",
    "pubmedqa": "Biomedical research yes/no reasoning",
    "medmcqa": "Indian medical entrance MCQ",
    "mmlu": "Massive Multitask Language Understanding",
    "mmlu-pro": "MMLU professional (harder, 10 options)",
    "medbullets": "Medical exam bullet reasoning",
    "afrimedqa": "African medical curriculum Q&A",
    "medexqa": "Medical exam-style extended Q&A",
    "medxpertqa-r": "MedXpertQA real-world cases",
    "medxpertqa-u": "MedXpertQA synthetic cases",
}

# -- auto-registration: 10 MedicalAgentsBench datasets ---------------------
# Full test_hard (1251 questions total) and full test (11636 questions total)

for _ds in [
    "medqa", "pubmedqa", "medmcqa", "mmlu", "mmlu-pro",
    "medbullets", "afrimedqa", "medexqa", "medxpertqa-r", "medxpertqa-u",
]:
    # test_hard split (hard subset, ~100 per dataset)
    register(Bench(
        id=_ds,
        name=_BENCH_NAMES[_ds],
        data_file=BENCHMARK_DATA_DIR / _ds / "test_hard.jsonl",
        language="en",
        split="test_hard",
        description=_BENCH_DESC[_ds],
    ))
    # full test split (all questions, ~1000 per dataset)
    _full_test = BENCHMARK_DATA_DIR / _ds / "test.jsonl"
    if _full_test.exists():
        register(Bench(
            id=f"{_ds}_full",
            name=_BENCH_NAMES[_ds] + " (\u5168\u91cf)",
            # "全量"
            data_file=_full_test,
            language="en",
            split="test",
            description=_BENCH_DESC[_ds] + " (full test split)",
        ))



# -- standard general benches: MMLU / GSM8K / C-Eval ----------------------
# Prepared by eval/prepare_general_benches.py into eval/data_general/.
# These sit in the "general" group (split == "general") in the UI, separate
# from the MedicalAgentsBench medical datasets.

_GENERAL_DATA_DIR = Path(__file__).resolve().parent / "data_general"

_mmlu_test = _GENERAL_DATA_DIR / "MMLU" / "test.jsonl"
if _mmlu_test.exists():
    register(Bench(
        id="mmlu_std",
        name="MMLU (\u6807\u51c6 57\u79d1)",  # MMLU (标准 57科)
        data_file=_mmlu_test,
        language="en",
        split="general",
        prompt_style="general",
        description="Standard MMLU test (57 subjects, 14,042 questions)",
    ))

_gsm8k_test = _GENERAL_DATA_DIR / "GSM8K" / "test.jsonl"
if _gsm8k_test.exists():
    register(Bench(
        id="gsm8k",
        name="GSM8K",
        data_file=_gsm8k_test,
        language="en",
        split="general",
        format="free",
        prompt_style="gsm8k",
        description="Grade-school math word problems (test, 1,319 questions)",
    ))

_ceval_test = _GENERAL_DATA_DIR / "C-Eval" / "test.jsonl"
if _ceval_test.exists():
    register(Bench(
        id="ceval",
        name="C-Eval",
        data_file=_ceval_test,
        language="zh",
        split="general",
        description="C-Eval test (52 subjects, 12,342 questions)",
    ))

# -- qwen35b error-focused subsets (run 20260803_175115) ------------------
# Composition (based on the fine-tuned model's current results):
#   * MMLU:  all tested-wrong + correct/untested fill -> 1000
#   * GSM8K: all wrong + random correct -> 1000
#   * C-Eval: 250 wrong + random correct -> 1000
# Each record keeps its original realidx plus a "subset_origin" tag.

_mmlu_sub = _GENERAL_DATA_DIR / "MMLU" / "subset.jsonl"
if _mmlu_sub.exists():
    register(Bench(
        id="mmlu_std_sub",
        name="MMLU (1000\u5b50\u96c6)",  # MMLU (1000子集)
        data_file=_mmlu_sub,
        language="en",
        split="general_subset",
        prompt_style="general",
        description="MMLU 1000-question subset: tested-wrong + correct/untested "
                    "fill (based on qwen35b run 20260803_175115)",
    ))

_gsm8k_sub = _GENERAL_DATA_DIR / "GSM8K" / "subset.jsonl"
if _gsm8k_sub.exists():
    register(Bench(
        id="gsm8k_sub",
        name="GSM8K (1000\u5b50\u96c6)",  # GSM8K (1000子集)
        data_file=_gsm8k_sub,
        language="en",
        split="general_subset",
        format="free",
        prompt_style="gsm8k",
        description="GSM8K 1000-question subset: all tested-wrong + random "
                    "correct fill (based on qwen35b run 20260803_175115)",
    ))

_ceval_sub = _GENERAL_DATA_DIR / "C-Eval" / "subset.jsonl"
if _ceval_sub.exists():
    register(Bench(
        id="ceval_sub",
        name="C-Eval (1000\u5b50\u96c6)",  # C-Eval (1000子集)
        data_file=_ceval_sub,
        language="zh",
        split="general_subset",
        description="C-Eval 1000-question subset: 250 tested-wrong + 750 "
                    "random correct (based on qwen35b run 20260803_175115)",
    ))


# -- extra general benches: GPQA Diamond / AIME 2026 / SWE-bench ---------
# Prepared by eval/prepare_extra_benches.py into eval/data_general/.

_gpqa_test = _GENERAL_DATA_DIR / "GPQA-Diamond" / "test.jsonl"
if _gpqa_test.exists():
    register(Bench(
        id="gpqa_diamond",
        name="GPQA Diamond",
        data_file=_gpqa_test,
        language="en",
        split="general",
        prompt_style="general",
        description="Graduate-Level Google-Proof Q&A Diamond (198 PhD-level "
                    "biology/physics/chemistry MCQs)",
    ))

_aime_test = _GENERAL_DATA_DIR / "AIME2026" / "test.jsonl"
if _aime_test.exists():
    register(Bench(
        id="aime2026",
        name="AIME 2026",
        data_file=_aime_test,
        language="en",
        split="general",
        format="free",
        prompt_style="aime",
        description="American Invitational Mathematics Examination 2026 "
                    "(30 problems, integer answers 0-999)",
    ))

# SWE-bench Verified 暂缓（下掉）：本环境没有 Docker，无法跑官方 harness
# 出分；生成 patch 不能当作正确率。等有 Docker 评测环境后，取消下面注释
# 并运行 eval/prepare_extra_benches.py 重新准备数据即可恢复。
# _swe_test = _GENERAL_DATA_DIR / "SWE-bench-Verified" / "test.jsonl"
# if _swe_test.exists():
#     register(Bench(
#         id="swebench_verified",
#         name="SWE-bench Verified",
#         data_file=_swe_test,
#         language="en",
#         split="general",
#         format="patch",
#         prompt_style="swebench",
#         scorable=False,
#         description="500 human-validated GitHub issues; runner saves "
#                     "generated patches, official harness scoring required",
#     ))

# -- instruction-following benches: IFEval / IFBench / Inverse IFEval -----
# Prepared by eval/prepare_follow_benches.py into eval/data_general/.
# These are format="follow" benches: there is no single extractable answer,
# so per-question correctness comes from a scorer callback --
#   * IFEval / IFBench: the (vendored) official heuristic verifiers, strict
#     prompt-level metric (every verifiable instruction must be satisfied)
#   * Inverse IFEval: the dataset's "LLM-as-a-Judge" protocol (0/1 score)

def _lazy_follow_scorer():
    """Strict IFEval-style verifier; converts (raw, problem) -> bool."""
    from eval.follow import verify_follow_strict
    return lambda raw, problem: verify_follow_strict(problem, raw)


def _lazy_follow_scorer_loose():
    """Official IFEval 'loose' (upper-bound) verifier; (raw, problem) -> bool."""
    from eval.follow import verify_follow_loose
    return lambda raw, problem: verify_follow_loose(problem, raw)


def _lazy_inverse_judge():
    """Inverse IFEval LLM-as-a-Judge scorer."""
    from eval.follow.judge import judge_inverse_ifeval
    return judge_inverse_ifeval


_ifeval_test = _GENERAL_DATA_DIR / "IFEval" / "test.jsonl"
if _ifeval_test.exists():
    register(Bench(
        id="ifeval",
        name="IFEval",
        data_file=_ifeval_test,
        language="en",
        split="general",
        format="follow",
        prompt_style="follow",
        scorer=_lazy_follow_scorer(),
        scorer_loose=_lazy_follow_scorer_loose(),
        description="Google IFEval (~500 prompts, 25 verifiable instruction "
                    "types; official strict + loose prompt-level following)",
    ))

_ifbench_test = _GENERAL_DATA_DIR / "IFBench" / "test.jsonl"
if _ifbench_test.exists():
    register(Bench(
        id="ifbench",
        name="IFBench",
        data_file=_ifbench_test,
        language="en",
        split="general",
        format="follow",
        prompt_style="follow",
        scorer=_lazy_follow_scorer(),
        scorer_loose=_lazy_follow_scorer_loose(),
        description="Allen AI IFBench (300 test samples, 58 constraint types "
                    "from WildChat; official strict + loose verifiable following)",
    ))

_inverse_test = _GENERAL_DATA_DIR / "InverseIFEval" / "test.jsonl"
if _inverse_test.exists():
    register(Bench(
        id="inverse_ifeval",
        name="Inverse IFEval",
        data_file=_inverse_test,
        language="en",
        split="general",
        format="follow",
        prompt_style="follow",
        scorer=_lazy_inverse_judge(),
        description="Inverse IFEval (1,012 questions, 8 reverse/counter-"
                    "intuitive instruction types; LLM-as-a-Judge scoring)",
    ))
