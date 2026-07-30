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

from eval.common import BENCHMARK_DATA_DIR, SUBSET_DATA_FILE, load_jsonl


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
    """

    id: str
    name: str
    data_file: Path
    language: str = "en"
    split: str = "test_hard"
    description: str = ""

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

        if self.language == "zh":
            return [
                {"role": "system", "content": _ZH_SYSTEM},
                {"role": "user", "content": f"{q_text}\n{options_listed}"},
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



# -- sampled-10 subsets (data_sampled10/) --------------------------------
# 10 MedicalAgentsBench datasets x 2 languages (EN + ZH) x 2 splits
# (test_hard + test). Each dataset has ~100 questions per hard split.
# These live in csp_dev/eval/data_sampled10/<Dataset>/.

_SAMPLED_DIR = Path(__file__).resolve().parent / "data_sampled10"

# Folder names in data_sampled10 use Title-Case (e.g. "MedQA"), but the
# bench ids use the lowercase canonical names from _BENCH_NAMES.
_SAMPLED_FOLDER = {
    "medqa": "MedQA",
    "pubmedqa": "PubMedQA",
    "medmcqa": "MedMCQA",
    "mmlu": "MMLU",
    "mmlu-pro": "MMLU-Pro",
    "medbullets": "MedBullets",
    "afrimedqa": "AfrimedQA",
    "medexqa": "MedExQA",
    "medxpertqa-r": "MedXpertQA-R",
    "medxpertqa-u": "MedXpertQA-U",
}

_SAMPLED_SPLITS = [
    ("hard", "test_hard-00000-of-00001"),
    ("easy", "test-00000-of-00001"),
]

for _ds in _SAMPLED_FOLDER:
    _folder = _SAMPLED_FOLDER[_ds]
    _ds_dir = _SAMPLED_DIR / _folder
    if not _ds_dir.exists():
        continue
    for _split_short, _split_prefix in _SAMPLED_SPLITS:
        for _lang_code, _lang_label in [("en", ""), ("zh", "_zh")]:
            _suffix = "" if _lang_code == "en" else "-zh_pure_standard"
            _fname = f"{_split_prefix}{_suffix}.jsonl"
            _fpath = _ds_dir / _fname
            if not _fpath.exists():
                continue
            _bid = f"{_ds}_{_lang_code}_{_split_short}"
            _bname = _BENCH_NAMES[_ds]
            if _lang_code == "zh":
                _bname += " (中文)"  # (中文)
            _bname += f" [s10-{_split_short}]"
            register(Bench(
                id=_bid,
                name=_bname,
                data_file=_fpath,
                language="zh" if _lang_code == "zh" else "en",
                split=f"sampled10_{_split_short}",
                description=_BENCH_DESC.get(_ds, "") + " (sampled 10%)",
            ))

# -- custom: MedQA Chinese subset (350 questions) --------------------------

register(Bench(
    id="medqa_cn",
    name="MedQA \u4e2d\u6587\u5b50\u96c6",  # MedQA 中文子集
    data_file=SUBSET_DATA_FILE,
    language="zh",
    split="subset",
    description="\u81ea\u5b9a\u4e49 MedQA \u4e2d\u6587 350 \u9898\u5b50\u96c6",
    # "自定义 MedQA 中文 350 题子集"
))
