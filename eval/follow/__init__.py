"""Instruction-following verifiers for IFEval / IFBench / Inverse IFEval.

This module wraps two vendored, faithful verifier packs:

* ``instruction_following_eval/`` -- the original Google IFEval checkers
  (25 verifiable instruction types) and their official ``INSTRUCTION_DICT``.
* ``ifbench/`` -- the Allen AI "Generalizing Verifiable Instruction
  Following" (IFBench) checkers (58 constraint types).

Both are Apache-2.0 licensed and are vendored unmodified (except that
``INSTRUCTION_DICT`` import paths resolve through this package). They rely on
NLTK corpora (``punkt`` / ``punkt_tab`` / ``stopwords`` /
``averaged_perceptron_tagger_eng``) plus the ``emoji``/``syllapy``/
``langdetect`` packages; the corpora are kept in a private, repo-local
``.nltk_data`` directory so the eval does not depend on any user NLTK config.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK data: force a private, repo-local location and ensure the corpora the
# verifiers need are present (download them once if missing). This must run
# before the vendored modules are imported, because they call
# ``os.environ.setdefault("NLTK_DATA", ...)`` / ``nltk.download`` at import.
# ---------------------------------------------------------------------------

_NLTK_DIR = Path(__file__).resolve().parent / ".nltk_data"
os.environ["NLTK_DATA"] = str(_NLTK_DIR)

_NLTK_RESOURCES = (
    "tokenizers/punkt",
    "tokenizers/punkt_tab",
    "corpora/stopwords",
    "taggers/averaged_perceptron_tagger_eng",
)


def ensure_nltk_data() -> None:
    """Make sure the NLTK corpora used by the IFEval/IFBench checkers are
    available, downloading them (into NLTK's normal cache, e.g.
    ``~/nltk_data``) on first use. Safe to call repeatedly; offline or
    permission-blocked environments simply degrade gracefully."""
    import nltk

    _NLTK_DIR.mkdir(parents=True, exist_ok=True)
    for res in _NLTK_RESOURCES:
        try:
            nltk.data.find(res)
        except LookupError:
            try:
                nltk.download(res, quiet=True)
            except Exception:  # noqa: BLE001 -- offline runs should still work
                log.warning("NLTK resource %s unavailable; verifier may degrade",
                            res)


# ---------------------------------------------------------------------------
# Verifier pack loading. Both packs declare globally-unique checker class
# names (only the ``Instruction`` base class is shared), so they are imported
# as separate modules and never merged into one namespace.
# ---------------------------------------------------------------------------

_FOLLOW_DIR = Path(__file__).resolve().parent

# Path setup for the vendored packs:
# * the original IFEval pack is imported as ``instruction_following_eval``,
#   so its *parent* (``eval/follow``) must be importable;
# * the IFBench pack imports bare modules (``import instructions``), so it is
#   added to ``sys.path`` directly.
if str(_FOLLOW_DIR) not in sys.path:
    sys.path.insert(0, str(_FOLLOW_DIR))
_IFB_DIR = _FOLLOW_DIR / "ifbench"
if str(_IFB_DIR) not in sys.path:
    sys.path.insert(0, str(_IFB_DIR))

# ---------------------------------------------------------------------------
# Public scoring helpers
# ---------------------------------------------------------------------------

_STYLE_CHECKERS = None  # lazily imported registry
_IFB_CHECKERS = None


def _load_ifeval():
    global _STYLE_CHECKERS
    if _STYLE_CHECKERS is None:
        ensure_nltk_data()
        import instruction_following_eval.instructions_registry as _reg
        _STYLE_CHECKERS = _reg.INSTRUCTION_DICT
    return _STYLE_CHECKERS


def _load_ifbench():
    global _IFB_CHECKERS
    if _IFB_CHECKERS is None:
        ensure_nltk_data()
        import instructions_registry as _reg  # from the ifbench pack
        _IFB_CHECKERS = _reg.INSTRUCTION_DICT
    return _IFB_CHECKERS


def verify_follow_strict(problem: dict, response: str) -> bool:
    """Strict IFEval-style instruction-following check.

    Returns ``True`` when *every* verifiable instruction in the problem is
    satisfied by ``response`` (matching the official "strict" prompt-level
    metric). Used for the IFEval and IFBench benches.
    """
    if not response or not response.strip():
        return False
    instr_ids = problem.get("instruction_id_list") or []
    kwargs = problem.get("kwargs") or []
    if not instr_ids:
        return bool(response.strip())

    registry = _load_follow_registry(instr_ids)

    for i, instr_id in enumerate(instr_ids):
        cls = registry.get(instr_id)
        if cls is None:
            return False
        inst = cls(instr_id)
        kw = kwargs[i] if i < len(kwargs) and kwargs[i] else {}
        kw = {k: v for k, v in kw.items() if v is not None}
        try:
            inst.build_description(**kw)
        except Exception:  # noqa: BLE001 -- malformed kwargs
            return False
        args = inst.get_instruction_args()
        if args and "prompt" in args:
            try:
                inst.build_description(prompt=problem.get("question", ""))
            except Exception:  # noqa: BLE001
                return False
        try:
            if not inst.check_following(response):
                return False
        except Exception:  # noqa: BLE001 -- a broken checker won't abort the run
            return False
    return True


def _load_follow_registry(instr_ids):
    """Return the registry that covers all the given instruction ids."""
    if _IFB_CHECKERS is None and _STYLE_CHECKERS is None:
        # prime both
        _load_ifeval()
        _load_ifbench()
    # If any id is not in the IFBench (superset) registry, fall back/merge.
    if all(i in _IFB_CHECKERS for i in instr_ids):
        return _IFB_CHECKERS
    return _STYLE_CHECKERS


def verify_follow_loose(problem: dict, response: str) -> bool:
    """Loose IFEval-style instruction-following check (official "upper bound").

    Mirrors ``test_instruction_following_loose``: each instruction is deemed
    followed if ANY of 8 response variants (strip first/last/both lines and
    remove ``*`` markers) satisfies it, and the prompt counts as followed only
    when every instruction is. Used as the second official IFEval metric.
    """
    if not response:
        return False
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        response_remove_first.replace("*", ""),
        response_remove_last.replace("*", ""),
        response_remove_both.replace("*", ""),
    ]

    instr_ids = problem.get("instruction_id_list") or []
    kwargs = problem.get("kwargs") or []
    if not instr_ids:
        return bool(response.strip())

    registry = _load_follow_registry(instr_ids)

    for i, instr_id in enumerate(instr_ids):
        cls = registry.get(instr_id)
        if cls is None:
            return False
        inst = cls(instr_id)
        kw = kwargs[i] if i < len(kwargs) and kwargs[i] else {}
        kw = {k: v for k, v in kw.items() if v is not None}
        try:
            inst.build_description(**kw)
        except Exception:  # noqa: BLE001
            continue
        args = inst.get_instruction_args()
        if args and "prompt" in args:
            try:
                inst.build_description(prompt=problem.get("question", ""))
            except Exception:  # noqa: BLE001
                continue
        followed = False
        for variant in all_responses:
            if variant.strip():
                try:
                    if inst.check_following(variant):
                        followed = True
                        break
                except Exception:  # noqa: BLE001
                    continue
        if not followed:
            return False
    return True
