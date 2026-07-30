"""Unified evaluation package — bench registry + runner.

All benches (MedicalAgentsBench datasets + custom subsets) are registered
in a flat registry. The runner treats every bench identically: load
questions → call model → extract answer → score. Add new benches by
calling ``register()`` in ``benches.py``.
"""
from eval.benches import BENCHES, Bench, list_benches, get_bench  # noqa: F401
