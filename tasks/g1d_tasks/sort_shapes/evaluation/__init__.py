"""Stage-based evaluation for the G1-D shape-sorting task."""

from .config import DEFAULT_EVALUATION, STAGE_NAMES, load_evaluation_settings
from .evaluator import StageEvaluator
from .recorder import EvaluationRecorderManagerCfg
from .state_machine import advance_ordered_stages

__all__ = [
    "DEFAULT_EVALUATION",
    "STAGE_NAMES",
    "EvaluationRecorderManagerCfg",
    "StageEvaluator",
    "advance_ordered_stages",
    "load_evaluation_settings",
]
