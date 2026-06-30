from .one_line_drawing import evaluate_sample as evaluate_one_line_drawing_sample
from .sokoban import evaluate_sample as evaluate_sokoban_sample
from .sliding_puzzle import evaluate_sample as evaluate_sliding_puzzle_sample
from .water_pour import evaluate_sample as evaluate_water_pour_sample

EVALUATORS = {
    "one_line_drawing": evaluate_one_line_drawing_sample,
    "sokoban": evaluate_sokoban_sample,
    "sliding_puzzle": evaluate_sliding_puzzle_sample,
    "water_pour": evaluate_water_pour_sample,
}

__all__ = [
    "EVALUATORS",
    "evaluate_one_line_drawing_sample",
    "evaluate_sokoban_sample",
    "evaluate_sliding_puzzle_sample",
    "evaluate_water_pour_sample",
]
