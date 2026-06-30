from __future__ import annotations

from typing import Any

Cell = tuple[int, int]


def evaluate_relaxed(state_trace: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    frame_states = state_trace["frame_states"]
    occupied_cells = {tuple(cell) for cell in metadata["graph"]["occupied_cells"]}
    start_cell = tuple(metadata["graph"]["start_cell"])
    final_visited = {tuple(cell) for cell in frame_states[-1]["visited_cells"]}
    coverage_ratio = len(final_visited & occupied_cells) / max(len(occupied_cells), 1)

    sequence = _collapse_sequence(frame_states, start_cell)
    legal_steps = 0
    total_steps = max(len(sequence) - 1, 0)
    revisits_detected = False
    seen = set()
    previous = None
    for cell in sequence:
        if cell in seen:
            revisits_detected = True
        seen.add(cell)
        if previous is not None and _is_adjacent(previous, cell):
            legal_steps += 1
        previous = cell

    legal_transition_ratio = 1.0 if total_steps == 0 else legal_steps / total_steps
    on_shape_ratio = sum(1 for frame in frame_states if frame["on_shape"]) / max(len(frame_states), 1)
    score = 0.5 * coverage_ratio + 0.3 * legal_transition_ratio + 0.2 * on_shape_ratio
    completed = coverage_ratio == 1.0 and not revisits_detected

    return {
        "score": score,
        "coverage_ratio": coverage_ratio,
        "legal_transition_ratio": legal_transition_ratio,
        "on_shape_ratio": on_shape_ratio,
        "completed": completed,
        "revisits_detected": revisits_detected,
        "resolved_sequence": [list(cell) for cell in sequence],
    }


def _collapse_sequence(frame_states: list[dict[str, Any]], start_cell: Cell) -> list[Cell]:
    sequence: list[Cell] = [start_cell]
    for frame in frame_states:
        head_cell = frame["head_cell"]
        if head_cell is None:
            continue
        cell = tuple(head_cell)
        if not sequence or sequence[-1] != cell:
            sequence.append(cell)
    return sequence


def _is_adjacent(a: Cell, b: Cell) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
