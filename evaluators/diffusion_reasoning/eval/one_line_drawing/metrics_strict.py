from __future__ import annotations

from typing import Any

Cell = tuple[int, int]


def evaluate_strict(state_trace: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    frame_states = state_trace["frame_states"]
    occupied_cells = {tuple(cell) for cell in metadata["graph"]["occupied_cells"]}
    start_cell = tuple(metadata["graph"]["start_cell"])
    expected_count = int(metadata["solution"]["cell_count"])

    resolved_sequence = _collapse_sequence(frame_states, start_cell)
    visited_cells = set()
    repeated_cells: list[list[int]] = []
    illegal_transitions: list[dict[str, list[int]]] = []
    unresolved_frames = 0

    previous = None
    for cell in resolved_sequence:
        if cell in visited_cells:
            repeated_cells.append(list(cell))
        if previous is not None and not _is_adjacent(previous, cell):
            illegal_transitions.append({"from": list(previous), "to": list(cell)})
        visited_cells.add(cell)
        previous = cell

    for frame in frame_states:
        if frame["head_cell"] is None:
            unresolved_frames += 1

    off_shape_frames = sum(1 for frame in frame_states if not frame["on_shape"])
    coverage_complete = len(frame_states[-1]["visited_cells"]) == expected_count
    missing_cells = sorted(occupied_cells - {tuple(cell) for cell in frame_states[-1]["visited_cells"]})
    starts_correctly = bool(resolved_sequence) and resolved_sequence[0] == start_cell

    passed = (
        starts_correctly
        and coverage_complete
        and not repeated_cells
        and not illegal_transitions
        and not missing_cells
        and off_shape_frames == 0
    )
    return {
        "passed": passed,
        "resolved_sequence": [list(cell) for cell in resolved_sequence],
        "coverage_complete": coverage_complete,
        "missing_cells": [list(cell) for cell in missing_cells],
        "repeated_cells": repeated_cells,
        "illegal_transitions": illegal_transitions,
        "unresolved_frames": unresolved_frames,
        "off_shape_frames": off_shape_frames,
        "starts_correctly": starts_correctly,
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
