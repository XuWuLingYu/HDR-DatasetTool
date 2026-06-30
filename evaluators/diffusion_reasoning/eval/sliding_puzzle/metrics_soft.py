from __future__ import annotations

from typing import Any

from .metrics_strict import _bridge_limit, _find_transition_path
from .vision_mapper import compute_board_size, flatten_board, manhattan_distance


def evaluate_relaxed(state_trace: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    frame_states = state_trace["frame_states"]
    resolved_entries = state_trace["resolved_states"]
    initial_state = flatten_board(metadata["puzzle"]["initial_board"])
    goal_state = flatten_board(metadata["puzzle"]["goal_board"])
    board_size = compute_board_size(metadata)
    bridge_limit = _bridge_limit(board_size)
    expected_state_count = int(metadata["solution"]["move_count"]) + 1
    initial_distance = manhattan_distance(initial_state, board_size)

    if not frame_states:
        return {
            "score": 0.0,
            "progress_ratio": 0.0,
            "final_state_quality": 0.0,
            "rule_adherence_ratio": 0.0,
            "observability_ratio": 0.0,
            "illegal_transition_ratio": 0.0,
            "correct_tile_ratio": 0.0,
        }

    stable_frames = [frame for frame in frame_states if frame["decoded_state"] is not None]
    if not stable_frames:
        stable_frames = frame_states

    rule_adherence_ratio = sum(
        1
        for frame in stable_frames
        if frame["decoded_state"] is not None
        and frame["tile_count_match"]
        and not frame["duplicate_tile_detected"]
        and frame["blank_count_ok"]
    ) / max(len(stable_frames), 1)

    progress_values: list[float] = []
    correct_tile_values: list[float] = []
    for frame in stable_frames:
        state = frame["decoded_state"]
        if state is None:
            continue
        distance = manhattan_distance(tuple(state), board_size)
        progress_values.append(_normalized_progress(distance, initial_distance))
        correct_tile_values.append(float(frame["correct_tile_ratio"]))

    progress_ratio = max(progress_values, default=0.0)
    correct_tile_ratio = max(correct_tile_values, default=0.0)

    final_resolved = next((frame for frame in reversed(stable_frames) if frame["decoded_state"] is not None), None)
    completed = False
    if final_resolved is None:
        final_state_quality = 0.0
    else:
        final_state = tuple(int(value) for value in final_resolved["decoded_state"])
        final_distance = manhattan_distance(final_state, board_size)
        final_progress = _normalized_progress(final_distance, initial_distance)
        completed = final_state == goal_state
        final_state_quality = 1.0 if completed else (0.7 * final_progress + 0.3 * float(final_resolved["correct_tile_ratio"]))

    illegal_transitions = 0
    total_transitions = max(len(resolved_entries) - 1, 0)
    for previous, current in zip(resolved_entries[:-1], resolved_entries[1:]):
        if _find_transition_path(
            tuple(int(value) for value in previous["state"]),
            tuple(int(value) for value in current["state"]),
            board_size,
            bridge_limit,
        ) is None:
            illegal_transitions += 1
    illegal_transition_ratio = 1.0 if total_transitions == 0 else 1.0 - (illegal_transitions / total_transitions)
    observability_ratio = min(len(resolved_entries) / max(expected_state_count, 1), 1.0)

    score = (
        0.45 * progress_ratio
        + 0.20 * final_state_quality
        + 0.20 * rule_adherence_ratio
        + 0.15 * observability_ratio
    )
    if completed and rule_adherence_ratio >= 0.99 and illegal_transition_ratio >= 0.99:
        score = 1.0

    return {
        "score": score,
        "progress_ratio": progress_ratio,
        "final_state_quality": final_state_quality,
        "rule_adherence_ratio": rule_adherence_ratio,
        "observability_ratio": observability_ratio,
        "illegal_transition_ratio": illegal_transition_ratio,
        "correct_tile_ratio": correct_tile_ratio,
        "completed": completed,
    }


def _normalized_progress(current_distance: int, initial_distance: int) -> float:
    if initial_distance <= 0:
        return 1.0 if current_distance <= 0 else 0.0
    improvement = initial_distance - current_distance
    return max(0.0, min(improvement / initial_distance, 1.0))
