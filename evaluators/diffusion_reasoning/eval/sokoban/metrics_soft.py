from __future__ import annotations

from typing import Any

from .metrics_strict import BRIDGE_LIMIT, _best_legal_subsequence, _find_transition_path


def evaluate_relaxed(state_trace: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    frame_states = state_trace["frame_states"]
    resolved_entries = state_trace["resolved_states"]
    initial_distance = int(state_trace["layout"]["initial_box_distance"])
    gt_action_count = int(metadata["solution"]["action_count"])
    wall_rows = tuple(str(row) for row in metadata["puzzle"]["wall_rows"])
    initial_state = (
        tuple(int(value) for value in metadata["puzzle"]["initial_player"]),
        tuple(int(value) for value in metadata["puzzle"]["initial_box"]),
    )

    if not frame_states:
        return {
            "score": 0.0,
            "progress_ratio": 0.0,
            "final_state_quality": 0.0,
            "rule_adherence_ratio": 0.0,
            "observability_ratio": 0.0,
            "illegal_transition_ratio": 0.0,
            "completed": False,
            "best_box_distance": initial_distance,
            "final_box_distance": initial_distance,
        }

    stable_frames = [frame for frame in frame_states if frame["resolved"]]
    if not stable_frames:
        stable_frames = frame_states

    rule_adherence_ratio = sum(
        1
        for frame in stable_frames
        if frame["resolved"]
        and frame["player_on_floor"]
        and frame["box_on_floor"]
        and not frame["overlap_detected"]
    ) / max(len(stable_frames), 1)

    progress_values = [
        _normalized_progress(int(frame["box_target_distance"]), initial_distance)
        for frame in stable_frames
        if frame["box_target_distance"] is not None
    ]
    progress_ratio = max(progress_values, default=0.0)

    final_resolved = next((frame for frame in reversed(stable_frames) if frame["resolved"]), None)
    if final_resolved is None or final_resolved["box_target_distance"] is None:
        final_box_distance = initial_distance
        final_state_quality = 0.0
        completed = False
    else:
        final_box_distance = int(final_resolved["box_target_distance"])
        completed = bool(final_resolved["box_on_target"])
        final_state_quality = 1.0 if completed else _normalized_progress(final_box_distance, initial_distance)

    raw_sequence = [
        (
            tuple(int(value) for value in item["state"]["player"]),
            tuple(int(value) for value in item["state"]["box"]),
        )
        for item in resolved_entries
    ]
    cleaned_sequence = _best_legal_subsequence(raw_sequence, initial_state, wall_rows, BRIDGE_LIMIT)
    illegal_transitions = 0
    total_transitions = max(len(cleaned_sequence) - 1, 0)
    for previous, current in zip(cleaned_sequence[:-1], cleaned_sequence[1:]):
        if _find_transition_path(previous, current, wall_rows, BRIDGE_LIMIT) is None:
            illegal_transitions += 1
    illegal_transition_ratio = 1.0 if total_transitions == 0 else 1.0 - (illegal_transitions / total_transitions)
    observability_ratio = min(len(resolved_entries) / max(gt_action_count + 1, 1), 1.0)

    score = (
        0.50 * progress_ratio
        + 0.20 * final_state_quality
        + 0.15 * rule_adherence_ratio
        + 0.15 * observability_ratio
    )

    distinct_states = {
        (
            tuple(int(value) for value in item["state"]["player"]),
            tuple(int(value) for value in item["state"]["box"]),
        )
        for item in resolved_entries
    }
    if progress_ratio < 0.15 and observability_ratio < 0.25:
        score = min(score, 0.15)
    if len(distinct_states) < 3 and not completed:
        score = min(score, 0.10)
    if progress_ratio <= 0.0:
        score = min(score, 0.20)
    if illegal_transition_ratio < 0.5:
        score *= max(illegal_transition_ratio, 0.0)
    if completed and rule_adherence_ratio >= 0.99 and illegal_transition_ratio >= 0.99:
        score = 1.0

    best_box_distance = initial_distance
    if progress_values:
        best_box_distance = min(
            int(frame["box_target_distance"])
            for frame in stable_frames
            if frame["box_target_distance"] is not None
        )

    return {
        "score": max(0.0, min(score, 1.0)),
        "progress_ratio": progress_ratio,
        "final_state_quality": final_state_quality,
        "rule_adherence_ratio": rule_adherence_ratio,
        "observability_ratio": observability_ratio,
        "illegal_transition_ratio": illegal_transition_ratio,
        "completed": completed,
        "best_box_distance": best_box_distance,
        "final_box_distance": final_box_distance,
    }


def _normalized_progress(current_distance: int, initial_distance: int) -> float:
    if initial_distance <= 0:
        return 1.0 if current_distance <= 0 else 0.0
    improvement = initial_distance - current_distance
    return max(0.0, min(improvement / initial_distance, 1.0))
