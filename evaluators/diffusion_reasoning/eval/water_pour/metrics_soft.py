from __future__ import annotations

from collections import deque
from typing import Any

MAX_BRIDGED_MOVE_GAP = 6
Move = tuple[int, int, int]


def evaluate_relaxed(state_trace: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    frame_states = state_trace["frame_states"]
    capacity = int(metadata["puzzle"]["capacity"])
    tube_count = int(metadata["puzzle"]["tube_count"])
    expected_state_count = int(metadata["solution"]["move_count"]) + 1
    if not frame_states:
        return {
            "score": 0.0,
            "rule_adherence_ratio": 0.0,
            "progress_ratio": 0.0,
            "final_state_quality": 0.0,
            "observability_ratio": 0.0,
            "solved_tube_ratio": 0.0,
            "monochrome_tube_ratio": 0.0,
            "conservation_ok_ratio": 0.0,
            "illegal_transition_ratio": 0.0,
        }

    stationary_frames = [frame for frame in frame_states if not frame["in_motion"]]
    if not stationary_frames:
        stationary_frames = frame_states
    valid_stationary = [
        frame["conservation_ok"] and frame["capacity_ok"] and frame["placement_ok"]
        for frame in stationary_frames
    ]
    rule_adherence_ratio = sum(1 for flag in valid_stationary if flag) / len(stationary_frames)
    conservation_ok_ratio = sum(1 for frame in stationary_frames if frame["conservation_ok"]) / len(stationary_frames)

    progress_values = []
    solved_ratios = []
    monochrome_ratios = []
    for frame in stationary_frames:
        if frame["tube_state"] is None:
            continue
        solved_ratio = frame["solved_tube_count"] / max(tube_count, 1)
        monochrome_ratio = frame["monochrome_tube_count"] / max(tube_count, 1)
        progress = (
            0.5 * float(frame["block_progress_ratio"])
            + 0.25 * solved_ratio
            + 0.25 * monochrome_ratio
        )
        progress_values.append(progress)
        solved_ratios.append(solved_ratio)
        monochrome_ratios.append(monochrome_ratio)

    progress_ratio = max(progress_values, default=0.0)
    solved_tube_ratio = max(solved_ratios, default=0.0)
    monochrome_tube_ratio = max(monochrome_ratios, default=0.0)

    final_resolved = next((frame for frame in reversed(stationary_frames) if frame["tube_state"] is not None), None)
    if final_resolved is None:
        final_state_quality = 0.0
        completed = False
    else:
        final_solved_ratio = final_resolved["solved_tube_count"] / max(tube_count, 1)
        final_monochrome_ratio = final_resolved["monochrome_tube_count"] / max(tube_count, 1)
        completed = _is_solved_state(tuple(tuple(tube) for tube in final_resolved["tube_state"]), capacity)
        final_state_quality = 1.0 if completed else (
            0.5 * float(final_resolved["block_progress_ratio"])
            + 0.3 * final_solved_ratio
            + 0.2 * final_monochrome_ratio
        )

    resolved_entries = state_trace["resolved_states"]
    illegal_transitions = 0
    total_transitions = max(len(resolved_entries) - 1, 0)
    for previous, current in zip(resolved_entries[:-1], resolved_entries[1:]):
        if not _is_legal_transition(
            tuple(tuple(tube) for tube in previous["state"]),
            tuple(tuple(tube) for tube in current["state"]),
            capacity,
        ):
            illegal_transitions += 1
    illegal_transition_ratio = 1.0 if total_transitions == 0 else 1.0 - (illegal_transitions / total_transitions)
    observability_ratio = min(len(resolved_entries) / max(expected_state_count, 1), 1.0)

    if completed:
        progress_ratio = 1.0
        solved_tube_ratio = 1.0
        monochrome_tube_ratio = 1.0

    score = (
        0.45 * rule_adherence_ratio
        + 0.35 * progress_ratio
        + 0.15 * final_state_quality
        + 0.05 * observability_ratio
    )
    if completed and rule_adherence_ratio >= 0.99 and illegal_transition_ratio >= 0.99:
        score = 1.0
    return {
        "score": score,
        "rule_adherence_ratio": rule_adherence_ratio,
        "progress_ratio": progress_ratio,
        "final_state_quality": final_state_quality,
        "observability_ratio": observability_ratio,
        "solved_tube_ratio": solved_tube_ratio,
        "monochrome_tube_ratio": monochrome_tube_ratio,
        "conservation_ok_ratio": conservation_ok_ratio,
        "illegal_transition_ratio": illegal_transition_ratio,
    }


def _is_legal_transition(previous: tuple[tuple[int, ...], ...], current: tuple[tuple[int, ...], ...], capacity: int) -> bool:
    return _find_transition_path(previous, current, capacity, max_depth=MAX_BRIDGED_MOVE_GAP) is not None


def _infer_move(previous: tuple[tuple[int, ...], ...], current: tuple[tuple[int, ...], ...], capacity: int) -> Move | None:
    source = None
    dest = None
    moved_blocks: tuple[int, ...] | None = None
    for index, (left, right) in enumerate(zip(previous, current)):
        if left == right:
            continue
        if len(left) > len(right) and left[: len(right)] == right:
            source = index
            moved_blocks = left[len(right) :]
        elif len(left) < len(right) and left == right[: len(left)]:
            dest = index
            if moved_blocks is None:
                moved_blocks = right[len(left) :]
        else:
            return None
    if source is None or dest is None or moved_blocks is None or source == dest:
        return None
    moved_count = len(moved_blocks)
    if moved_count <= 0 or len(current[dest]) > capacity:
        return None
    moved_color = moved_blocks[0]
    if any(block != moved_color for block in moved_blocks):
        return None
    if previous[source][-moved_count:] != moved_blocks:
        return None
    if current[dest][-moved_count:] != moved_blocks:
        return None
    if previous[dest] and previous[dest][-1] != moved_color:
        return None
    source_run_length = _top_run_length(previous[source])
    if moved_count != min(source_run_length, capacity - len(previous[dest])):
        return None
    return (source, dest, moved_count)


def _find_transition_path(
    previous: tuple[tuple[int, ...], ...],
    current: tuple[tuple[int, ...], ...],
    capacity: int,
    max_depth: int,
) -> list[Move] | None:
    direct = _infer_move(previous, current, capacity)
    if direct is not None:
        return [direct]
    if max_depth <= 1:
        return None

    queue = deque([(previous, [])])
    seen = {previous}
    while queue:
        state, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for next_state, move in _forward_moves(state, capacity):
            if next_state in seen:
                continue
            next_path = path + [move]
            if next_state == current:
                return next_path
            seen.add(next_state)
            queue.append((next_state, next_path))
    return None


def _forward_moves(
    state: tuple[tuple[int, ...], ...],
    capacity: int,
) -> list[tuple[tuple[tuple[int, ...], ...], tuple[int, int]]]:
    moves: list[tuple[tuple[tuple[int, ...], ...], Move]] = []
    for source_idx, source in enumerate(state):
        if not source:
            continue
        block = source[-1]
        run_length = _top_run_length(source)
        for dest_idx, dest in enumerate(state):
            if source_idx == dest_idx or len(dest) >= capacity:
                continue
            if dest and dest[-1] != block:
                continue
            moved_count = min(run_length, capacity - len(dest))
            if moved_count <= 0:
                continue
            next_state = [list(tube) for tube in state]
            moved_blocks = next_state[source_idx][-moved_count:]
            del next_state[source_idx][-moved_count:]
            next_state[dest_idx].extend(moved_blocks)
            moves.append((tuple(tuple(tube) for tube in next_state), (source_idx, dest_idx, moved_count)))
    return moves


def _top_run_length(tube: tuple[int, ...]) -> int:
    if not tube:
        return 0
    top = tube[-1]
    run_length = 0
    for block in reversed(tube):
        if block != top:
            break
        run_length += 1
    return run_length


def _is_solved_state(state: tuple[tuple[int, ...], ...], capacity: int) -> bool:
    for tube in state:
        if not tube:
            continue
        if len(tube) != capacity or len(set(tube)) != 1:
            return False
    return True
