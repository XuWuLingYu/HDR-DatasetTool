from __future__ import annotations

from collections import deque
from typing import Any

State = tuple[tuple[int, ...], ...]
Move = tuple[int, int, int]

MAX_UNRESOLVED_RATIO = 0.35
# Two-row level-4 videos often miss several stable intermediate states even when
# the recovered states are all legal. Allow a slightly wider legal bridge so
# strict scoring reflects the intended GT2GT correctness rather than CV sparsity.
MAX_BRIDGED_MOVE_GAP = 6


def evaluate_strict(state_trace: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    frame_states = state_trace["frame_states"]
    resolved_entries = state_trace["resolved_states"]
    initial_state = tuple(tuple(tube) for tube in metadata["puzzle"]["initial_tubes"])
    capacity = int(metadata["puzzle"]["capacity"])

    raw_sequence: list[State] = [tuple(tuple(tube) for tube in item["state"]) for item in resolved_entries]
    resolved_sequence = _best_legal_subsequence(raw_sequence, initial_state, capacity)
    starts_correctly = bool(resolved_sequence) and resolved_sequence[0] == initial_state
    ends_solved = bool(resolved_sequence) and _is_solved(resolved_sequence[-1], capacity)
    illegal_transitions: list[dict[str, Any]] = []
    resolved_moves: list[list[int]] = []

    previous = None
    for current in resolved_sequence:
        if previous is not None:
            path = _find_transition_path(previous, current, capacity, max_depth=MAX_BRIDGED_MOVE_GAP)
            if path is None:
                illegal_transitions.append(
                    {
                        "from": [list(tube) for tube in previous],
                        "to": [list(tube) for tube in current],
                        "reported_move": None,
                    }
                )
            else:
                resolved_moves.extend([list(move) for move in path])
        previous = current

    conservation_violations = sum(1 for frame in frame_states if frame["resolved"] and not frame["conservation_ok"])
    placement_violations = sum(1 for frame in frame_states if frame["resolved"] and not frame["placement_ok"])
    capacity_violations = sum(1 for frame in frame_states if frame["resolved"] and not frame["capacity_ok"])
    unresolved_frames = sum(1 for frame in frame_states if not frame["resolved"] and not frame["in_motion"])
    unresolved_ratio = unresolved_frames / max(len(frame_states), 1)

    passed = (
        starts_correctly
        and ends_solved
        and not illegal_transitions
        and conservation_violations == 0
        and placement_violations == 0
        and capacity_violations == 0
        and bool(resolved_sequence)
        and unresolved_ratio <= MAX_UNRESOLVED_RATIO
    )
    return {
        "passed": passed,
        "starts_correctly": starts_correctly,
        "ends_solved": ends_solved,
        "resolved_state_sequence": [[list(tube) for tube in state] for state in resolved_sequence],
        "raw_resolved_state_count": len(raw_sequence),
        "discarded_resolved_states": max(len(raw_sequence) - len(resolved_sequence), 0),
        "resolved_moves": resolved_moves,
        "resolved_move_count": len(resolved_moves),
        "matches_gt_solution_length": len(resolved_moves) == int(metadata["solution"]["move_count"]),
        "illegal_transitions": illegal_transitions,
        "conservation_violations": conservation_violations,
        "placement_violations": placement_violations,
        "capacity_violations": capacity_violations,
        "unresolved_frames": unresolved_frames,
        "unresolved_ratio": unresolved_ratio,
    }


def _is_solved(state: State, capacity: int) -> bool:
    for tube in state:
        if not tube:
            continue
        if len(tube) != capacity or len(set(tube)) != 1:
            return False
    return True


def _is_legal_transition(previous: State, current: State, capacity: int) -> bool:
    return _find_transition_path(previous, current, capacity, max_depth=MAX_BRIDGED_MOVE_GAP) is not None


def _infer_move(previous: State, current: State, capacity: int) -> Move | None:
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


def _best_legal_subsequence(sequence: list[State], initial_state: State, capacity: int) -> list[State]:
    if not sequence:
        return []
    n = len(sequence)
    dp = [0] * n
    prev = [-1] * n
    for index, state in enumerate(sequence):
        if state == initial_state:
            dp[index] = 1
        for left in range(index):
            if dp[left] <= 0:
                continue
            if _find_transition_path(sequence[left], state, capacity, max_depth=MAX_BRIDGED_MOVE_GAP) is None:
                continue
            if dp[left] + 1 > dp[index]:
                dp[index] = dp[left] + 1
                prev[index] = left

    solved_indices = [index for index, state in enumerate(sequence) if dp[index] > 0 and _is_solved(state, capacity)]
    if solved_indices:
        end = max(solved_indices, key=lambda index: dp[index])
    else:
        reachable = [index for index, value in enumerate(dp) if value > 0]
        if not reachable:
            return []
        end = max(reachable, key=lambda index: dp[index])

    ordered: list[State] = []
    cursor = end
    while cursor != -1:
        ordered.append(sequence[cursor])
        cursor = prev[cursor]
    ordered.reverse()
    deduped: list[State] = []
    for state in ordered:
        if not deduped or deduped[-1] != state:
            deduped.append(state)
    return deduped


def _find_transition_path(previous: State, current: State, capacity: int, max_depth: int) -> list[Move] | None:
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


def _forward_moves(state: State, capacity: int) -> list[tuple[State, Move]]:
    moves: list[tuple[State, Move]] = []
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
