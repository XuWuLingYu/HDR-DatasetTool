from __future__ import annotations

from typing import Any

from .vision_mapper import compute_board_size, flatten_board

State = tuple[int, ...]
Move = tuple[int, int]

MAX_UNRESOLVED_RATIO = 0.8


def evaluate_strict(state_trace: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    frame_states = state_trace["frame_states"]
    resolved_entries = state_trace["resolved_states"]
    initial_state = flatten_board(metadata["puzzle"]["initial_board"])
    goal_state = flatten_board(metadata["puzzle"]["goal_board"])
    board_size = compute_board_size(metadata)
    bridge_limit = _bridge_limit(board_size)

    raw_sequence: list[State] = [tuple(int(value) for value in item["state"]) for item in resolved_entries]
    resolved_sequence = _best_legal_subsequence(raw_sequence, initial_state, board_size, bridge_limit)
    starts_correctly = bool(resolved_sequence) and resolved_sequence[0] == initial_state
    ends_solved = bool(resolved_sequence) and resolved_sequence[-1] == goal_state

    illegal_transitions: list[dict[str, Any]] = []
    resolved_moves: list[list[int]] = []
    previous = None
    for current in resolved_sequence:
        if previous is not None:
            path = _find_transition_path(previous, current, board_size, bridge_limit)
            if path is None:
                illegal_transitions.append(
                    {
                        "from": list(previous),
                        "to": list(current),
                    }
                )
            else:
                resolved_moves.extend([list(move) for move in path])
        previous = current

    unresolved_frames = sum(1 for frame in frame_states if not frame["resolved"])
    unresolved_ratio = unresolved_frames / max(len(frame_states), 1)
    inventory_violations = sum(
        1
        for frame in frame_states
        if frame["decoded_state"] is not None
        and (
            not frame["tile_count_match"]
            or frame["duplicate_tile_detected"]
            or not frame["blank_count_ok"]
        )
    )
    stable_inventory_violations = sum(
        1
        for frame in frame_states
        if frame["resolved"]
        and (
            not frame["tile_count_match"]
            or frame["duplicate_tile_detected"]
            or not frame["blank_count_ok"]
        )
    )
    passed = (
        starts_correctly
        and ends_solved
        and not illegal_transitions
        and stable_inventory_violations == 0
        and unresolved_ratio <= MAX_UNRESOLVED_RATIO
        and bool(resolved_sequence)
    )
    return {
        "passed": passed,
        "starts_correctly": starts_correctly,
        "ends_solved": ends_solved,
        "resolved_state_sequence": [list(state) for state in resolved_sequence],
        "raw_resolved_state_count": len(raw_sequence),
        "discarded_resolved_states": max(len(raw_sequence) - len(resolved_sequence), 0),
        "resolved_moves": resolved_moves,
        "resolved_move_count": len(resolved_moves),
        "matches_gt_solution_length": len(resolved_moves) == int(metadata["solution"]["move_count"]),
        "illegal_transitions": illegal_transitions,
        "unresolved_frames": unresolved_frames,
        "unresolved_ratio": unresolved_ratio,
        "inventory_violations": inventory_violations,
        "bridge_limit": bridge_limit,
    }


def _best_legal_subsequence(
    sequence: list[State],
    initial_state: State,
    board_size: int,
    bridge_limit: int,
) -> list[State]:
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
            if _find_transition_path(sequence[left], state, board_size, bridge_limit) is None:
                continue
            if dp[left] + 1 > dp[index]:
                dp[index] = dp[left] + 1
                prev[index] = left

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


def _find_transition_path(
    previous: State,
    current: State,
    board_size: int,
    max_depth: int,
) -> list[Move] | None:
    if previous == current:
        return []
    direct = _infer_move(previous, current, board_size)
    if direct is not None:
        return [direct]
    if max_depth <= 1:
        return None

    queue: list[tuple[State, list[Move]]] = [(previous, [])]
    seen = {previous}
    index = 0
    while index < len(queue):
        state, path = queue[index]
        index += 1
        if len(path) >= max_depth:
            continue
        for next_state, move in _forward_moves(state, board_size):
            if next_state in seen:
                continue
            next_path = path + [move]
            if next_state == current:
                return next_path
            seen.add(next_state)
            queue.append((next_state, next_path))
    return None


def _infer_move(previous: State, current: State, board_size: int) -> Move | None:
    previous_blank = previous.index(0)
    current_blank = current.index(0)
    if previous_blank == current_blank:
        return None
    if abs(previous_blank // board_size - current_blank // board_size) + abs(previous_blank % board_size - current_blank % board_size) != 1:
        return None
    if previous[current_blank] == 0 or current[previous_blank] != 0:
        return None
    candidate = list(previous)
    candidate[previous_blank], candidate[current_blank] = candidate[current_blank], candidate[previous_blank]
    if tuple(candidate) != current:
        return None
    return (current_blank, previous_blank)


def _forward_moves(state: State, board_size: int) -> list[tuple[State, Move]]:
    blank = state.index(0)
    row, col = divmod(blank, board_size)
    neighbors: list[tuple[State, Move]] = []
    for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        next_row = row + d_row
        next_col = col + d_col
        if next_row < 0 or next_col < 0 or next_row >= board_size or next_col >= board_size:
            continue
        tile_index = next_row * board_size + next_col
        board = list(state)
        board[blank], board[tile_index] = board[tile_index], board[blank]
        neighbors.append((tuple(board), (tile_index, blank)))
    return neighbors


def _bridge_limit(board_size: int) -> int:
    return 2 if board_size <= 3 else 3
