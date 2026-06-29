from __future__ import annotations

from collections import deque
from typing import Any

Cell = tuple[int, int]
State = tuple[Cell, Cell]

MAX_UNRESOLVED_RATIO = 0.80
BRIDGE_LIMIT = 3


def evaluate_strict(state_trace: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    frame_states = state_trace["frame_states"]
    resolved_entries = state_trace["resolved_states"]
    wall_rows = tuple(str(row) for row in metadata["puzzle"]["wall_rows"])
    initial_state: State = (
        tuple(int(value) for value in metadata["puzzle"]["initial_player"]),
        tuple(int(value) for value in metadata["puzzle"]["initial_box"]),
    )
    target = tuple(int(value) for value in metadata["puzzle"]["target"])

    raw_sequence: list[State] = [
        (
            tuple(int(value) for value in item["state"]["player"]),
            tuple(int(value) for value in item["state"]["box"]),
        )
        for item in resolved_entries
    ]
    resolved_sequence = _best_legal_subsequence(raw_sequence, initial_state, wall_rows, BRIDGE_LIMIT)
    starts_correctly = bool(resolved_sequence) and resolved_sequence[0] == initial_state
    ends_solved = bool(resolved_sequence) and resolved_sequence[-1][1] == target

    illegal_transitions: list[dict[str, Any]] = []
    resolved_actions: list[str] = []
    previous = None
    for current in resolved_sequence:
        if previous is not None:
            path = _find_transition_path(previous, current, wall_rows, BRIDGE_LIMIT)
            if path is None:
                illegal_transitions.append(
                    {
                        "from": {"player": list(previous[0]), "box": list(previous[1])},
                        "to": {"player": list(current[0]), "box": list(current[1])},
                    }
                )
            else:
                resolved_actions.extend(path)
        previous = current

    unresolved_frames = sum(1 for frame in frame_states if not frame["resolved"])
    unresolved_ratio = unresolved_frames / max(len(frame_states), 1)
    entity_validity_violations = sum(
        1
        for frame in frame_states
        if frame["resolved"]
        and (
            not frame["player_on_floor"]
            or not frame["box_on_floor"]
            or frame["overlap_detected"]
        )
    )
    gt_actions = metadata["solution"]["actions"]
    gt_pushes = sum(1 for item in gt_actions if item["push"])
    recovered_pushes = 0
    for previous, current in zip(resolved_sequence, resolved_sequence[1:]):
        if previous[1] != current[1]:
            recovered_pushes += 1

    passed = (
        starts_correctly
        and ends_solved
        and not illegal_transitions
        and entity_validity_violations == 0
        and unresolved_ratio <= MAX_UNRESOLVED_RATIO
        and bool(resolved_sequence)
    )
    return {
        "passed": passed,
        "starts_correctly": starts_correctly,
        "ends_solved": ends_solved,
        "resolved_state_sequence": [
            {"player": list(state[0]), "box": list(state[1])}
            for state in resolved_sequence
        ],
        "raw_resolved_state_count": len(raw_sequence),
        "discarded_resolved_states": max(len(raw_sequence) - len(resolved_sequence), 0),
        "resolved_actions": resolved_actions,
        "resolved_action_count": len(resolved_actions),
        "matches_gt_action_count": len(resolved_actions) == len(gt_actions),
        "matches_gt_push_count": recovered_pushes == gt_pushes,
        "illegal_transitions": illegal_transitions,
        "unresolved_frames": unresolved_frames,
        "unresolved_ratio": unresolved_ratio,
        "entity_validity_violations": entity_validity_violations,
        "bridge_limit": BRIDGE_LIMIT,
    }


def _best_legal_subsequence(
    sequence: list[State],
    initial_state: State,
    wall_rows: tuple[str, ...],
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
            if _find_transition_path(sequence[left], state, wall_rows, bridge_limit) is None:
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
    wall_rows: tuple[str, ...],
    max_depth: int,
) -> list[str] | None:
    if previous == current:
        return []
    direct = _infer_move(previous, current, wall_rows)
    if direct is not None:
        return [direct]
    if max_depth <= 1:
        return None

    queue: deque[tuple[State, list[str]]] = deque([(previous, [])])
    seen = {previous}
    while queue:
        state, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for next_state, action in _forward_moves(state, wall_rows):
            if next_state in seen:
                continue
            next_path = path + [action]
            if next_state == current:
                return next_path
            seen.add(next_state)
            queue.append((next_state, next_path))
    return None


def _infer_move(previous: State, current: State, wall_rows: tuple[str, ...]) -> str | None:
    for next_state, action in _forward_moves(previous, wall_rows):
        if next_state == current:
            return action
    return None


def _forward_moves(state: State, wall_rows: tuple[str, ...]) -> list[tuple[State, str]]:
    player, box = state
    moves: list[tuple[State, str]] = []
    for action, (dr, dc) in (("U", (-1, 0)), ("D", (1, 0)), ("L", (0, -1)), ("R", (0, 1))):
        next_player = (player[0] + dr, player[1] + dc)
        if not _is_floor(wall_rows, next_player):
            continue
        if next_player == box:
            next_box = (box[0] + dr, box[1] + dc)
            if not _is_floor(wall_rows, next_box):
                continue
            moves.append(((next_player, next_box), action))
        else:
            moves.append(((next_player, box), action))
    return moves


def _is_floor(wall_rows: tuple[str, ...], cell: Cell) -> bool:
    row, col = cell
    return 0 <= row < len(wall_rows) and 0 <= col < len(wall_rows[0]) and wall_rows[row][col] == "."
