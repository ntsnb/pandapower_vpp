from __future__ import annotations

from collections import defaultdict, deque

import pandas as pd
import pandapower as pp


def _adjacency_from_net(net: pp.pandapowerNet) -> dict[int, list[int]]:
    adjacency: dict[int, list[int]] = defaultdict(list)
    if len(net.line):
        for _, row in net.line.iterrows():
            from_bus = int(row["from_bus"])
            to_bus = int(row["to_bus"])
            adjacency[from_bus].append(to_bus)
            adjacency[to_bus].append(from_bus)
    if len(net.trafo):
        for _, row in net.trafo.iterrows():
            hv_bus = int(row["hv_bus"])
            lv_bus = int(row["lv_bus"])
            adjacency[hv_bus].append(lv_bus)
            adjacency[lv_bus].append(hv_bus)
    for bus in net.bus.index:
        adjacency.setdefault(int(bus), [])
    return {bus: sorted(neighbors) for bus, neighbors in adjacency.items()}


def _root_buses(net: pp.pandapowerNet) -> list[int]:
    if len(net.ext_grid):
        return sorted({int(bus) for bus in net.ext_grid["bus"].tolist()})
    if len(net.bus):
        return [int(net.bus.index[0])]
    return []


def _component_order(adjacency: dict[int, list[int]], roots: list[int]) -> list[int]:
    ordered_roots = list(roots)
    seen = set(ordered_roots)
    for bus in sorted(adjacency):
        if bus not in seen:
            ordered_roots.append(bus)
            seen.add(bus)
    return ordered_roots


def _tree_from_root(
    adjacency: dict[int, list[int]],
    root: int,
    visited: set[int],
) -> tuple[dict[int, list[int]], dict[int, int]]:
    children: dict[int, list[int]] = defaultdict(list)
    depths = {root: 0}
    queue: deque[int] = deque([root])
    visited.add(root)
    while queue:
        bus = queue.popleft()
        for neighbor in adjacency[bus]:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            children[bus].append(neighbor)
            children.setdefault(neighbor, [])
            depths[neighbor] = depths[bus] + 1
            queue.append(neighbor)
    return {bus: sorted(items) for bus, items in children.items()}, depths


def _assign_tree_lanes(
    root: int,
    children: dict[int, list[int]],
    cursor: list[float],
) -> dict[int, float]:
    """Assign y lanes from leaves upward so radial branches are visually separated."""

    lanes: dict[int, float] = {}

    def visit(bus: int) -> float:
        child_lanes = [visit(child) for child in children.get(bus, [])]
        if not child_lanes:
            lanes[bus] = cursor[0]
            cursor[0] += 1.0
        else:
            lanes[bus] = sum(child_lanes) / len(child_lanes)
        return lanes[bus]

    visit(root)
    return lanes


def deterministic_feeder_layout(net: pp.pandapowerNet) -> pd.DataFrame:
    """Return deterministic x/y coordinates for a radial-style pandapower network.

    The project does not have GIS coordinates yet. This helper derives a stable
    feeder layout from line/transformer connectivity, keeping the slack bus at
    the left and downstream buses toward the right. The y coordinate is a
    radial branch lane assigned from the tree leaves upward; it is a schematic
    one-line layout, not a geographic map.
    """

    if {"schematic_x", "schematic_y"}.issubset(set(net.bus.columns)):
        hinted = net.bus[["schematic_x", "schematic_y"]].copy()
        if hinted.notna().all().all():
            return (
                hinted.rename(columns={"schematic_x": "x", "schematic_y": "y"})
                .reset_index(names="bus_id")
                .astype({"bus_id": int, "x": float, "y": float})
                .sort_values("bus_id")
                .reset_index(drop=True)
            )

    adjacency = _adjacency_from_net(net)
    roots = _component_order(adjacency, _root_buses(net))
    visited: set[int] = set()
    rows: list[dict[str, float | int]] = []
    component_y_offset = 0.0

    for root in roots:
        if root in visited:
            continue
        children, depths = _tree_from_root(adjacency, root, visited)
        lanes = _assign_tree_lanes(root, children, [0.0])
        max_lane = max(lanes.values(), default=0.0)

        local_rows = [
            {
                "bus_id": int(bus),
                "x": float(depths[bus]),
                "y": float(component_y_offset + (max_lane - lanes.get(bus, 0.0))),
            }
            for bus in sorted(depths)
        ]
        rows.extend(local_rows)
        component_y_offset += max(3.0, float(max_lane + 3.0))

    layout = pd.DataFrame(rows)
    if layout.empty:
        return pd.DataFrame(columns=["bus_id", "x", "y"])
    return layout.sort_values("bus_id").reset_index(drop=True)
