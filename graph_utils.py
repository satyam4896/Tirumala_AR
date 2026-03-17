"""
graph_utils.py
────────────────────────────────────────────────────────────────────────────
Shared graph primitives loaded once at import time.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import json, math, heapq, os

_NAV_GRAPH_PATH = os.getenv("NAV_GRAPH_PATH", "tirumala_navigation_graph_final.json")

print("[GraphUtils] Loading navigation JSON...")
with open(_NAV_GRAPH_PATH, "r", encoding="utf-8") as _f:
    _RAW = json.load(_f)

NODES:    dict[int, dict]            = {n["id"]: n for n in _RAW["nodes"]}
ADJ:      dict[str, list[int]]       = _RAW["adjacency"]
EDGE_MAP: dict[tuple[int,int], dict] = {(e["from"], e["to"]): e for e in _RAW["edges"]}
POIS:     list[dict]                 = [p for p in _RAW["pois"] if p.get("name", "").strip()]


def _build_lcc() -> set[int]:
    adj_ids  = set(int(k) for k in ADJ.keys())
    visited: set[int] = set()
    components: list[set[int]] = []
    for start in adj_ids:
        if start in visited:
            continue
        comp: set[int] = set()
        queue = [start]
        while queue:
            cur = queue.pop()
            if cur in comp:
                continue
            comp.add(cur)
            for nb in ADJ.get(str(cur), []):
                if nb not in comp:
                    queue.append(nb)
        components.append(comp)
        visited |= comp
    return max(components, key=len)


LCC: set[int] = _build_lcc()
print(f"[GraphUtils] {len(NODES)} nodes | {len(EDGE_MAP)} edges | LCC={len(LCC)} routable nodes | {len(POIS)} POIs")


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R  = 6_371_000
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x  = math.sin(dl) * math.cos(p2)
    y  = math.cos(p1)*math.sin(p2) - math.sin(p1)*math.cos(p2)*math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def bearing_to_cardinal(b: float) -> str:
    dirs = ["north", "northeast", "east", "southeast",
            "south", "southwest", "west", "northwest"]
    return dirs[round(b / 45) % 8]


def nearest_node(lat: float, lon: float) -> int:
    return min(
        LCC,
        key=lambda nid: haversine(lat, lon, NODES[nid]["lat"], NODES[nid]["lon"])
        if nid in NODES else 1e9
    )


def nearest_landmark(lat: float, lon: float,
                     radius_m: float = 80,
                     exclude: str = None) -> dict | None:
    best, best_d = None, float("inf")
    for p in POIS:
        if p["name"] == exclude:
            continue
        d = haversine(lat, lon, p["lat"], p["lon"])
        if d < best_d and d <= radius_m:
            best, best_d = p, d
    if best:
        return {"name": best["name"], "dist_m": round(best_d),
                "type": best.get("type", "")}
    return None


def astar(source_id: int, target_id: int,
          crowd_weights: dict[tuple[int,int], float] | None = None) -> tuple[list[int], float]:
    """
    A* on the LCC graph.
    Returns (path_node_ids, total_distance_m) or ([], 0) if unreachable.
    """
    # FIXED: Early exit when source == destination
    if source_id == target_id:
        return [source_id], 0.0

    tn = NODES.get(target_id)
    if not tn:
        return [], 0

    def h(nid: int) -> float:
        n = NODES.get(nid)
        return haversine(n["lat"], n["lon"], tn["lat"], tn["lon"]) if n else 0

    def edge_cost(u: int, v: int) -> float:
        e = EDGE_MAP.get((u, v), {})
        if e.get("ritual_block", False):
            return float("inf")
        base  = e.get("distance", 1)
        crowd = crowd_weights.get((u, v), 0) if crowd_weights else e.get("crowd_weight", 0)
        return base + 5 * crowd

    open_heap: list           = [(h(source_id), 0.0, source_id)]
    came_from: dict[int, int] = {}
    g_score:   dict[int, float] = {source_id: 0.0}

    while open_heap:
        _, gc, cur = heapq.heappop(open_heap)
        if cur == target_id:
            path: list[int] = []
            c = cur
            while c in came_from:
                path.append(c)
                c = came_from[c]
            path.append(source_id)
            return list(reversed(path)), gc
        for nb in ADJ.get(str(cur), []):
            cost = edge_cost(cur, nb)
            tg   = gc + cost
            if tg < g_score.get(nb, float("inf")):
                came_from[nb] = cur
                g_score[nb]   = tg
                heapq.heappush(open_heap, (tg + h(nb), tg, nb))

    return [], 0