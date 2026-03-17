"""
multi_stop_router.py
────────────────────────────────────────────────────────────────────────────
Multi-Stop Pilgrimage Router

Handles queries like:
  "Take me to Laddoo Counter, then Navagraha Shrine, then the exit"

Features:
  - Nearest-neighbour TSP heuristic for waypoint ordering
  - 2-opt improvement pass for shorter total routes
  - Per-leg instruction generation
  - Full LangGraph integration as a drop-in node
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from graph_utils import (
    NODES, POIS, LCC,
    haversine, bearing, bearing_to_cardinal,
    nearest_node, nearest_landmark, astar,
)
from sentence_transformers import SentenceTransformer, util as st_util

# ── Semantic matcher (re-use same model as main graph) ───────────────────────
_embed_model  = SentenceTransformer("all-MiniLM-L6-v2")
_poi_names    = [p["name"] for p in POIS]
_poi_embeds   = _embed_model.encode(_poi_names, convert_to_tensor=True)


def match_place(query: str) -> dict | None:
    """Return best-matching POI for a query string."""
    qe     = _embed_model.encode(query, convert_to_tensor=True)
    scores = st_util.cos_sim(qe, _poi_embeds)[0]
    idx    = int(scores.argmax())
    score  = float(scores[idx])
    if score < 0.25:
        return None
    return {**POIS[idx], "score": score}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Waypoint:
    name:      str
    lat:       float
    lon:       float
    node_id:   int
    poi_type:  str = ""


@dataclass
class Leg:
    """One segment of a multi-stop journey."""
    from_wp:      Waypoint
    to_wp:        Waypoint
    path_nodes:   list[int]
    distance_m:   float
    instructions: list[str] = field(default_factory=list)


@dataclass
class MultiStopRoute:
    waypoints:       list[Waypoint]
    legs:            list[Leg]
    total_distance_m: float
    ordered_stops:   list[str]


# ── Turn / instruction helpers ────────────────────────────────────────────────

def _turn_type(prev_b: float, new_b: float) -> str:
    diff = ((new_b - prev_b) + 360) % 360
    if diff > 180: diff -= 360
    if abs(diff) < 20:  return "straight"
    if diff < 60:        return "slight_right"
    if diff < 120:       return "right"
    if diff >= 120:      return "sharp_right"
    if diff > -60:       return "slight_left"
    if diff > -120:      return "left"
    return "sharp_left"


_PHRASES = {
    "straight":    "Continue {dir}",
    "slight_right":"Bear right",
    "right":       "Turn right",
    "sharp_right": "Turn sharp right",
    "slight_left": "Bear left",
    "left":        "Turn left",
    "sharp_left":  "Turn sharp left",
}


def _path_to_instructions(path_nodes: list[int], dest_name: str) -> list[str]:
    """Convert a node-ID path to text instructions with landmark references."""
    coords = [
        {"lat": NODES[nid]["lat"], "lng": NODES[nid]["lon"]}
        for nid in path_nodes if nid in NODES
    ]
    if len(coords) < 2:
        return [f"Proceed to {dest_name}."]

    # Simplify
    simplified = [coords[0]]
    for pt in coords[1:]:
        if haversine(simplified[-1]["lat"], simplified[-1]["lng"],
                     pt["lat"], pt["lng"]) >= 15:
            simplified.append(pt)

    steps      = []
    seg_start  = 0
    seg_b      = bearing(simplified[0]["lat"], simplified[0]["lng"],
                         simplified[1]["lat"], simplified[1]["lng"])

    for i in range(1, len(simplified)):
        curr = simplified[i]
        if i < len(simplified) - 1:
            nxt    = simplified[i + 1]
            next_b = bearing(curr["lat"], curr["lng"], nxt["lat"], nxt["lng"])
            turn   = _turn_type(seg_b, next_b)
        else:
            next_b = seg_b
            turn   = "arrive"

        if turn != "straight":
            seg_dist = sum(
                haversine(simplified[j]["lat"], simplified[j]["lng"],
                          simplified[j+1]["lat"], simplified[j+1]["lng"])
                for j in range(seg_start, i)
            )
            lm       = nearest_landmark(curr["lat"], curr["lng"], exclude=dest_name)
            dist_str = f"{round(seg_dist)}m" if seg_dist < 1000 else f"{seg_dist/1000:.1f}km"

            if turn == "arrive":
                text = f"You have arrived at {dest_name}."
            else:
                action = _PHRASES[turn].format(dir=bearing_to_cardinal(seg_b))
                text   = f"{action} and walk {dist_str}"
                if lm:
                    ref  = "just after" if turn != "straight" else "passing near"
                    text += f", {ref} {lm['name']}"
                text += "."

            steps.append(text)
            seg_start = i
            seg_b     = next_b

    return steps if steps else [f"Head to {dest_name}."]


# ── TSP ordering ──────────────────────────────────────────────────────────────

def _nn_order(source: Waypoint, waypoints: list[Waypoint]) -> list[Waypoint]:
    """Nearest-neighbour TSP heuristic: greedily pick closest unvisited stop."""
    remaining = list(waypoints)
    ordered   = []
    current   = source

    while remaining:
        closest = min(
            remaining,
            key=lambda wp: haversine(current.lat, current.lon, wp.lat, wp.lon)
        )
        ordered.append(closest)
        remaining.remove(closest)
        current = closest

    return ordered


def _two_opt(waypoints: list[Waypoint]) -> list[Waypoint]:
    """2-opt improvement on waypoint ordering."""
    best = list(waypoints)
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best)):
                # Try reversing segment [i+1..j]
                new_route = best[:i+1] + list(reversed(best[i+1:j+1])) + best[j+1:]
                old_d = sum(
                    haversine(best[k].lat, best[k].lon,
                              best[k+1].lat, best[k+1].lon)
                    for k in range(len(best)-1)
                )
                new_d = sum(
                    haversine(new_route[k].lat, new_route[k].lon,
                              new_route[k+1].lat, new_route[k+1].lon)
                    for k in range(len(new_route)-1)
                )
                if new_d < old_d - 0.1:
                    best = new_route
                    improved = True
    return best


# ── Main router ───────────────────────────────────────────────────────────────

class MultiStopRouter:
    """
    Plan a multi-stop pilgrimage route.

    Usage:
        router = MultiStopRouter()
        result = router.plan(
            source_lat=13.6800,
            source_lng=79.3490,
            stop_names=["Laddoo Counter", "Co-operative Bank", "Vaikunta Nilayam"],
            optimize_order=True,
        )
    """

    def plan(
        self,
        source_lat:     float,
        source_lng:     float,
        stop_names:     list[str],
        optimize_order: bool = True,
        crowd_weights:  dict | None = None,
    ) -> MultiStopRoute | None:

        # 1. Resolve each stop name to a Waypoint
        waypoints: list[Waypoint] = []
        for name in stop_names:
            poi = match_place(name)
            if not poi:
                print(f"[MultiStop] Could not match '{name}' — skipping")
                continue
            node_id = nearest_node(poi["lat"], poi["lon"])
            waypoints.append(Waypoint(
                name     = poi["name"],
                lat      = poi["lat"],
                lon      = poi["lon"],
                node_id  = node_id,
                poi_type = poi.get("type", ""),
            ))

        if not waypoints:
            return None

        source_wp = Waypoint(
            name    = "Your Location",
            lat     = source_lat,
            lon     = source_lng,
            node_id = nearest_node(source_lat, source_lng),
        )

        # 2. Optimise order
        if optimize_order and len(waypoints) > 1:
            ordered = _nn_order(source_wp, waypoints)
            if len(ordered) > 2:
                ordered = _two_opt(ordered)
        else:
            ordered = waypoints

        # 3. Compute leg-by-leg paths
        all_wps  = [source_wp] + ordered
        legs: list[Leg] = []
        total_dist = 0.0

        for i in range(len(all_wps) - 1):
            src = all_wps[i]
            dst = all_wps[i + 1]
            path_nodes, dist = astar(src.node_id, dst.node_id,
                                     crowd_weights=crowd_weights)
            if not path_nodes:
                print(f"[MultiStop] No path from {src.name} → {dst.name}")
                continue

            instructions = _path_to_instructions(path_nodes, dst.name)
            legs.append(Leg(
                from_wp      = src,
                to_wp        = dst,
                path_nodes   = path_nodes,
                distance_m   = dist,
                instructions = instructions,
            ))
            total_dist += dist

        return MultiStopRoute(
            waypoints        = ordered,
            legs             = legs,
            total_distance_m = total_dist,
            ordered_stops    = [wp.name for wp in ordered],
        )

    def to_response(self, route: MultiStopRoute) -> dict:
        """Serialise a MultiStopRoute to a JSON-ready dict."""
        legs_out = []
        for leg in route.legs:
            path_coords = [
                {"lat": NODES[nid]["lat"], "lng": NODES[nid]["lon"]}
                for nid in leg.path_nodes if nid in NODES
            ]
            legs_out.append({
                "from":         leg.from_wp.name,
                "to":           leg.to_wp.name,
                "distance_m":   round(leg.distance_m),
                "instructions": leg.instructions,
                "path":         path_coords,
            })

        return {
            "status":            "success",
            "stop_count":        len(route.waypoints),
            "total_distance_m":  round(route.total_distance_m),
            "ordered_stops":     route.ordered_stops,
            "legs":              legs_out,
        }
