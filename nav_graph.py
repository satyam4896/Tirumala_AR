"""
nav_graph.py  (v2 — LCC fix + crowd weights + multi-stop)
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import os, json
from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer, util as st_util

from nav_state import NavState
from graph_utils import (
    NODES, POIS, haversine, bearing, bearing_to_cardinal,
    nearest_node, nearest_landmark, astar, EDGE_MAP,
)
from crowd_manager import crowd_manager

load_dotenv()

_groq        = Groq(api_key=os.getenv("GROQ_API_KEY"))
_embed_model = SentenceTransformer("all-MiniLM-L6-v2")
_poi_names   = [p["name"] for p in POIS]
_poi_embeds  = _embed_model.encode(_poi_names, convert_to_tensor=True)
print("[NavGraph v2] Models ready.")


# ── Node 1: QueryUnderstanding ────────────────────────────────────────────────
def query_understanding_node(state: NavState) -> NavState:
    print("[Node 1] QueryUnderstanding")
    query = state["user_query"]
    try:
        resp = _groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content":
                 "Extract destination place name from a Tirumala navigation query. Return ONLY the name."},
                {"role": "user", "content": f"Query: {query}"},
            ], max_tokens=30,
        )
        place      = resp.choices[0].message.content.strip().strip('"\'')
        confidence = 0.9 if len(place.split()) <= 4 else 0.6
    except Exception as e:
        print(f"  LLM fallback: {e}")
        place, confidence = query, 0.3
    print(f"  → '{place}' conf={confidence:.1f}")
    return {**state, "extracted_place": place, "query_confidence": confidence}


# ── Node 2: SemanticMatch ─────────────────────────────────────────────────────
def semantic_match_node(state: NavState) -> NavState:
    print("[Node 2] SemanticMatch")
    if state.get("error"): return state
    qe     = _embed_model.encode(state["extracted_place"], convert_to_tensor=True)
    scores = st_util.cos_sim(qe, _poi_embeds)[0]
    idx    = int(scores.argmax())
    score  = float(scores[idx])
    poi    = POIS[idx]
    print(f"  → '{poi['name']}' score={score:.3f}")
    if score < 0.25:
        return {**state, "error": f"Cannot match '{state['extracted_place']}'.",
                "failed_node": "SemanticMatch"}
    return {**state,
            "matched_place": poi["name"],
            "matched_lat":   poi["lat"],
            "matched_lng":   poi["lon"],
            "semantic_score": score}


# ── Node 3: Pathfinding ───────────────────────────────────────────────────────
def pathfinding_node(state: NavState) -> NavState:
    print("[Node 3] Pathfinding")
    if state.get("error"): return state
    src  = nearest_node(state["source_lat"],  state["source_lng"])
    dst  = nearest_node(state["matched_lat"], state["matched_lng"])
    print(f"  src={src} dst={dst}")
    live = crowd_manager.get_all_weights()
    path, dist = astar(src, dst, crowd_weights=live)
    if not path:
        return {**state, "path_found": False, "error": "No path found.",
                "failed_node": "Pathfinding"}
    print(f"  → {len(path)} nodes {dist:.0f}m")
    return {**state, "raw_path": path, "path_distance_m": dist, "path_found": True}


# ── Node 4: SafetyCheck ───────────────────────────────────────────────────────
def safety_check_node(state: NavState) -> NavState:
    print("[Node 4] SafetyCheck")
    if state.get("error"): return state
    path     = state["raw_path"]
    warnings = []
    steps = crowds = 0
    for i in range(len(path) - 1):
        e = EDGE_MAP.get((path[i], path[i+1]), {})
        if e.get("type") == "steps":         steps  += 1
        if e.get("crowd_weight", 0) > 0.5:   crowds += 1
    if steps:  warnings.append(f"{steps} staircase segment(s) — difficult for elderly pilgrims.")
    if crowds: warnings.append(f"{crowds} high-crowd segment(s) on this route.")
    if state.get("path_distance_m", 0) > 2000:
        warnings.append(f"Long route ({state['path_distance_m']:.0f}m) — consider a free bus.")
    print(f"  → {len(warnings)} warnings")
    return {**state, "safety_ok": True, "safety_warnings": warnings}


# ── Node 5: SpatialReasoning ──────────────────────────────────────────────────
_TURN_PHRASES = {
    "straight":    "Continue {dir}",
    "slight_right":"Bear right",
    "right":       "Turn right",
    "sharp_right": "Turn sharp right",
    "slight_left": "Bear left",
    "left":        "Turn left",
    "sharp_left":  "Turn sharp left",
}


def _turn_type(pb, nb):
    d = ((nb - pb) + 360) % 360
    if d > 180: d -= 360
    if abs(d) < 20:  return "straight"
    if d < 60:       return "slight_right"
    if d < 120:      return "right"
    if d >= 120:     return "sharp_right"
    if d > -60:      return "slight_left"
    if d > -120:     return "left"
    return "sharp_left"


def spatial_reasoning_node(state: NavState) -> NavState:
    print("[Node 5] SpatialReasoning")
    if state.get("error"): return state
    path  = state["raw_path"]
    dest  = state["matched_place"]
    coords = [{"lat": NODES[n]["lat"], "lng": NODES[n]["lon"]}
               for n in path if n in NODES]
    if len(coords) < 2:
        return {**state, "error": "Path too short.",
                "failed_node": "SpatialReasoning"}

    simp = [coords[0]]
    for pt in coords[1:]:
        if haversine(simp[-1]["lat"], simp[-1]["lng"],
                     pt["lat"], pt["lng"]) >= 15:
            simp.append(pt)

    ar = []
    for i in range(len(simp) - 1):
        h = bearing(simp[i]["lat"], simp[i]["lng"],
                    simp[i+1]["lat"], simp[i+1]["lng"])
        ar.append({**simp[i], "heading": round(h, 1)})
    ar.append({**simp[-1], "heading": None})

    steps     = []
    seg_start = 0
    seg_b     = bearing(simp[0]["lat"], simp[0]["lng"],
                        simp[1]["lat"], simp[1]["lng"])

    for i in range(1, len(simp)):
        curr = simp[i]
        if i < len(simp) - 1:
            nb   = bearing(curr["lat"], curr["lng"],
                           simp[i+1]["lat"], simp[i+1]["lng"])
            turn = _turn_type(seg_b, nb)
        else:
            nb   = seg_b
            turn = "arrive"

        if turn != "straight":
            sd = sum(
                haversine(simp[j]["lat"], simp[j]["lng"],
                          simp[j+1]["lat"], simp[j+1]["lng"])
                for j in range(seg_start, i)
            )
            lm = nearest_landmark(curr["lat"], curr["lng"], exclude=dest)
            steps.append({
                "type":          turn,
                "distance_m":    round(sd),
                "direction":     bearing_to_cardinal(seg_b),
                "bearing":       round(seg_b, 1),
                "lat":           curr["lat"],
                "lng":           curr["lng"],
                "landmark":      lm["name"]   if lm else None,
                "landmark_dist": lm["dist_m"] if lm else None,
            })
            seg_start = i
            seg_b     = nb

    raw = []
    for s in steps:
        if s["type"] == "arrive":
            raw.append(f"You have arrived at {dest}.")
        else:
            action = _TURN_PHRASES[s["type"]].format(dir=s["direction"])
            ds     = (f"{s['distance_m']}m" if s["distance_m"] < 1000
                      else f"{s['distance_m']/1000:.1f}km")
            text   = f"{action} and walk {ds}"
            if s["landmark"]:
                ref   = "passing near" if s["type"] == "straight" else "just after"
                text += f", {ref} {s['landmark']}"
            raw.append(text + ".")

    print(f"  → {len(ar)} AR pts | {len(steps)} steps")
    return {**state, "ar_path": ar, "raw_steps": steps, "raw_instructions": raw}


# ── Node 6: LLMEnhancement ───────────────────────────────────────────────────
_SYS = ("Warm local guide at Tirumala. Rewrite navigation steps as natural pilgrim-friendly "
        "instructions. Keep landmark names. Return ONLY a JSON array of strings.")


def llm_enhancement_node(state: NavState) -> NavState:
    print("[Node 6] LLMEnhancement")
    if state.get("error"): return state
    raw = state["raw_instructions"]

    if not raw:
        return {**state, "instructions": [], "llm_enhanced": False}

    try:
        resp = _groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": _SYS},
                {"role": "user", "content":
                 f'Steps to "{state["matched_place"]}":\n' +
                 "\n".join(f"{i+1}. {t}" for i, t in enumerate(raw)) +
                 "\n\nRewrite as JSON array."},
            ], max_tokens=600,
        )
        c = resp.choices[0].message.content.strip()
        if "```" in c:
            c = c.split("```")[1]
            if c.startswith("json"): c = c[4:]
            c = c.strip()
        enh = json.loads(c)

        if isinstance(enh, list):
            print(f"  → enhanced {len(enh)} steps ✓")
            return {**state, "instructions": enh, "llm_enhanced": True}
        else:
            print("  LLM returned non-list JSON, using raw")

    except Exception as e:
        print(f"  LLM failed ({e})")

    return {**state, "instructions": raw, "llm_enhanced": False}


# ── Node 7: Response ──────────────────────────────────────────────────────────
def response_node(state: NavState) -> NavState:
    print("[Node 7] Response")
    return {**state, "response": {
        "status":           "success",
        "destination":      state.get("matched_place"),
        "matched_lat":      state.get("matched_lat", 0.0),   # FIXED
        "matched_lng":      state.get("matched_lng", 0.0),   # FIXED
        "semantic_score":   round(state.get("semantic_score", 0), 3),
        "total_distance_m": round(state.get("path_distance_m", 0)),
        "node_count":       len(state.get("ar_path", [])),
        "llm_enhanced":     state.get("llm_enhanced", False),
        "safety_warnings":  state.get("safety_warnings", []),
        "instructions":     state.get("instructions", []),
        "path":             state.get("ar_path", []),
    }}


def error_node(state: NavState) -> NavState:
    print(f"[Error] {state.get('failed_node')}: {state.get('error')}")
    return {**state, "response": {
        "status":       "error",
        "failed_node":  state.get("failed_node"),
        "message":      state.get("error", "Unknown error"),
        "instructions": [],
        "path":         [],
    }}


# ── Routing ───────────────────────────────────────────────────────────────────
def _r_semantic(s):    return "error" if s.get("error") else "pathfinding"
def _r_pathfinding(s): return "error" if s.get("error") or not s.get("path_found") else "safety_check"
def _r_spatial(s):     return "error" if s.get("error") else "llm_enhancement"


# ── Compile ───────────────────────────────────────────────────────────────────
def build_navigation_graph():
    b = StateGraph(NavState)
    for name, fn in [
        ("query_understanding", query_understanding_node),
        ("semantic_match",      semantic_match_node),
        ("pathfinding",         pathfinding_node),
        ("safety_check",        safety_check_node),
        ("spatial_reasoning",   spatial_reasoning_node),
        ("llm_enhancement",     llm_enhancement_node),
        ("response",            response_node),
        ("error",               error_node),
    ]:
        b.add_node(name, fn)
    b.set_entry_point("query_understanding")
    b.add_edge("query_understanding", "semantic_match")
    b.add_edge("safety_check",        "spatial_reasoning")
    b.add_edge("llm_enhancement",     "response")
    b.add_edge("response", END)
    b.add_edge("error",    END)
    b.add_conditional_edges("semantic_match",    _r_semantic,
                            {"pathfinding": "pathfinding", "error": "error"})
    b.add_conditional_edges("pathfinding",       _r_pathfinding,
                            {"safety_check": "safety_check", "error": "error"})
    b.add_conditional_edges("spatial_reasoning", _r_spatial,
                            {"llm_enhancement": "llm_enhancement", "error": "error"})
    return b.compile()


navigation_graph = build_navigation_graph()
print("[NavGraph v2] Ready.")