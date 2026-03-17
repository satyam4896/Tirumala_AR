"""
nav_state.py
────────────────────────────────────────────────────────────────────────────
Shared State for the Tirumala LangGraph Navigation System.

LangGraph passes this TypedDict between every node.  Each node reads what
it needs and writes its outputs back — nothing else is shared.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
from typing import TypedDict, Optional


class NavState(TypedDict, total=False):
    """
    The single state object that flows through the entire LangGraph pipeline.

    Fields are grouped by the agent that primarily writes them.
    """

    # ── INPUT (set by the caller before graph.invoke()) ──────────────────────
    user_query:     str           # e.g. "Take me to the Laddoo Counter"
    source_lat:     float         # pilgrim's current latitude
    source_lng:     float         # pilgrim's current longitude

    # ── QueryUnderstandingNode ────────────────────────────────────────────────
    extracted_place: str          # LLM-extracted destination name
    query_confidence: float       # 0.0–1.0, how confident the LLM was

    # ── SemanticMatchNode ─────────────────────────────────────────────────────
    matched_place:   str          # best-matched POI name from the graph
    matched_lat:     float        # matched POI latitude
    matched_lng:     float        # matched POI longitude
    semantic_score:  float        # cosine similarity score

    # ── PathfindingNode ───────────────────────────────────────────────────────
    raw_path:        list[int]    # list of graph node IDs
    path_distance_m: float        # total path distance in metres
    path_found:      bool         # True if a path was computed

    # ── SpatialReasoningNode ──────────────────────────────────────────────────
    ar_path:         list[dict]   # [{lat, lng, heading}, ...] AR-ready coords
    raw_steps:       list[dict]   # geometric steps with landmark info
    raw_instructions: list[str]   # plain-text geometric instructions

    # ── LLMEnhancementNode ────────────────────────────────────────────────────
    instructions:    list[str]    # final human-friendly instructions
    llm_enhanced:    bool         # True if LLM enhancement succeeded

    # ── SafetyCheckNode ───────────────────────────────────────────────────────
    safety_ok:       bool         # True if path passes safety checks
    safety_warnings: list[str]    # any warnings (crowd, steps, etc.)

    # ── ErrorNode ─────────────────────────────────────────────────────────────
    error:           Optional[str]  # error message if anything failed
    failed_node:     Optional[str]  # which node raised the error

    # ── FINAL RESPONSE (assembled by ResponseNode) ───────────────────────────
    response:        dict          # the full API response dict
