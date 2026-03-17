"""
crowd_manager.py
────────────────────────────────────────────────────────────────────────────
Real-Time Crowd Weight Manager
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import time
import random
import threading
from datetime import datetime
from typing import Dict, Tuple

from graph_utils import EDGE_MAP, NODES, haversine

EdgeKey = Tuple[int, int]

DECAY_HALF_LIFE_SECONDS = 300
MAX_CROWD_WEIGHT        = 1.0
CROWD_COST_MULTIPLIER   = 5


class CrowdManager:
    def __init__(self):
        self._lock:    threading.RLock      = threading.RLock()
        self._weights: Dict[EdgeKey, float] = {}
        self._updated: Dict[EdgeKey, float] = {}
        self._history: list[dict]           = []

        self._decay_thread = threading.Thread(
            target=self._decay_loop, daemon=True, name="CrowdDecay"
        )
        self._decay_thread.start()

    def set_crowd(self, from_id: int, to_id: int, weight: float,
                  source: str = "manual") -> None:
        weight = max(0.0, min(MAX_CROWD_WEIGHT, weight))
        key = (from_id, to_id)
        with self._lock:
            self._weights[key] = weight
            self._updated[key] = time.time()
            self._history.append({
                "ts":     datetime.utcnow().isoformat(),
                "edge":   key,
                "weight": weight,
                "source": source,
            })

    def set_zone_crowd(self, center_lat: float, center_lon: float,
                       radius_m: float, weight: float,
                       source: str = "zone") -> int:
        count = 0
        for (u, v), edge in EDGE_MAP.items():
            nu, nv = NODES.get(u), NODES.get(v)
            if not nu or not nv:
                continue
            mid_lat = (nu["lat"] + nv["lat"]) / 2
            mid_lon = (nu["lon"] + nv["lon"]) / 2
            if haversine(center_lat, center_lon, mid_lat, mid_lon) <= radius_m:
                self.set_crowd(u, v, weight, source=source)
                count += 1
        return count

    def get_crowd(self, from_id: int, to_id: int) -> float:
        with self._lock:
            return self._weights.get((from_id, to_id), 0.0)

    def get_all_weights(self) -> Dict[EdgeKey, float]:
        with self._lock:
            return dict(self._weights)

    def clear_all(self) -> None:
        with self._lock:
            self._weights.clear()
            self._updated.clear()

    def get_hotspots(self, threshold: float = 0.5) -> list[dict]:
        hot = []
        with self._lock:
            for (u, v), w in self._weights.items():
                if w >= threshold:
                    nu, nv = NODES.get(u), NODES.get(v)
                    if nu and nv:
                        hot.append({
                            "from_node": u,
                            "to_node":   v,
                            "weight":    round(w, 3),
                            "from_lat":  nu["lat"],
                            "from_lon":  nu["lon"],
                            "to_lat":    nv["lat"],
                            "to_lon":    nv["lon"],
                        })
        return sorted(hot, key=lambda x: -x["weight"])

    def get_stats(self) -> dict:
        with self._lock:
            weights = list(self._weights.values())
        if not weights:
            return {"total_affected_edges": 0, "mean_weight": 0,
                    "max_weight": 0, "hotspot_count": 0}
        return {
            "total_affected_edges": len(weights),
            "mean_weight":   round(sum(weights) / len(weights), 3),
            "max_weight":    round(max(weights), 3),
            "hotspot_count": sum(1 for w in weights if w >= 0.5),
        }

    def simulate_time_of_day(self, hour: int | None = None) -> dict:
        if hour is None:
            hour = datetime.now().hour

        # FIXED: Realistic base weights (was all 0.0 before)
        BASE_ZONES = [
            # (lat, lon, radius_m, base_weight)
            (13.6835, 79.3473, 80, 0.8),  # Vaikuntam Queue Complex
            (13.6832, 79.3472, 60, 0.9),  # Main temple entrance
            (13.6838, 79.3474, 40, 0.7),  # Laddoo counter
            (13.6827, 79.3452, 50, 0.5),  # Alwar Tank surroundings
        ]

        if 5 <= hour < 7:
            multiplier = 0.4
            label = "early_morning"
        elif 7 <= hour < 10:
            multiplier = 1.0
            label = "peak_darshan"
        elif 10 <= hour < 12:
            multiplier = 0.6
            label = "mid_morning"
        elif 12 <= hour < 14:
            multiplier = 0.8
            label = "afternoon_rush"
        elif 14 <= hour < 16:
            multiplier = 0.3
            label = "midday_lull"
        elif 16 <= hour < 19:
            multiplier = 0.9
            label = "evening_peak"
        elif 19 <= hour < 21:
            multiplier = 0.5
            label = "evening_moderate"
        else:
            multiplier = 0.1
            label = "night"

        total_edges = 0
        self.clear_all()

        for lat, lon, radius, base_weight in BASE_ZONES:
            weight = min(1.0, base_weight * multiplier + random.uniform(-0.05, 0.05))
            affected = self.set_zone_crowd(lat, lon, radius, weight,
                                           source=f"sim_{label}")
            total_edges += affected

        return {
            "hour":           hour,
            "pattern":        label,
            "multiplier":     multiplier,
            "edges_affected": total_edges,
        }

    def simulate_event_spike(self, event_lat: float, event_lon: float,
                             radius_m: float = 100,
                             intensity: float = 0.9) -> int:
        return self.set_zone_crowd(
            event_lat, event_lon, radius_m, intensity, source="event_spike"
        )

    def _decay_loop(self) -> None:
        while True:
            time.sleep(30)
            now = time.time()
            with self._lock:
                to_delete = []
                for key, w in self._weights.items():
                    last    = self._updated.get(key, now)
                    elapsed = now - last
                    decayed = w * (0.5 ** (elapsed / DECAY_HALF_LIFE_SECONDS))
                    if decayed < 0.01:
                        to_delete.append(key)
                    else:
                        self._weights[key] = decayed
                for key in to_delete:
                    del self._weights[key]
                    self._updated.pop(key, None)


crowd_manager = CrowdManager()