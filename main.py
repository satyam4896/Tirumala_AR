from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from langdetect import detect_langs
from deep_translator import GoogleTranslator

from nav_graph import navigation_graph, POIS
from crowd_manager import crowd_manager
from multi_stop_router import MultiStopRouter

router = MultiStopRouter()


# ── Startup ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    crowd_manager.simulate_time_of_day()
    print("[Startup] Initial crowd simulation applied.")
    yield


app = FastAPI(
    title="Tirumala AR Navigation API",
    version="2.1",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request Models ────────────────────────────────────────────────────────────

class NavRequest(BaseModel):
    user_query: str   = Field(..., example="Take me to the Laddoo Counter")
    source_lat: float = Field(..., example=13.6827)
    source_lng: float = Field(..., example=79.3468)


class MultiStopRequest(BaseModel):
    stops:          list[str]
    source_lat:     float
    source_lng:     float
    optimize_order: bool = True


class CrowdUpdateRequest(BaseModel):
    center_lat: float
    center_lng: float
    radius_m:   float = 60.0
    weight:     float = Field(..., ge=0.0, le=1.0)
    source:     str   = "api"


class CrowdSimRequest(BaseModel):
    hour: Optional[int] = Field(None, ge=0, le=23)


# ── Navigation ────────────────────────────────────────────────────────────────

@app.post("/navigate")
def navigate(req: NavRequest):
    try:
        original_query = req.user_query

        # FIXED: Only translate if very confident it's not English
        try:
            langs      = detect_langs(original_query)
            top_lang   = langs[0]
            lang       = top_lang.lang
            confidence = top_lang.prob

            # Only trust detection if confidence > 95% AND not English
            if confidence < 0.95 or lang == "en":
                lang = "en"

        except:
            lang = "en"

        # Translate input to English if needed
        if lang != "en":
            try:
                translated_query = GoogleTranslator(
                    source="auto",
                    target="en"
                ).translate(original_query)
            except:
                translated_query = original_query
        else:
            translated_query = original_query

        state = navigation_graph.invoke({
            "user_query": translated_query,
            "source_lat": req.source_lat,
            "source_lng": req.source_lng,
        })

        response = state["response"]

        # Only translate back if clearly not English
        if lang != "en" and response.get("instructions"):
            translated_instructions = []
            for step in response["instructions"]:
                try:
                    translated_step = GoogleTranslator(
                        source="en",
                        target=lang
                    ).translate(step)
                except:
                    translated_step = step
                translated_instructions.append(translated_step)
            response["instructions"] = translated_instructions

        response["detected_language"] = lang
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Multi Stop ────────────────────────────────────────────────────────────────

@app.post("/navigate/multi")
def navigate_multi(req: MultiStopRequest):
    try:
        live_weights = crowd_manager.get_all_weights()
        route = router.plan(
            req.source_lat,
            req.source_lng,
            req.stops,
            req.optimize_order,
            crowd_weights=live_weights,
        )
        if not route:
            raise HTTPException(404, "Could not resolve stops")
        return router.to_response(route)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Crowd ─────────────────────────────────────────────────────────────────────

@app.post("/crowd/update")
def crowd_update(req: CrowdUpdateRequest):
    edges = crowd_manager.set_zone_crowd(
        req.center_lat, req.center_lng,
        req.radius_m, req.weight, req.source
    )
    return {"status": "updated", "edges_affected": edges, "weight": req.weight}


@app.post("/crowd/simulate")
def crowd_simulate(req: CrowdSimRequest):
    result = crowd_manager.simulate_time_of_day(req.hour)
    return {"status": "simulated", **result}


@app.get("/crowd/hotspots")
def crowd_hotspots(threshold: float = 0.5):
    return {"hotspots": crowd_manager.get_hotspots(threshold)}


@app.get("/crowd/stats")
def crowd_stats():
    return crowd_manager.get_stats()


# ── Destinations ──────────────────────────────────────────────────────────────

@app.get("/destinations")
def destinations(type_filter: Optional[str] = None):
    results = [
        {
            "name": p["name"],
            "type": p.get("type", ""),
            "lat":  p["lat"],
            "lon":  p["lon"],
        }
        for p in POIS
        if p.get("name")
        and (not type_filter or p.get("type") == type_filter)
    ]
    return {"count": len(results), "destinations": results}


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":  "ok",
        "version": "2.1",
        "service": "Tirumala AR Navigation Backend",
    }