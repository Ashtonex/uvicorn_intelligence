import os
from dotenv import load_dotenv
load_dotenv("../.env.local")

from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, Depends, HTTPException, Security, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from loguru import logger
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache

from models.inventory_forecast import run_sales_forecast, calculate_inventory_velocity
from models.finance_optimizer import optimize_capital_allocation
from analytics.nirvana_analytics import demand_forecast, expense_anomaly, inventory_velocity, capital_allocation, operations_overview

API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)):
    expected_key = os.getenv("INTELLIGENCE_API_KEY")
    if expected_key and api_key_header != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key_header

@asynccontextmanager
async def lifespan(app: FastAPI):
    FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")
    logger.info("Nirvana Intelligence Engine Cache Initialized")
    yield

app = FastAPI(title="Nirvana Intelligence API", lifespan=lifespan)

# Configure CORS for Next.js app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyticsRequest(BaseModel):
    kind: str = "all"

@app.get("/")
def root():
    return {
        "message": "Nirvana Intelligence Engine is running",
        "env_check": {
            "has_url": bool(os.getenv("NEXT_PUBLIC_SUPABASE_URL") or os.getenv("SUPABASE_URL")),
            "has_key": bool(os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_KEY")),
        }
    }

@app.post("/api/py/analytics/run", dependencies=[Depends(get_api_key)])
def run_analytics_job(request: AnalyticsRequest):
    """
    Triggers the high-power analytics snapshots and saves results to DB.
    Runs in ThreadPool so it doesn't block the event loop.
    """
    kind = request.kind
    logger.info(f"Running analytics job: {kind}")
    results = []
    jobs = {
        "demand_forecast": demand_forecast,
        "expense_anomaly": expense_anomaly,
        "inventory_velocity": inventory_velocity,
        "capital_allocation": capital_allocation,
        "operations_overview": operations_overview,
    }
    
    kinds = jobs.keys() if kind == "all" else [kind]
    
    for k in kinds:
        if k not in jobs:
            continue
        
        module = jobs[k]
        try:
            logger.debug(f"Executing module {k}")
            if k == "demand_forecast":
                payload = module.run(days=90, horizon=14, shop_id=None)
            elif k == "expense_anomaly":
                payload = module.run(days=60, limit=15)
            elif k == "inventory_velocity":
                payload = module.run(days=60, dead_stock_days=60, limit=25)
            elif k == "capital_allocation":
                payload = module.run(days=90, risk_aversion=2.5)
            elif k == "operations_overview":
                payload = module.run(days=180, horizon=3, limit=12)
            else:
                payload = module.run() if hasattr(module, "run") else {}
            
            summary = "Snapshot generated"
            if k == "demand_forecast":
                summary = f"{len(payload.get('forecasts', []))} shop forecasts generated"
            elif k == "expense_anomaly":
                summary = f"{len(payload.get('anomalies', []))} expense anomalies flagged"
            elif k == "inventory_velocity":
                summary = f"{len(payload.get('priority_items', []))} priority inventory items identified"
            elif k == "capital_allocation":
                summary = f"${payload.get('total_capital', 0):,.2f} capital optimized"
            elif k == "operations_overview":
                summary = f"{payload.get('summary', {}).get('accounts_tracked', 0)} accounts analyzed; {payload.get('summary', {}).get('anomalies_flagged', 0)} flags"
            
            from analytics.nirvana_analytics.data_loader import save_analytics_result
            save_analytics_result(k, payload, summary)
            
            logger.info(f"Completed {k}: {summary}")
            results.append({"kind": k, "ok": True, "summary": summary, "payload": payload})
        except Exception as e:
            logger.error(f"Error in {k}: {e}")
            results.append({"kind": k, "ok": False, "error": str(e)})
            
    return {"success": all(r["ok"] for r in results), "results": results}

@app.get("/api/py/forecast/sales")
@cache(expire=900)
def get_sales_forecast(
    days: int = Query(90, description="History window in days"),
    horizon: int = Query(14, description="Forecast horizon in days"),
    shopId: str = Query(None, description="Optional shop filter")
):
    """
    Returns predicted sales for the next N days.
    """
    logger.info("Computing sales forecast...")
    return run_sales_forecast(days=days, horizon=horizon, shop_id=shopId)

@app.get("/api/py/inventory/velocity")
@cache(expire=900)
def get_inventory_velocity():
    """
    Returns how fast products are selling per shop.
    """
    logger.info("Computing inventory velocity...")
    return calculate_inventory_velocity()

@app.get("/api/py/finance/optimize")
@cache(expire=900)
def get_finance_optimization():
    """
    Returns mathematical optimization for capital allocation.
    """
    logger.info("Computing finance optimization...")
    return optimize_capital_allocation()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
