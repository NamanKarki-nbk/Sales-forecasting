from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
from src import (
    SalesQueryEngine1,
    RAGQueryEngine,
    QueryRouter,
    HybridQueryEngine,
    SalesForecastingSystem,
    RAGBuilder
)

app = FastAPI(title="Sales Forecasting & Query System")

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engines
hybrid_engine = None
sales_engine = None
rag_engine = None

# Request/Response Models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    classification: str
    success: bool
    params: Optional[Dict[str, Any]] = None

class ForecastRequest(BaseModel):
    store: str
    department: str
    periods: int = 52

# Startup - Initialize all engines
@app.on_event("startup")
async def startup_event():
    global hybrid_engine, sales_engine, rag_engine
    
    print("🚀 Starting up FastAPI server...")
    
    # Paths
    sales_data_path = "data/sales/synthetic_nepal_sales_with_more_holidays.csv"
    holiday_path = "data/knowledge_base/downloadable_kb_csv.csv"
    forecast_csv_path = "data/forecasts/all_consolidated_forecasts.csv"
    vector_store_path = "vector_store"
    
    # Check if files exist
    if not os.path.exists(sales_data_path):
        raise FileNotFoundError(f"Sales data not found: {sales_data_path}")
    
    if not os.path.exists(forecast_csv_path):
        raise FileNotFoundError(f"Forecast data not found: {forecast_csv_path}")
    
    # Initialize Sales Query Engine
    print("📊 Initializing Sales Query Engine...")
    sales_engine = SalesQueryEngine1(
        historical_csv=sales_data_path,
        forecast_csv=forecast_csv_path
    )
    
    # Initialize RAG Engine
    print("📚 Initializing RAG Query Engine...")
    rag_engine = RAGQueryEngine(
        vectorstore_path=vector_store_path,
        llm_model="mistral",
        top_k=3
    )
    
    # Initialize Router
    print("🔀 Initializing Query Router...")
    router = QueryRouter(llm_model='mistral')
    
    # Initialize Hybrid Engine
    print("🎯 Initializing Hybrid Query Engine...")
    hybrid_engine = HybridQueryEngine(
        sales_engine=sales_engine,
        rag_engine=rag_engine,
        router=router,
        llm_model="mistral"
    )
    
    print("✅ All engines initialized successfully!")

# Health check
@app.get("/")
async def root():
    return {
        "status": "running",
        "message": "Sales Forecasting & Query API",
        "endpoints": ["/api/query", "/api/stores", "/api/departments"]
    }

# Main query endpoint
@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    if not hybrid_engine:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        result = hybrid_engine.query(request.query, verbose=False)
        
        return QueryResponse(
            answer=result.get("answer", "No answer generated"),
            classification=result.get("classification", "UNKNOWN"),
            success=result.get("success", False),
            params=result.get("params")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get all stores
@app.get("/api/stores")
async def get_stores():
    if not sales_engine:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return {"stores": sales_engine.stores}

# Get all departments
@app.get("/api/departments")
async def get_departments():
    if not sales_engine:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return {"departments": sales_engine.departments}

# Search stores
@app.get("/api/stores/search")
async def search_stores(q: str):
    if not sales_engine:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    results = sales_engine.search_stores(q)
    return {"results": results}

# Search departments
@app.get("/api/departments/search")
async def search_departments(q: str):
    if not sales_engine:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    results = sales_engine.search_department(q)
    return {"results": results}

# Get sales summary
@app.get("/api/sales/summary")
async def get_summary(
    store: str,
    department: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    if not sales_engine:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        summary = sales_engine.get_sales_summary(store, department, start_date, end_date)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get sales trend
@app.get("/api/sales/trend")
async def get_trend(
    store: str,
    department: str,
    period: str = "monthly",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    if not sales_engine:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        trend = sales_engine.get_sales_trend(store, department, start_date, end_date, period)
        return {"trend": trend}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get holiday impact
@app.get("/api/sales/holiday-impact")
async def get_holiday_impact(store: str, department: str, year: int):
    if not sales_engine:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        impact = sales_engine.get_holiday_impact(store, department, year)
        return impact
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)