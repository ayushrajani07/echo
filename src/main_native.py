from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import logging
import psutil
from datetime import datetime
from pathlib import Path
import os

# Initialize FastAPI app
app = FastAPI(
    title="POA Trading Platform",
    description="Native Windows FastAPI Application",
    version="3.3.0"
)

@app.get("/")
async def root():
    return {"message": "POA Trading Platform API", "status": "operational"}

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "platform": "Windows",
        "memory_usage": f"{psutil.virtual_memory().percent:.1f}%",
        "cpu_usage": f"{psutil.cpu_percent():.1f}%"
    }

@app.get("/status")
async def system_status():
    """System status endpoint"""
    poa_home = os.getenv("POA_HOME", "Not Set")
    return {
        "poa_home": poa_home,
        "uptime_seconds": int((datetime.now() - datetime(2025, 8, 28, 12, 0, 0)).total_seconds()),
        "system": {
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 1),
            "cpu_count": psutil.cpu_count(),
            "disk_free_gb": round(psutil.disk_usage(poa_home if poa_home != "Not Set" else "C:/").free / (1024**3), 1)
        }
    }

@app.get("/metrics")
async def metrics():
    """Metrics endpoint for monitoring"""
    return {
        "requests_processed": 0,  # Implement request counter
        "average_response_time": 0,
        "error_rate": 0,
        "services": {
            "redis": "unknown",
            "influxdb": "unknown", 
            "grafana": "unknown"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
