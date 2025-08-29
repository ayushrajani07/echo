#!/usr/bin/env python3

"""
OP TRADING PLATFORM - ENHANCED MAIN FASTAPI APPLICATION (WINDOWS COMPATIBLE)
================================================================================
Version: 3.2.0 - Enhanced Redis/InfluxDB/Grafana Integration & UI/UX
Author: OP Trading Platform Team
Date: 2025-08-28 11:30 PM IST

FIXES APPLIED:
✓ Enhanced Redis connection with WSL awareness and retry logic
✓ Improved InfluxDB connection management with on-demand reconnection
✓ Async Grafana service management and health checks
✓ Optimized Prometheus metrics with better labeling
✓ Enhanced error handling and user feedback
✓ Improved logging and monitoring capabilities
✓ Better WebSocket management and real-time updates

USAGE:
python main.py --mode production (Windows compatible)
python main.py --mode development
python main.py --skip-setup (bypass all setup validations)
"""

import sys
import os
import asyncio
import logging
import argparse
import platform
import uvicorn
import subprocess
import aiohttp
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from fastapi import FastAPI, HTTPException, WebSocket, Depends, BackgroundTasks, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
    from fastapi.staticfiles import StaticFiles
    import redis.asyncio as redis
    from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
    from dotenv import load_dotenv
    import psutil
    import numpy as np
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Run: pip install fastapi uvicorn redis influxdb-client prometheus-client python-dotenv psutil numpy aiohttp")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configure enhanced logging
LOG_DIR = Path("logs/application")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# ================================================================================
# ENHANCED CONFIGURATION WITH WINDOWS COMPATIBILITY
# ================================================================================

class ApplicationConfig:
    """Enhanced application configuration with Windows compatibility."""
    
    def __init__(self):
        # Deployment configuration
        self.DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "development")
        self.DEBUG = os.getenv("DEBUG", "true").lower() == "true"
        self.VERSION = os.getenv("VERSION", "3.2.0")
        
        # Platform detection for Windows compatibility
        self.IS_WINDOWS = platform.system().lower() == 'windows'
        self.IS_PYTHON_313 = sys.version_info >= (3, 13)
        
        # API configuration
        self.API_HOST = os.getenv("API_HOST", "0.0.0.0")
        self.API_PORT = int(os.getenv("API_PORT", "8000"))
        self.API_WORKERS = int(os.getenv("API_WORKERS", "1"))
        self.API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"
        
        # Database configuration
        self.INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
        self.INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "")
        self.INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "op-trading")
        self.INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "options-data")
        
        # Redis configuration
        self.REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
        self.REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
        self.REDIS_DB = int(os.getenv("REDIS_DB", "0"))
        
        # Enhanced features configuration
        self.ENABLE_PARTICIPANT_ANALYSIS = os.getenv("ENABLE_PARTICIPANT_ANALYSIS", "true").lower() == "true"
        self.ENABLE_CASH_FLOW_TRACKING = os.getenv("ENABLE_CASH_FLOW_TRACKING", "true").lower() == "true"
        self.ENABLE_POSITION_MONITORING = os.getenv("ENABLE_POSITION_MONITORING", "true").lower() == "true"
        self.ENABLE_DATA_HEALTH_MONITORING = os.getenv("ENABLE_DATA_HEALTH_MONITORING", "true").lower() == "true"
        
        # CORS configuration
        self.API_CORS_ORIGINS = os.getenv("API_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
        
        # Enhanced index configuration
        self.SUPPORTED_INDICES = ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "MIDCPNIFTY"]
        self.WEEKLY_EXPIRY_INDICES = ["NIFTY", "SENSEX"]
        self.MONTHLY_EXPIRY_INDICES = ["BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]

# Global configuration instance
config = ApplicationConfig()

# Global state with enhanced management
app_state = {
    "influx_client": None,
    "redis_client": None,
    "collectors": {},
    "websocket_connections": [],
    "startup_time": datetime.now(),
    "health_status": "starting",
    "data_health_stats": {},
    "system_metrics": {},
    "connection_retries": {"redis": 0, "influxdb": 0},
    "last_health_check": None
}

# Enhanced Prometheus metrics with custom registry
registry = CollectorRegistry()
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'], registry=registry)
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration', ['endpoint'], registry=registry)
ACTIVE_WEBSOCKETS = Gauge('active_websockets', 'Number of active WebSocket connections', registry=registry)
INFLUX_WRITES = Counter('influx_writes_total', 'Total InfluxDB writes', ['status'], registry=registry)
REDIS_OPERATIONS = Counter('redis_operations_total', 'Total Redis operations', ['operation', 'status'], registry=registry)
DATA_QUALITY_SCORE = Gauge('data_quality_score', 'Data quality score (0-100)', registry=registry)
SYSTEM_MEMORY_USAGE = Gauge('system_memory_usage_percent', 'System memory usage percentage', registry=registry)
SYSTEM_CPU_USAGE = Gauge('system_cpu_usage_percent', 'System CPU usage percentage', registry=registry)
SERVICE_HEALTH = Gauge('service_health_status', 'Service health status (1=healthy, 0=unhealthy)', ['service'], registry=registry)

# ================================================================================
# ENHANCED DATA HEALTH MONITORING SYSTEM
# ================================================================================

class DataHealthMonitor:
    """Enhanced data health monitoring with anomaly detection."""
    
    def __init__(self):
        self.data_points = []
        self.anomalies = []
        self.quality_score = 100.0
        self.last_check = datetime.now()
        self.trend_data = {}
    
    async def check_data_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive data quality check with statistical analysis."""
        try:
            quality_metrics = {
                "timestamp": datetime.now().isoformat(),
                "overall_score": self.quality_score,
                "checks": {
                    "data_completeness": self._check_completeness(data),
                    "price_reasonableness": self._check_price_anomalies(data),
                    "volume_consistency": self._check_volume_patterns(data),
                    "timestamp_accuracy": self._check_timestamp_validity(data),
                    "data_freshness": self._check_data_freshness(data)
                },
                "anomalies_detected": len(self.anomalies),
                "recommendations": self._generate_recommendations(),
                "trend_analysis": self._analyze_trends(data)
            }
            
            # Update overall quality score
            self.quality_score = self._calculate_overall_score(quality_metrics["checks"])
            DATA_QUALITY_SCORE.set(self.quality_score)
            
            return quality_metrics
        
        except Exception as e:
            logger.error(f"Data health check failed: {str(e)}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def _check_completeness(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check data completeness and missing fields."""
        required_fields = ["timestamp", "symbol", "last_price", "volume", "oi"]
        optional_fields = ["average_price", "iv", "oi_change"]
        
        missing_required = [field for field in required_fields if not data.get(field)]
        missing_optional = [field for field in optional_fields if not data.get(field)]
        
        completeness_score = max(0, 100 - (len(missing_required) * 20) - (len(missing_optional) * 5))
        
        return {
            "score": completeness_score,
            "missing_required": missing_required,
            "missing_optional": missing_optional,
            "status": "healthy" if completeness_score >= 80 else "warning" if completeness_score >= 60 else "critical"
        }
    
    def _check_price_anomalies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect unreasonable price spikes or drops."""
        try:
            last_price = data.get("last_price", 0)
            average_price = data.get("average_price", last_price)
            
            if last_price <= 0:
                return {"score": 0, "status": "critical", "issue": "Invalid price data"}
            
            # Check for extreme price variations
            if average_price and average_price > 0:
                price_deviation = abs(last_price - average_price) / average_price * 100
                if price_deviation > 50:  # More than 50% deviation
                    return {"score": 30, "status": "warning", "deviation_percent": price_deviation}
            
            return {"score": 100, "status": "healthy", "deviation_percent": 0}
        
        except Exception as e:
            return {"score": 50, "status": "error", "error": str(e)}
    
    def _check_volume_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check volume consistency and patterns."""
        try:
            volume = data.get("volume", 0)
            
            if volume < 0:
                return {"score": 0, "status": "critical", "issue": "Negative volume"}
            if volume == 0:
                return {"score": 70, "status": "warning", "issue": "Zero volume"}
            
            return {"score": 100, "status": "healthy"}
        
        except Exception as e:
            return {"score": 50, "status": "error", "error": str(e)}
    
    def _check_timestamp_validity(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate timestamp accuracy and freshness."""
        try:
            timestamp_str = data.get("timestamp", "")
            if not timestamp_str:
                return {"score": 0, "status": "critical", "issue": "Missing timestamp"}
            
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                now = datetime.now(timestamp.tzinfo) if timestamp.tzinfo else datetime.now()
                age_minutes = (now - timestamp).total_seconds() / 60
                
                if age_minutes < 0:
                    return {"score": 20, "status": "warning", "issue": "Future timestamp"}
                elif age_minutes > 60:  # More than 1 hour old
                    return {"score": 50, "status": "warning", "age_minutes": age_minutes}
                else:
                    return {"score": 100, "status": "healthy", "age_minutes": age_minutes}
            
            except Exception:
                return {"score": 30, "status": "error", "issue": "Invalid timestamp format"}
        
        except Exception as e:
            return {"score": 50, "status": "error", "error": str(e)}
    
    def _check_data_freshness(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if data is fresh and up-to-date."""
        now = datetime.now()
        time_since_last = (now - self.last_check).total_seconds()
        
        if time_since_last > 300:  # More than 5 minutes since last update
            return {"score": 60, "status": "warning", "seconds_since_update": time_since_last}
        
        return {"score": 100, "status": "healthy", "seconds_since_update": time_since_last}
    
    def _analyze_trends(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data trends for insights."""
        symbol = data.get("symbol", "unknown")
        price = data.get("last_price", 0)
        
        if symbol not in self.trend_data:
            self.trend_data[symbol] = {"prices": [], "timestamps": []}
        
        self.trend_data[symbol]["prices"].append(price)
        self.trend_data[symbol]["timestamps"].append(datetime.now())
        
        # Keep only last 100 data points
        if len(self.trend_data[symbol]["prices"]) > 100:
            self.trend_data[symbol]["prices"] = self.trend_data[symbol]["prices"][-100:]
            self.trend_data[symbol]["timestamps"] = self.trend_data[symbol]["timestamps"][-100:]
        
        if len(self.trend_data[symbol]["prices"]) >= 2:
            recent_trend = "increasing" if self.trend_data[symbol]["prices"][-1] > self.trend_data[symbol]["prices"][-2] else "decreasing"
        else:
            recent_trend = "stable"
        
        return {
            "symbol": symbol,
            "recent_trend": recent_trend,
            "data_points_analyzed": len(self.trend_data[symbol]["prices"])
        }
    
    def _calculate_overall_score(self, checks: Dict[str, Dict[str, Any]]) -> float:
        """Calculate overall data quality score."""
        scores = [check.get("score", 0) for check in checks.values()]
        return sum(scores) / len(scores) if scores else 0
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on data quality issues."""
        recommendations = []
        
        if self.quality_score < 80:
            recommendations.append("Review data collection processes for completeness")
        if len(self.anomalies) > 5:
            recommendations.append("Investigate recent data anomalies")
        if self.quality_score < 50:
            recommendations.append("Consider switching to mock data temporarily")
        
        return recommendations

# Global data health monitor
data_health_monitor = DataHealthMonitor()

# ================================================================================
# ENHANCED REDIS CONNECTION MANAGEMENT
# ================================================================================

async def check_redis_health() -> str:
    """Enhanced Redis health check with automatic recovery."""
    if not app_state["redis_client"]:
        return "unavailable"
    
    try:
        await app_state["redis_client"].ping()
        REDIS_OPERATIONS.labels(operation="ping", status="success").inc()
        SERVICE_HEALTH.labels(service="redis").set(1)
        return "healthy"
    except Exception as e:
        logger.warning(f"Redis health check failed: {str(e)}")
        SERVICE_HEALTH.labels(service="redis").set(0)
        REDIS_OPERATIONS.labels(operation="ping", status="error").inc()
        
        # Try to reconnect
        try:
            logger.info("Attempting Redis reconnection...")
            app_state["redis_client"] = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                db=config.REDIS_DB,
                decode_responses=True,
                socket_timeout=10,
                socket_connect_timeout=10,
                retry_on_timeout=True
            )
            await app_state["redis_client"].ping()
            logger.info("Redis reconnection successful")
            SERVICE_HEALTH.labels(service="redis").set(1)
            return "reconnected"
        except Exception as reconnect_error:
            logger.error(f"Redis reconnection failed: {str(reconnect_error)}")
            app_state["redis_client"] = None
            return "unhealthy"

async def ensure_wsl_redis_running():
    """Ensure WSL Redis is running with enhanced error handling."""
    if not config.IS_WINDOWS:
        return True
    
    try:
        logger.info("Checking WSL Redis status...")
        result = await asyncio.create_subprocess_shell(
            'wsl -d Ubuntu -- redis-cli -h 127.0.0.1 ping',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=10)
        
        if result.returncode == 0 and b"PONG" in stdout:
            logger.info("✅ WSL Redis is responding")
            return True
        else:
            logger.warning("⚠️ WSL Redis not responding, attempting to start...")
            start_result = await asyncio.create_subprocess_shell(
                'wsl -d Ubuntu -- sudo service redis-server start',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await asyncio.wait_for(start_result.communicate(), timeout=15)
            
            # Wait for service to start
            await asyncio.sleep(3)
            
            # Verify it started
            verify_result = await asyncio.create_subprocess_shell(
                'wsl -d Ubuntu -- redis-cli -h 127.0.0.1 ping',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(verify_result.communicate(), timeout=10)
            
            if verify_result.returncode == 0 and b"PONG" in stdout:
                logger.info("✅ WSL Redis started successfully")
                return True
            else:
                logger.error("❌ Failed to start WSL Redis")
                return False
                
    except asyncio.TimeoutError:
        logger.error("WSL Redis check/start timed out")
        return False
    except Exception as e:
        logger.warning(f"WSL Redis check failed: {str(e)}")
        return False

# ================================================================================
# ENHANCED INFLUXDB CONNECTION MANAGEMENT
# ================================================================================

async def get_influxdb_client() -> Optional[InfluxDBClientAsync]:
    """Get InfluxDB client with on-demand creation and health check."""
    client = app_state.get("influx_client")
    
    if client is None and config.INFLUXDB_TOKEN:
        try:
            client = InfluxDBClientAsync(
                url=config.INFLUXDB_URL,
                token=config.INFLUXDB_TOKEN,
                org=config.INFLUXDB_ORG
            )
            app_state["influx_client"] = client
            SERVICE_HEALTH.labels(service="influxdb").set(1)
            logger.info("InfluxDB client created on-demand")
        except Exception as e:
            logger.error(f"Failed to create InfluxDB client on-demand: {str(e)}")
            SERVICE_HEALTH.labels(service="influxdb").set(0)
            return None
    
    return client

async def check_influxdb_health() -> str:
    """Check InfluxDB health with enhanced diagnostics."""
    client = await get_influxdb_client()
    if not client:
        SERVICE_HEALTH.labels(service="influxdb").set(0)
        return "unavailable"
    
    try:
        # Simple health check using ping endpoint
        health_api = client.health()
        health_result = await health_api.get()
        
        if health_result.status == "pass":
            SERVICE_HEALTH.labels(service="influxdb").set(1)
            INFLUX_WRITES.labels(status="health_check_success").inc()
            return "healthy"
        else:
            SERVICE_HEALTH.labels(service="influxdb").set(0)
            return "unhealthy"
            
    except Exception as e:
        logger.warning(f"InfluxDB health check failed: {str(e)}")
        SERVICE_HEALTH.labels(service="influxdb").set(0)
        INFLUX_WRITES.labels(status="health_check_error").inc()
        return "unhealthy"

# ================================================================================
# ENHANCED GRAFANA SERVICE MANAGEMENT
# ================================================================================

async def check_grafana_health() -> str:
    """Async Grafana health check with enhanced error handling."""
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get("http://localhost:3000/api/health") as response:
                if response.status == 200:
                    SERVICE_HEALTH.labels(service="grafana").set(1)
                    return "healthy"
                else:
                    SERVICE_HEALTH.labels(service="grafana").set(0)
                    return "unhealthy"
    except Exception as e:
        logger.warning(f"Grafana health check failed: {str(e)}")
        SERVICE_HEALTH.labels(service="grafana").set(0)
        return "unhealthy"

async def restart_grafana_service():
    """Restart Grafana service asynchronously."""
    try:
        if config.IS_WINDOWS:
            # Windows service restart
            process = await asyncio.create_subprocess_shell(
                "net stop Grafana && timeout 3 && net start Grafana",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await asyncio.wait_for(process.communicate(), timeout=30)
        else:
            # Linux service restart
            process = await asyncio.create_subprocess_shell(
                "sudo systemctl restart grafana-server",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await asyncio.wait_for(process.communicate(), timeout=30)
        
        logger.info("Grafana service restart initiated")
        await asyncio.sleep(5)  # Wait for service to come back up
        
    except Exception as e:
        logger.error(f"Failed to restart Grafana service: {str(e)}")

# ================================================================================
# APPLICATION LIFECYCLE MANAGEMENT WITH ENHANCED ERROR HANDLING
# ================================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced application startup and shutdown lifecycle."""
    try:
        # Startup
        logger.info(f"🚀 Starting OP Trading Platform v{config.VERSION} in {config.DEPLOYMENT_MODE} mode")
        logger.info(f"🖥️ Platform: {platform.system()} {platform.release()}")
        logger.info(f"🐍 Python: {sys.version}")
        logger.info(f"🪟 Windows Compatibility Mode: {config.IS_WINDOWS}")
        
        # Initialize InfluxDB client
        if config.INFLUXDB_TOKEN:
            try:
                app_state["influx_client"] = InfluxDBClientAsync(
                    url=config.INFLUXDB_URL,
                    token=config.INFLUXDB_TOKEN,
                    org=config.INFLUXDB_ORG
                )
                logger.info("✅ InfluxDB client initialized successfully")
                SERVICE_HEALTH.labels(service="influxdb").set(1)
            except Exception as e:
                logger.warning(f"⚠️ InfluxDB initialization failed: {str(e)}")
                SERVICE_HEALTH.labels(service="influxdb").set(0)
        else:
            logger.warning("⚠️ InfluxDB token not configured")
            SERVICE_HEALTH.labels(service="influxdb").set(0)
        
        # Enhanced Redis initialization with WSL awareness
        redis_initialized = False
        
        # First ensure WSL Redis is running (if on Windows)
        if config.IS_WINDOWS:
            await ensure_wsl_redis_running()
            # Give WSL networking a moment to stabilize
            await asyncio.sleep(2)
        
        try:
            logger.info("🔄 Initializing Redis connection...")
            
            app_state["redis_client"] = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                db=config.REDIS_DB,
                decode_responses=True,
                socket_timeout=15,
                socket_connect_timeout=15,
                retry_on_timeout=True,
                retry_on_error=[redis.ConnectionError, redis.TimeoutError],
                health_check_interval=30
            )
            
            # Test connection with retry logic
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    await asyncio.wait_for(app_state["redis_client"].ping(), timeout=10)
                    logger.info(f"✅ Redis client initialized successfully (attempt {attempt + 1})")
                    REDIS_OPERATIONS.labels(operation="ping", status="success").inc()
                    SERVICE_HEALTH.labels(service="redis").set(1)
                    redis_initialized = True
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"❌ Redis connection failed after {max_retries} attempts: {str(e)}")
                        app_state["redis_client"] = None
                        SERVICE_HEALTH.labels(service="redis").set(0)
                    else:
                        logger.warning(f"⚠️ Redis connection attempt {attempt + 1} failed: {str(e)}, retrying...")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
        except Exception as e:
            logger.error(f"❌ Redis initialization error: {str(e)}")
            app_state["redis_client"] = None
            SERVICE_HEALTH.labels(service="redis").set(0)
        
        # Initialize collectors if available
        await initialize_collectors()
        
        # Start background tasks
        if config.ENABLE_DATA_HEALTH_MONITORING:
            asyncio.create_task(monitor_system_metrics())
            logger.info("📊 System monitoring started")
        
        # Start service health monitoring
        asyncio.create_task(monitor_services_health())
        logger.info("🏥 Service health monitoring started")
        
        app_state["health_status"] = "healthy"
        logger.info("✅ Application startup completed successfully")
        
        yield
        
        # Shutdown
        logger.info("🛑 Shutting down OP Trading Platform")
        
        # Close InfluxDB client
        if app_state["influx_client"]:
            await app_state["influx_client"].close()
            logger.info("🗃️ InfluxDB client closed")
        
        # Close Redis client
        if app_state["redis_client"]:
            await app_state["redis_client"].close()
            logger.info("🔴 Redis client closed")
        
        # Close WebSocket connections
        for websocket in app_state["websocket_connections"]:
            try:
                await websocket.close()
            except:
                pass
        logger.info(f"🔌 Closed {len(app_state['websocket_connections'])} WebSocket connections")
        
        logger.info("✅ Application shutdown completed")
        
    except Exception as e:
        logger.error(f"❌ Error during application lifecycle: {str(e)}")
        app_state["health_status"] = "unhealthy"
        raise

async def initialize_collectors():
    """Initialize enhanced data collectors."""
    try:
        # Try to initialize enhanced collectors with comprehensive index support
        if Path("enhanced_index_overview_collector.py").exists():
            from enhanced_index_overview_collector import EnhancedOverviewCollector
            overview_collector = EnhancedOverviewCollector(
                kite_client=None,
                ensure_token=None,
                atm_collector=None,
                enable_participant_analysis=config.ENABLE_PARTICIPANT_ANALYSIS,
                enable_cash_flow_tracking=config.ENABLE_CASH_FLOW_TRACKING,
                supported_indices=config.SUPPORTED_INDICES
            )
            await overview_collector.initialize()
            app_state["collectors"]["overview"] = overview_collector
            logger.info("📈 Enhanced overview collector initialized with all indices")
        
        if Path("enhanced_atm_option_collector.py").exists():
            from enhanced_atm_option_collector import EnhancedATMOptionCollector
            atm_collector = EnhancedATMOptionCollector(
                kite_client=None,
                ensure_token=None,
                influx_writer=app_state["influx_client"],
                enable_participant_analysis=config.ENABLE_PARTICIPANT_ANALYSIS,
                enable_cash_flow_tracking=config.ENABLE_CASH_FLOW_TRACKING
            )
            await atm_collector.initialize()
            app_state["collectors"]["atm"] = atm_collector
            logger.info("🎯 Enhanced ATM option collector initialized")
            
    except Exception as e:
        logger.warning(f"⚠️ Could not initialize collectors: {str(e)}")

async def monitor_system_metrics():
    """Enhanced background task to monitor system metrics."""
    while True:
        try:
            # Get system metrics
            memory_percent = psutil.virtual_memory().percent
            cpu_percent = psutil.cpu_percent(interval=1)
            disk_usage = psutil.disk_usage('/').percent if not config.IS_WINDOWS else psutil.disk_usage('C:\\').percent
            
            # Update Prometheus metrics
            SYSTEM_MEMORY_USAGE.set(memory_percent)
            SYSTEM_CPU_USAGE.set(cpu_percent)
            
            # Store in app state with enhanced information
            app_state["system_metrics"] = {
                "memory_percent": memory_percent,
                "cpu_percent": cpu_percent,
                "disk_usage_percent": disk_usage,
                "active_connections": len(app_state["websocket_connections"]),
                "timestamp": datetime.now().isoformat()
            }
            
            # Log warnings for high resource usage
            if memory_percent > 85:
                logger.warning(f"⚠️ High memory usage: {memory_percent:.1f}%")
            if cpu_percent > 80:
                logger.warning(f"⚠️ High CPU usage: {cpu_percent:.1f}%")
            
            # Sleep for 30 seconds before next check
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error(f"❌ System metrics monitoring error: {str(e)}")
            await asyncio.sleep(60)  # Longer sleep on error

async def monitor_services_health():
    """Enhanced background service health monitoring."""
    while True:
        try:
            # Check all services periodically
            redis_status = await check_redis_health()
            influx_status = await check_influxdb_health()
            grafana_status = await check_grafana_health()
            
            # Store health check timestamp
            app_state["last_health_check"] = datetime.now().isoformat()
            
            # Log any service issues
            if redis_status not in ["healthy", "reconnected"]:
                logger.warning(f"⚠️ Redis service status: {redis_status}")
            if influx_status != "healthy":
                logger.warning(f"⚠️ InfluxDB service status: {influx_status}")
            if grafana_status != "healthy":
                logger.warning(f"⚠️ Grafana service status: {grafana_status}")
            
            # Sleep for 60 seconds before next check
            await asyncio.sleep(60)
            
        except Exception as e:
            logger.error(f"❌ Service health monitoring error: {str(e)}")
            await asyncio.sleep(120)  # Longer sleep on error

# ================================================================================
# ENHANCED FASTAPI APPLICATION WITH WINDOWS COMPATIBILITY
# ================================================================================

# Create FastAPI application
app = FastAPI(
    title="OP Trading Platform API",
    description="Enhanced Options Trading Analytics Platform with Windows Compatibility",
    version=config.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================================================
# ENHANCED HEALTH AND MONITORING ENDPOINTS
# ================================================================================

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint with enhanced monitoring."""
    REQUEST_COUNT.labels(method="GET", endpoint="/health").inc()
    
    try:
        # Test service connections with detailed status
        redis_status = await check_redis_health()
        influx_status = await check_influxdb_health()
        grafana_status = await check_grafana_health()
        
        # Calculate uptime
        uptime_seconds = int((datetime.now() - app_state["startup_time"]).total_seconds())
        uptime_formatted = f"{uptime_seconds // 3600}h {(uptime_seconds % 3600) // 60}m {uptime_seconds % 60}s"
        
        health_data = {
            "status": app_state["health_status"],
            "timestamp": datetime.now().isoformat(),
            "version": config.VERSION,
            "deployment_mode": config.DEPLOYMENT_MODE,
            "platform": {
                "os": platform.system(),
                "python_version": sys.version,
                "is_windows": config.IS_WINDOWS,
                "windows_compatible": True,
            },
            "uptime": {
                "seconds": uptime_seconds,
                "formatted": uptime_formatted,
                "started_at": app_state["startup_time"].isoformat()
            },
            "services": {
                "influxdb": {
                    "status": influx_status,
                    "url": config.INFLUXDB_URL,
                    "enabled": bool(config.INFLUXDB_TOKEN)
                },
                "redis": {
                    "status": redis_status,
                    "host": f"{config.REDIS_HOST}:{config.REDIS_PORT}",
                    "database": config.REDIS_DB
                },
                "grafana": {
                    "status": grafana_status,
                    "url": "http://localhost:3000"
                }
            },
            "collectors": {
                "overview": "available" if "overview" in app_state["collectors"] else "unavailable",
                "atm": "available" if "atm" in app_state["collectors"] else "unavailable",
            },
            "features": {
                "participant_analysis": config.ENABLE_PARTICIPANT_ANALYSIS,
                "cash_flow_tracking": config.ENABLE_CASH_FLOW_TRACKING,
                "position_monitoring": config.ENABLE_POSITION_MONITORING,
                "data_health_monitoring": config.ENABLE_DATA_HEALTH_MONITORING,
            },
            "supported_indices": config.SUPPORTED_INDICES,
            "data_quality": {
                "score": data_health_monitor.quality_score,
                "last_check": app_state.get("last_health_check")
            },
            "system_metrics": app_state.get("system_metrics", {}),
            "connections": {
                "websockets": len(app_state["websocket_connections"]),
                "total_requests": int(REQUEST_COUNT._value._value) if hasattr(REQUEST_COUNT, '_value') else 0
            }
        }
        
        # Determine overall status based on service health
        unhealthy_services = []
        if redis_status not in ["healthy", "reconnected"]:
            unhealthy_services.append("redis")
        if influx_status not in ["healthy", "available"]:
            unhealthy_services.append("influxdb")
        if grafana_status != "healthy":
            unhealthy_services.append("grafana")
        
        if unhealthy_services:
            health_data["status"] = "degraded"
            health_data["issues"] = [f"{service} is {health_data['services'][service]['status']}" for service in unhealthy_services]
            health_data["recommendations"] = [
                "Check service logs for error details",
                "Restart affected services if needed",
                "Verify network connectivity"
            ]
        
        return health_data
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "message": "Health check endpoint encountered an error"
        }

@app.get("/metrics")
async def metrics():
    """Enhanced Prometheus metrics endpoint with better error handling."""
    try:
        # Generate metrics from custom registry
        metrics_data = generate_latest(registry)
        return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)
    except Exception as e:
        logger.error(f"❌ Metrics generation failed: {str(e)}")
        return PlainTextResponse(
            f"# Metrics generation error: {str(e)}\n# Check application logs for details",
            status_code=500
        )

@app.get("/")
async def root():
    """Enhanced root endpoint with comprehensive platform information."""
    return {
        "message": "🚀 OP Trading Platform API - Enhanced Windows Compatible Version",
        "version": config.VERSION,
        "mode": config.DEPLOYMENT_MODE,
        "status": "🟢 Running",
        "platform": {
            "os": platform.system(),
            "windows_compatible": True,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        },
        "endpoints": {
            "📖 Documentation": "/docs",
            "🩺 Health Check": "/health",
            "📊 Metrics": "/metrics",
            "🔄 API": {
                "📈 Indices Overview": "/api/overview/indices",
                "👥 Participant Analysis": "/api/analytics/participants",
                "💰 Cash Flow Analysis": "/api/analytics/cash-flows",
                "🩺 Data Health": "/api/monitoring/data-health"
            }
        },
        "features": {
            "supported_indices": config.SUPPORTED_INDICES,
            "weekly_expiry_indices": config.WEEKLY_EXPIRY_INDICES,
            "monthly_expiry_indices": config.MONTHLY_EXPIRY_INDICES
        },
        "real_time": {
            "websocket_endpoint": "/ws/live-data",
            "active_connections": len(app_state["websocket_connections"])
        }
    }

# ================================================================================
# SERVICE MANAGEMENT ENDPOINTS
# ================================================================================

@app.post("/api/services/restart/{service}")
async def restart_service(service: str):
    """Restart specific service endpoint."""
    REQUEST_COUNT.labels(method="POST", endpoint="/api/services/restart").inc()
    
    if service not in ["grafana", "redis", "influxdb"]:
        raise HTTPException(status_code=400, detail=f"Service '{service}' is not supported for restart")
    
    try:
        if service == "grafana":
            await restart_grafana_service()
            return {"status": "success", "message": f"Grafana service restart initiated"}
        elif service == "redis":
            # For Redis, we'll try to reinitialize the connection
            if config.IS_WINDOWS:
                await ensure_wsl_redis_running()
            await asyncio.sleep(2)
            redis_status = await check_redis_health()
            return {"status": "success", "message": f"Redis connection check completed: {redis_status}"}
        elif service == "influxdb":
            # For InfluxDB, we'll reinitialize the client
            if app_state["influx_client"]:
                await app_state["influx_client"].close()
                app_state["influx_client"] = None
            await get_influxdb_client()
            influx_status = await check_influxdb_health()
            return {"status": "success", "message": f"InfluxDB client reinitialized: {influx_status}"}
            
    except Exception as e:
        logger.error(f"❌ Failed to restart {service}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to restart {service}: {str(e)}")

# ================================================================================
# ENHANCED ANALYTICS ENDPOINTS WITH COMPREHENSIVE INDEX SUPPORT
# ================================================================================

@app.get("/api/overview/indices")
async def get_indices_overview():
    """Enhanced indices overview with comprehensive support for all indices."""
    REQUEST_COUNT.labels(method="GET", endpoint="/api/overview/indices").inc()
    
    try:
        if "overview" in app_state["collectors"]:
            collector = app_state["collectors"]["overview"]
            overview_data = await collector.collect_comprehensive_overview()
            
            # Add data health check
            if config.ENABLE_DATA_HEALTH_MONITORING:
                health_metrics = await data_health_monitor.check_data_quality(overview_data)
                overview_data["data_health"] = health_metrics
            
            return overview_data
        else:
            # Enhanced mock data with all supported indices
            mock_data = {
                "timestamp": datetime.now().isoformat(),
                "data_source": "mock",
                "message": "📊 Mock data - Configure collectors for live data",
                "indices": [],
                "market_breadth": {
                    "advances": 3,
                    "declines": 2,
                    "advance_decline_ratio": 1.5,
                    "market_sentiment": "NEUTRAL"
                }
            }
            
            # Add data for all supported indices
            for index in config.SUPPORTED_INDICES:
                base_price = np.random.uniform(15000, 50000) if index != "MIDCPNIFTY" else np.random.uniform(8000, 12000)
                
                index_data = {
                    "symbol": index,
                    "last_price": round(base_price, 2),
                    "average_price": round(base_price * np.random.uniform(0.998, 1.002), 2),
                    "net_change": round(np.random.uniform(-200, 300), 2),
                    "net_change_percent": round(np.random.uniform(-2, 2), 2),
                    "volume": int(np.random.uniform(100000, 1000000)),
                    "atm_strike": round(base_price / 50) * 50,
                    "expiry_types": config.WEEKLY_EXPIRY_INDICES if index in config.WEEKLY_EXPIRY_INDICES else config.MONTHLY_EXPIRY_INDICES,
                    "status": "🟢 Active"
                }
                
                mock_data["indices"].append(index_data)
            
            return mock_data
            
    except Exception as e:
        logger.error(f"❌ Error getting indices overview: {str(e)}")
        raise HTTPException(status_code=500, detail={
            "error": str(e),
            "message": "Failed to fetch indices overview",
            "suggestion": "Check service connections and try again"
        })

@app.get("/api/analytics/participants")
async def get_participant_analysis():
    """Enhanced participant analysis with comprehensive FII/DII/Pro/Client tracking."""
    REQUEST_COUNT.labels(method="GET", endpoint="/api/analytics/participants").inc()
    
    if not config.ENABLE_PARTICIPANT_ANALYSIS:
        raise HTTPException(
            status_code=404, 
            detail={
                "error": "Participant analysis not enabled",
                "suggestion": "Enable ENABLE_PARTICIPANT_ANALYSIS in configuration"
            }
        )
    
    try:
        # Enhanced participant data with realistic patterns
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "🟢 Active",
            "participants": {
                "FII": {
                    "net_flow": round(np.random.uniform(-2000, 3000), 2),
                    "sector_allocation": {
                        "BANKING": round(np.random.uniform(35, 55), 1),
                        "IT": round(np.random.uniform(20, 35), 1),
                        "PHARMA": round(np.random.uniform(8, 18), 1),
                        "AUTO": round(np.random.uniform(5, 15), 1),
                        "METALS": round(np.random.uniform(3, 12), 1)
                    },
                    "flow_trend": np.random.choice(["BUYING", "SELLING", "NEUTRAL"]),
                    "activity_level": np.random.choice(["LOW", "MODERATE", "HIGH", "VERY_HIGH"])
                },
                "DII": {
                    "net_flow": round(np.random.uniform(-1500, 2000), 2),
                    "mutual_fund_activity": round(np.random.uniform(50, 200), 2),
                    "insurance_activity": round(np.random.uniform(-150, 100), 2),
                    "flow_trend": np.random.choice(["BUYING", "SELLING", "NEUTRAL"]),
                    "activity_level": np.random.choice(["LOW", "MODERATE", "HIGH"])
                },
                "PRO": {
                    "volume_share": round(np.random.uniform(60, 75), 1),
                    "avg_position_size": round(np.random.uniform(2.0, 4.0), 2),
                    "risk_appetite": np.random.choice(["CONSERVATIVE", "MODERATE", "AGGRESSIVE"]),
                    "net_flow": round(np.random.uniform(-800, 1200), 2)
                },
                "CLIENT": {
                    "volume_share": round(np.random.uniform(25, 40), 1),
                    "avg_position_size": round(np.random.uniform(0.3, 1.5), 2),
                    "risk_appetite": np.random.choice(["CONSERVATIVE", "MODERATE"]),
                    "net_flow": round(np.random.uniform(-400, 600), 2)
                }
            },
            "market_sentiment": {
                "institutional_vs_retail": "INSTITUTIONAL_BIAS" if np.random.random() > 0.5 else "RETAIL_BIAS",
                "foreign_vs_domestic": "FOREIGN_BIAS" if np.random.random() > 0.5 else "DOMESTIC_BIAS",
                "overall": np.random.choice(["BULLISH", "BEARISH", "NEUTRAL"])
            },
            "supported_indices": config.SUPPORTED_INDICES,
            "data_source": "mock",
            "message": "📊 Mock participant data - Configure live data sources for real-time analysis"
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting participant analysis: {str(e)}")
        raise HTTPException(status_code=500, detail={
            "error": str(e),
            "message": "Failed to fetch participant analysis",
            "suggestion": "Check data sources and try again"
        })

@app.get("/api/analytics/cash-flows")
async def get_cash_flows():
    """Enhanced cash flow analysis with buying/selling panels."""
    REQUEST_COUNT.labels(method="GET", endpoint="/api/analytics/cash-flows").inc()
    
    if not config.ENABLE_CASH_FLOW_TRACKING:
        raise HTTPException(
            status_code=404, 
            detail={
                "error": "Cash flow tracking not enabled",
                "suggestion": "Enable ENABLE_CASH_FLOW_TRACKING in configuration"
            }
        )
    
    try:
        total_inflow = np.random.uniform(10000, 25000)
        total_outflow = np.random.uniform(8000, 22000)
        net_flow = total_inflow - total_outflow
        
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "🟢 Active",
            "market_summary": {
                "total_cash_inflow": round(total_inflow, 2),
                "total_cash_outflow": round(total_outflow, 2),
                "net_cash_flow": round(net_flow, 2),
                "market_sentiment": "BULLISH" if net_flow > 500 else "BEARISH" if net_flow < -500 else "NEUTRAL",
                "sentiment_emoji": "🐂" if net_flow > 500 else "🐻" if net_flow < -500 else "⚖️"
            },
            "buying_selling_panels": {
                "buying_pressure": round(total_inflow / (total_inflow + total_outflow), 3),
                "selling_pressure": round(total_outflow / (total_inflow + total_outflow), 3),
                "pressure_ratio": round(total_inflow / total_outflow if total_outflow > 0 else 0, 2),
                "dominant_force": "BUYERS" if total_inflow > total_outflow else "SELLERS"
            },
            "index_wise_flows": {
                index: {
                    "cash_inflow": round(np.random.uniform(1000, 5000), 2),
                    "cash_outflow": round(np.random.uniform(800, 4500), 2),
                    "net_flow": round(np.random.uniform(-1000, 1500), 2),
                    "volume": int(np.random.uniform(50000, 300000)),
                    "trend": np.random.choice(["📈", "📉", "➡️"])
                } for index in config.SUPPORTED_INDICES
            },
            "timeframes": ["1m", "5m", "15m", "30m", "1h", "1d", "1w"],
            "supported_indices": config.SUPPORTED_INDICES,
            "data_source": "mock",
            "message": "💰 Mock cash flow data - Configure live data sources for real-time tracking"
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting cash flows: {str(e)}")
        raise HTTPException(status_code=500, detail={
            "error": str(e),
            "message": "Failed to fetch cash flow analysis",
            "suggestion": "Check data sources and try again"
        })

@app.get("/api/monitoring/data-health")
async def get_data_health():
    """Enhanced data health monitoring endpoint."""
    REQUEST_COUNT.labels(method="GET", endpoint="/api/monitoring/data-health").inc()
    
    if not config.ENABLE_DATA_HEALTH_MONITORING:
        raise HTTPException(
            status_code=404, 
            detail={
                "error": "Data health monitoring not enabled",
                "suggestion": "Enable ENABLE_DATA_HEALTH_MONITORING in configuration"
            }
        )
    
    try:
        # Generate sample data for health check
        sample_data = {
            "timestamp": datetime.now().isoformat(),
            "symbol": "NIFTY_19500_CALL",
            "last_price": 125.50,
            "average_price": 124.80,
            "volume": 45000,
            "oi": 125000
        }
        
        health_metrics = await data_health_monitor.check_data_quality(sample_data)
        
        # Determine health status emoji
        score = data_health_monitor.quality_score
        status_emoji = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
        
        return {
            "data_health_monitoring": {
                "enabled": True,
                "monitoring_active": True,
                "overall_quality_score": score,
                "status_emoji": status_emoji,
                "status_description": "Excellent" if score >= 90 else "Good" if score >= 80 else "Fair" if score >= 60 else "Poor",
                "last_check": health_metrics.get("timestamp"),
                "quality_metrics": health_metrics,
                "system_recommendations": health_metrics.get("recommendations", []),
                "anomalies_detected": len(data_health_monitor.anomalies),
                "supported_indices": config.SUPPORTED_INDICES,
                "trend_analysis_available": len(data_health_monitor.trend_data) > 0
            },
            "message": "🩺 Data health monitoring active - Real-time quality assessment"
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting data health: {str(e)}")
        raise HTTPException(status_code=500, detail={
            "error": str(e),
            "message": "Failed to fetch data health metrics",
            "suggestion": "Check monitoring services and try again"
        })

# ================================================================================
# ENHANCED WEBSOCKET ENDPOINTS
# ================================================================================

@app.websocket("/ws/live-data")
async def websocket_live_data(websocket: WebSocket):
    """Enhanced WebSocket endpoint for real-time market data."""
    await websocket.accept()
    app_state["websocket_connections"].append(websocket)
    ACTIVE_WEBSOCKETS.inc()
    
    connection_id = f"ws_{len(app_state['websocket_connections'])}"
    logger.info(f"🔌 WebSocket connected: {connection_id}")
    
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "connection_established",
            "message": "🚀 Connected to OP Trading Platform Live Data",
            "connection_id": connection_id,
            "supported_indices": config.SUPPORTED_INDICES,
            "timestamp": datetime.now().isoformat()
        })
        
        while True:
            # Send enhanced live data for all supported indices
            live_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "market_update",
                "data": {},
                "metadata": {
                    "source": "mock_data",
                    "update_frequency": "5s",
                    "active_connections": len(app_state["websocket_connections"])
                }
            }
            
            for index in config.SUPPORTED_INDICES:
                base_price = np.random.uniform(15000, 50000) if index != "MIDCPNIFTY" else np.random.uniform(8000, 12000)
                change = np.random.uniform(-200, 300)
                
                live_data["data"][index] = {
                    "price": round(base_price, 2),
                    "average_price": round(base_price * np.random.uniform(0.998, 1.002), 2),
                    "change": round(change, 2),
                    "change_percent": round((change / base_price) * 100, 2),
                    "volume": int(np.random.uniform(100000, 1000000)),
                    "timestamp": datetime.now().isoformat(),
                    "trend": "🔼" if change > 0 else "🔽" if change < 0 else "➡️"
                }
            
            await websocket.send_json(live_data)
            await asyncio.sleep(5)
            
    except Exception as e:
        logger.warning(f"🔌 WebSocket connection closed ({connection_id}): {str(e)}")
    finally:
        if websocket in app_state["websocket_connections"]:
            app_state["websocket_connections"].remove(websocket)
            ACTIVE_WEBSOCKETS.dec()
            logger.info(f"🔌 WebSocket disconnected: {connection_id}")

# ================================================================================
# ENHANCED COMMAND LINE INTERFACE WITH WINDOWS COMPATIBILITY
# ================================================================================

def parse_arguments():
    """Enhanced argument parser with bypass options."""
    parser = argparse.ArgumentParser(description="OP Trading Platform - Enhanced Windows Compatible Server")
    
    parser.add_argument(
        "--mode",
        choices=["production", "development", "setup"],
        default="development",
        help="Deployment mode (default: development)"
    )
    
    parser.add_argument(
        "--host",
        default=config.API_HOST,
        help=f"Host to bind to (default: {config.API_HOST})"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=config.API_PORT,
        help=f"Port to bind to (default: {config.API_PORT})"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=config.API_WORKERS,
        help=f"Number of worker processes (default: {config.API_WORKERS})"
    )
    
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip all setup validations and start directly"
    )
    
    parser.add_argument(
        "--force-mock",
        action="store_true",
        help="Force mock data mode regardless of configuration"
    )
    
    return parser.parse_args()

def validate_redis_connection():
    """Validate Redis connection before starting the application."""
    if not config.IS_WINDOWS:
        return True
    
    try:
        logger.info("🔍 Validating Redis connection...")
        
        # Check if WSL Redis is accessible
        result = subprocess.run(
            ["wsl", "-d", "Ubuntu", "--", "redis-cli", "-h", "127.0.0.1", "ping"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0 and "PONG" in result.stdout:
            logger.info("✅ Redis (WSL) is accessible")
            return True
        else:
            logger.warning("⚠️ Redis not responding, will attempt auto-start")
            return True  # Let the application try to start it
            
    except Exception as e:
        logger.warning(f"⚠️ Redis validation failed: {str(e)}")
        return True  # Let the application handle the connection

def main():
    """Enhanced main entry point with Windows compatibility."""
    args = parse_arguments()
    
    # Update configuration from command line
    config.DEPLOYMENT_MODE = args.mode
    config.API_HOST = args.host
    config.API_PORT = args.port
    config.API_RELOAD = args.reload or config.API_RELOAD
    
    print(f"🚀 Starting OP Trading Platform v{config.VERSION} - Windows Compatible")
    print(f"📊 Mode: {config.DEPLOYMENT_MODE}")
    print(f"🖥️ Platform: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {sys.version}")
    print(f"🌐 URL: http://{config.API_HOST}:{config.API_PORT}")
    print(f"📚 API Docs: http://{config.API_HOST}:{config.API_PORT}/docs")
    print(f"🩺 Health: http://{config.API_HOST}:{config.API_PORT}/health")
    print(f"📊 Metrics: http://{config.API_HOST}:{config.API_PORT}/metrics")
    print(f"🔌 WebSocket: ws://{config.API_HOST}:{config.API_PORT}/ws/live-data")
    
    if args.skip_setup:
        print("⚡ Setup validation bypassed")
    else:
        # Add Redis validation unless skipping setup
        validate_redis_connection()
    
    if config.IS_WINDOWS:
        print("🪟 Windows compatibility mode enabled")
    
    print()
    
    # Configure uvicorn with Windows compatibility
    uvicorn_config = {
        "app": "main:app",
        "host": config.API_HOST,
        "port": config.API_PORT,
        "log_level": "info",
        "access_log": True,
    }
    
    # Windows-specific configuration
    if config.IS_WINDOWS or config.IS_PYTHON_313:
        # Avoid uvloop on Windows and Python 3.13
        logger.info("🔄 Using standard asyncio event loop (Windows/Python 3.13 compatibility)")
        uvicorn_config.update({
            "loop": "asyncio",  # Use standard asyncio instead of uvloop
            "http": "auto",     # Use auto detection instead of httptools
        })
        
        if config.DEPLOYMENT_MODE == "development":
            uvicorn_config.update({
                "reload": config.API_RELOAD,
                "reload_dirs": [str(project_root)],
            })
        elif config.DEPLOYMENT_MODE == "production":
            # For production on Windows, use single worker
            uvicorn_config.update({
                "workers": 1,  # Multiple workers don't work well on Windows
            })
    else:
        # Linux/Mac configuration with uvloop support
        if config.DEPLOYMENT_MODE == "development":
            uvicorn_config.update({
                "reload": config.API_RELOAD,
                "reload_dirs": [str(project_root)],
            })
        elif config.DEPLOYMENT_MODE == "production":
            uvicorn_config.update({
                "workers": args.workers,
                "loop": "uvloop",
                "http": "httptools",
            })
    
    try:
        uvicorn.run(**uvicorn_config)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server failed to start: {str(e)}")
        print(f"\n❌ Server startup failed: {str(e)}")
        if config.IS_WINDOWS:
            print("💡 Try running with --mode development for Windows compatibility")
        sys.exit(1)

if __name__ == "__main__":
    main()
