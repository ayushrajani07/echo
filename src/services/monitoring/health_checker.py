#!/usr/bin/env python3
"""
OP TRADING PLATFORM - HEALTH CHECKER
====================================
Version: 3.1.2 - Enhanced Health Checker
Author: OP Trading Platform Team
Date: 2025-08-25 10:50 PM IST
"""

import logging
import asyncio
import psutil
from datetime import datetime
from typing import Dict, List, Any, Optional
import aiohttp
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class HealthChecker:
    """Enhanced health checker for system components."""
    
    def __init__(self):
        self.health_history = []
        self.component_status = {}
        self.thresholds = {
            "cpu_usage_percent": 80,
            "memory_usage_percent": 85,
            "disk_usage_percent": 90,
            "response_time_ms": 5000
        }
    
    async def perform_comprehensive_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of all components."""
        try:
            health_report = {
                "timestamp": datetime.now().isoformat(),
                "overall_status": "HEALTHY",
                "components": {}
            }
            
            # Check system resources
            health_report["components"]["system"] = await self._check_system_health()
            
            # Check external services
            health_report["components"]["influxdb"] = await self._check_influxdb_health()
            health_report["components"]["redis"] = await self._check_redis_health()
            
            # Determine overall status
            component_statuses = [comp.get("status", "UNKNOWN") for comp in health_report["components"].values()]
            
            if "UNHEALTHY" in component_statuses:
                health_report["overall_status"] = "UNHEALTHY"
            elif "DEGRADED" in component_statuses:
                health_report["overall_status"] = "DEGRADED"
            else:
                health_report["overall_status"] = "HEALTHY"
            
            # Store in history
            self.health_history.append(health_report)
            
            # Keep only last 100 health checks
            if len(self.health_history) > 100:
                self.health_history = self.health_history[-100:]
            
            return health_report
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_status": "UNKNOWN",
                "error": str(e)
            }
    
    async def _check_system_health(self) -> Dict[str, Any]:
        """Check system resource health."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Determine system status
            if (cpu_percent > self.thresholds["cpu_usage_percent"] or
                memory_percent > self.thresholds["memory_usage_percent"] or
                disk_percent > self.thresholds["disk_usage_percent"]):
                status = "DEGRADED"
            else:
                status = "HEALTHY"
            
            return {
                "status": status,
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory_percent,
                "disk_usage_percent": disk_percent,
                "available_memory_gb": memory.available / (1024**3),
                "free_disk_gb": disk.free / (1024**3),
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"System health check failed: {str(e)}")
            return {"status": "UNKNOWN", "error": str(e)}
    
    async def _check_influxdb_health(self) -> Dict[str, Any]:
        """Check InfluxDB health."""
        try:
            start_time = datetime.now()
            
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8086/health", timeout=10) as response:
                    response_time = (datetime.now() - start_time).total_seconds() * 1000
                    
                    if response.status == 200:
                        health_data = await response.json()
                        status = "HEALTHY" if health_data.get("status") == "pass" else "DEGRADED"
                    else:
                        status = "UNHEALTHY"
                    
                    return {
                        "status": status,
                        "response_time_ms": response_time,
                        "http_status": response.status,
                        "health_data": health_data if response.status == 200 else None,
                        "timestamp": datetime.now().isoformat()
                    }
                    
        except asyncio.TimeoutError:
            return {
                "status": "UNHEALTHY",
                "error": "Connection timeout",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "UNHEALTHY",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _check_redis_health(self) -> Dict[str, Any]:
        """Check Redis health."""
        try:
            start_time = datetime.now()
            
            redis_client = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=5)
            
            # Test ping
            await redis_client.ping()
            
            # Test basic operations
            test_key = "health_check_test"
            await redis_client.set(test_key, "test_value", ex=60)
            test_value = await redis_client.get(test_key)
            await redis_client.delete(test_key)
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Get Redis info
            info = await redis_client.info()
            
            await redis_client.close()
            
            return {
                "status": "HEALTHY",
                "response_time_ms": response_time,
                "test_operations": "PASSED",
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "Unknown"),
                "uptime_in_seconds": info.get("uptime_in_seconds", 0),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "UNHEALTHY",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_health_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get health trends over specified time period."""
        try:
            if not self.health_history:
                return {"message": "No health history available"}
            
            # Filter recent health checks
            cutoff_time = datetime.now().timestamp() - (hours * 3600)
            recent_checks = [
                check for check in self.health_history
                if datetime.fromisoformat(check["timestamp"]).timestamp() > cutoff_time
            ]
            
            if not recent_checks:
                return {"message": f"No health checks in last {hours} hours"}
            
            # Calculate trends
            healthy_count = sum(1 for check in recent_checks if check.get("overall_status") == "HEALTHY")
            degraded_count = sum(1 for check in recent_checks if check.get("overall_status") == "DEGRADED")
            unhealthy_count = sum(1 for check in recent_checks if check.get("overall_status") == "UNHEALTHY")
            
            total_checks = len(recent_checks)
            
            return {
                "time_period_hours": hours,
                "total_health_checks": total_checks,
                "healthy_percentage": (healthy_count / total_checks) * 100,
                "degraded_percentage": (degraded_count / total_checks) * 100,
                "unhealthy_percentage": (unhealthy_count / total_checks) * 100,
                "latest_check": recent_checks[-1] if recent_checks else None,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health trends calculation failed: {str(e)}")
            return {"error": "Trends calculation failed"}
