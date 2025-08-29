import logging
import subprocess
import redis
import requests
import time

logger = logging.getLogger(__name__)

# -------------------------
# Service Check Functions
# -------------------------
def redis_running(host='127.0.0.1', port=6379, timeout=1):
    try:
        r = redis.Redis(host=host, port=port, socket_connect_timeout=timeout)
        return r.ping()
    except redis.exceptions.ConnectionError:
        return False

def influxdb_running(url="http://localhost:8086/health"):
    try:
        r = requests.get(url, timeout=2)
        return r.status_code == 200 and r.json().get("status") == "pass"
    except Exception:
        return False

def grafana_running(url="http://localhost:3000/login"):
    try:
        r = requests.get(url, timeout=2)
        return r.status_code == 200
    except Exception:
        return False

# -------------------------
# Service Start Functions
# -------------------------
def start_redis_service():
    if redis_running():
        logger.info("✅ Redis already running — skipping start.")
        return True
    logger.info("🚀 Starting Redis service...")
    try:
        subprocess.run(["wsl", "sudo", "service", "redis-server", "start"], check=True)
        time.sleep(2)
        if redis_running():
            logger.info("✅ Redis started successfully.")
            return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Redis failed to start: {e}")
    return False

def start_influxdb_service():
    if influxdb_running():
        logger.info("✅ InfluxDB already running — skipping start.")
        return True
    logger.info("🚀 Starting InfluxDB service...")
    try:
        subprocess.run(["net", "start", "InfluxDB"], check=True)
        time.sleep(2)
        if influxdb_running():
            logger.info("✅ InfluxDB started successfully.")
            return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ InfluxDB failed to start: {e}")
    return False

def start_grafana_service():
    if grafana_running():
        logger.info("✅ Grafana already running — skipping start.")
        return True
    logger.info("🚀 Starting Grafana service...")
    try:
        subprocess.run(["net", "start", "Grafana"], check=True)
        time.sleep(2)
        if grafana_running():
            logger.info("✅ Grafana started successfully.")
            return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Grafana failed to start: {e}")
    return False

# -------------------------
# Service Stop Functions
# -------------------------
def stop_redis_service():
    if not redis_running():
        logger.info("ℹ️ Redis already stopped.")
        return True
    logger.info("🛑 Stopping Redis service...")
    try:
        subprocess.run(["wsl", "sudo", "service", "redis-server", "stop"], check=True)
        time.sleep(1)
        if not redis_running():
            logger.info("✅ Redis stopped successfully.")
            return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Redis failed to stop: {e}")
    return False

def stop_influxdb_service():
    if not influxdb_running():
        logger.info("ℹ️ InfluxDB already stopped.")
        return True
    logger.info("🛑 Stopping InfluxDB service...")
    try:
        subprocess.run(["net", "stop", "InfluxDB"], check=True)
        time.sleep(1)
        if not influxdb_running():
            logger.info("✅ InfluxDB stopped successfully.")
            return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ InfluxDB failed to stop: {e}")
    return False

def stop_grafana_service():
    if not grafana_running():
        logger.info("ℹ️ Grafana already stopped.")
        return True
    logger.info("🛑 Stopping Grafana service...")
    try:
        subprocess.run(["net", "stop", "Grafana"], check=True)
        time.sleep(1)
        if not grafana_running():
            logger.info("✅ Grafana stopped successfully.")
            return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Grafana failed to stop: {e}")
    return False

# -------------------------
# Combined Utility Functions
# -------------------------
def stop_all_services():
    logger.info("🛑 Stopping all services...")
    stop_grafana_service()
    stop_influxdb_service()
    stop_redis_service()
    logger.info("✅ All services stopped.")

def status_report():
    logger.info("📊 Service Status Report:")
    logger.info(f"Redis:    {'🟢 Running' if redis_running() else '🔴 Stopped'}")
    logger.info(f"InfluxDB: {'🟢 Running' if influxdb_running() else '🔴 Stopped'}")
    logger.info(f"Grafana:  {'🟢 Running' if grafana_running() else '🔴 Stopped'}")