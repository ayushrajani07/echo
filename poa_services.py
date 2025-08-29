import os
import subprocess
import time
import redis
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class NativeWindowsServices:
    """Manages native Windows services for POA platform"""
    
    def __init__(self):
        self.poa_home = Path(os.getenv("POA_HOME", Path.cwd()))
        self.services = ["Redis", "influxdb", "grafana"]
        
    def start_redis_wsl(self):
        """Start Redis via WSL PowerShell script (fallback)"""
        ps_script = self.poa_home / "setup" / "Start-WSLRedis.ps1"
        try:
            result = subprocess.run(
                ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(ps_script)],
                check=True, capture_output=True, text=True, timeout=30
            )
            logging.info("Redis WSL startup completed")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Redis WSL startup failed: {e.stderr}")
            return False
        except Exception as e:
            logging.error(f"Redis WSL startup error: {e}")
            return False

    def start_native_services(self):
        """Start native Windows services"""
        results = {}
        for service in self.services:
            try:
                logging.info(f"Starting {service} service...")
                result = subprocess.run(
                    f'sc start "{service}"', 
                    shell=True, capture_output=True, text=True, timeout=30
                )
                
                if result.returncode == 0 or "already running" in result.stdout.lower():
                    results[service] = "running"
                    logging.info(f"{service} started successfully")
                else:
                    results[service] = "failed"
                    logging.error(f"{service} failed to start: {result.stderr}")
                    
            except Exception as e:
                results[service] = "error"
                logging.error(f"{service} startup error: {e}")
                
        return results

    def wait_for_redis(self, host='localhost', port=6379, retries=10, delay=3):
        """Wait for Redis to become available"""
        for attempt in range(retries):
            try:
                r = redis.Redis(host=host, port=port, socket_connect_timeout=5)
                if r.ping():
                    logging.info("Redis connection established")
                    return r
            except redis.exceptions.ConnectionError:
                logging.info(f"Redis connection attempt {attempt + 1}/{retries}")
                time.sleep(delay)
        
        raise Exception(f"Redis unavailable after {retries} attempts")

    def get_service_status(self):
        """Get status of all services"""
        status = {}
        for service in self.services:
            try:
                result = subprocess.run(
                    f'sc query "{service}"', 
                    shell=True, capture_output=True, text=True, timeout=10
                )
                status[service] = "running" if "RUNNING" in result.stdout else "stopped"
            except Exception:
                status[service] = "unknown"
        return status
