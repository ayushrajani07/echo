import os
from pathlib import Path
import subprocess
import time
import redis
from dotenv import load_dotenv

load_dotenv()

def get_path_env(var_name, default=None):
    raw = os.getenv(var_name, default)
    if not raw:
        raise EnvironmentError(f"Environment variable {var_name} not set and no default provided.")
    expanded = os.path.expandvars(raw)
    return Path(expanded)

def start_redis_wsl():
    poa_home = get_path_env("POA_HOME", str(Path(__file__).parent))
    ps_script = poa_home / "setup" / "Start-WSLRedis.ps1"
    try:
        result = subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(ps_script)],
            check=True,
            capture_output=True,
            text=True
        )
        print("Redis start script output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error starting Redis via PowerShell script:", e.stderr)

def wait_for_redis(host='localhost', port=6379, retries=10, delay=3):
    r = redis.Redis(host=host, port=port)
    for attempt in range(retries):
        try:
            if r.ping():
                print("Connected to Redis!")
                return r
        except redis.exceptions.ConnectionError:
            print(f"Redis not available, retry {attempt + 1}/{retries}. Waiting {delay}s...")
            time.sleep(delay)
    raise Exception("Failed to connect to Redis after retries.")
