#!/usr/bin/env python3
"""
POA Trading Platform – Native Windows Installation Script
Automatically installs Chocolatey, required services, directories, and Python dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_powershell(command):
    result = subprocess.run(
        ["powershell", "-Command", command],
        capture_output=True, text=True
    )
    return result.returncode == 0, result.stdout + result.stderr

def ensure_env_dirs():
    poa_home   = Path(os.getenv("POA_HOME", r"C:\POA"))
    log_dir    = Path(os.getenv("POA_LOGS", poa_home / "logs"))
    data_dir   = Path(os.getenv("POA_DATA",  poa_home / "data"))
    config_dir = Path(os.getenv("POA_CONFIG",poa_home / "config"))
    for d in (log_dir, data_dir, config_dir):
        d.mkdir(parents=True, exist_ok=True)
    print(f"✅ Directories ready under {poa_home}")

def install_chocolatey():
    print("📦 Installing Chocolatey...")
    cmd = """
    Set-ExecutionPolicy Bypass -Scope Process -Force;
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072;
    iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    """
    success, output = run_powershell(cmd)
    print(output)
    return success

def install_services():
    print("🐳 Installing services via Chocolatey...")
    for pkg in ("redis-64", "influxdb", "grafana"):
        print(f"   • Installing {pkg}...")
        ok, out = run_powershell(f"choco install {pkg} -y")
        print(out)

def install_prometheus():
    print("🐳 Installing Prometheus manually...")
    url = "https://github.com/prometheus/prometheus/releases/download/v2.48.0/prometheus-2.48.0.windows-amd64.zip"
    subprocess.run(["powershell", "-Command", f"Invoke-WebRequest -Uri {url} -OutFile prometheus.zip"])
    subprocess.run(["powershell", "-Command", "Expand-Archive -Path prometheus.zip -DestinationPath C:\\prometheus"])
    print("✅ Prometheus extracted to C:\\prometheus")

def install_python_requirements():
    print("🐍 Installing Python requirements...")
    req_file = "requirements-windows.txt"
    if Path(req_file).exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_file], check=False)
        print(f"✅ Installed via {req_file}")
    else:
        print("⚠️ requirements-windows.txt not found")

def main():
    print("🚀 Starting Native Windows Installation for POA")
    ensure_env_dirs()
    if install_chocolatey():
        install_services()
        install_prometheus()
    else:
        print("⚠️ Chocolatey install failed, please install manually.")
    install_python_requirements()
    print("🎉 Installation complete. Next: run setup_native_services.ps1 as Administrator.")

if __name__ == "__main__":
    main()
