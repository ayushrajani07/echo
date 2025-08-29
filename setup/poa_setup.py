#!/usr/bin/env python3
"""
POA Trading Platform - Native Windows Setup
Handles installation and configuration of native Windows services
"""

import logging
import subprocess
import time
from pathlib import Path
from poa_logging import setup_logging
from poa_services import NativeWindowsServices
from dotenv import load_dotenv

load_dotenv()

def install_chocolatey():
    """Install Chocolatey package manager"""
    logging.info("Installing Chocolatey...")
    cmd = """
    Set-ExecutionPolicy Bypass -Scope Process -Force;
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072;
    iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    """
    try:
        subprocess.run(f'powershell -Command "{cmd}"', check=True, timeout=300)
        logging.info("Chocolatey installed successfully")
        return True
    except Exception as e:
        logging.error(f"Chocolatey installation failed: {e}")
        return False

def install_services():
    """Install native Windows services via Chocolatey"""
    services = [("Redis", "redis-64"), ("InfluxDB", "influxdb"), ("Grafana", "grafana")]
    
    for service_name, package in services:
        logging.info(f"Installing {service_name}...")
        try:
            subprocess.run(f"choco install {package} -y", check=True, timeout=600)
            logging.info(f"{service_name} installed successfully")
        except Exception as e:
            logging.error(f"{service_name} installation failed: {e}")

def install_python_packages():
    """Install required Python packages"""
    packages = [
        "fastapi", "uvicorn[standard]", "redis", "influxdb-client",
        "prometheus-client", "python-dotenv", "psutil", "numpy"
    ]
    
    for package in packages:
        logging.info(f"Installing {package}...")
        try:
            subprocess.run(f"pip install {package}", check=True, timeout=180)
        except Exception as e:
            logging.error(f"Failed to install {package}: {e}")

def main():
    setup_logging(component_name="setup")
    logging.info("Starting POA Native Windows setup")
    
    # Initialize services manager
    services = NativeWindowsServices()
    
    try:
        # Setup sequence
        install_chocolatey()
        install_services()
        install_python_packages()
        
        # Test service startup
        services.start_native_services()
        
        logging.info("Setup completed successfully")
        print("\n✅ Setup completed! You can now run:")
        print("   python poa_development.py  (for development)")
        print("   python poa_production.py   (for production)")
        
    except Exception as e:
        logging.error(f"Setup failed: {e}")
        raise

if __name__ == "__main__":
    main()
