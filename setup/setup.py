#!/usr/bin/env python3
"""
POA TRADING PLATFORM - SETUP COMPONENT
======================================
Version: 3.3.0 - Complete Setup & Installation System
Handles: Installation, Configuration, Service Setup
"""

import os
import sys
import subprocess
import time
import logging
from datetime import datetime
from pathlib import Path
import platform

# Environment setup
POA_HOME = Path(os.getenv("POA_HOME", Path.home() / "POA"))
LOG_DIR = Path(os.getenv("POA_LOGS", str(POA_HOME / "logs")))
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"poa_setup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class POASetup:
    def __init__(self):
        self.setup_start = time.time()
        self.steps_completed = 0
        self.warnings = []
        self.errors = []
        
    def print_banner(self):
        print("=" * 80)
        print("🚀 POA TRADING PLATFORM - SETUP COMPONENT v3.3.0")
        print("=" * 80)
        print(f"📋 Platform: {platform.platform()}")
        print(f"🐍 Python: {platform.python_version()}")
        print(f"🏠 POA_HOME: {POA_HOME}")
        print(f"📊 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST")
        print("-" * 80)

    def run_command(self, command, description=""):
        """Execute system command with logging."""
        try:
            logger.info(f"Executing: {command}")
            if description:
                print(f"🔄 {description}...")
                
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                logger.info(f"✅ Command succeeded: {command}")
                return True, result.stdout
            else:
                logger.error(f"❌ Command failed: {command}")
                logger.error(f"Error: {result.stderr}")
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏱️ Command timed out: {command}")
            return False, "Command timed out"
        except Exception as e:
            logger.error(f"💥 Command execution error: {str(e)}")
            return False, str(e)

    def install_chocolatey(self):
        """Install Chocolatey package manager."""
        print("\n📦 Step 1: Installing Chocolatey Package Manager")
        
        cmd = """
        Set-ExecutionPolicy Bypass -Scope Process -Force;
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072;
        iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
        """
        
        success, output = self.run_command(f'powershell -Command "{cmd}"', "Installing Chocolatey")
        
        if success:
            print("   ✅ Chocolatey installed successfully")
            self.steps_completed += 1
            return True
        else:
            print(f"   ❌ Chocolatey installation failed: {output}")
            self.errors.append("Chocolatey installation failed")
            return False

    def install_native_services(self):
        """Install all native Windows services."""
        print("\n🐳 Step 2: Installing Native Windows Services")
        
        services = [
            ("Redis", "redis-64"),
            ("InfluxDB", "influxdb"), 
            ("Grafana", "grafana")
        ]
        
        installed_services = 0
        
        for service_name, package in services:
            print(f"   📊 Installing {service_name}...")
            success, output = self.run_command(f"choco install {package} -y", f"Installing {service_name}")
            
            if success:
                print(f"   ✅ {service_name} installed")
                installed_services += 1
            else:
                print(f"   ❌ {service_name} failed: {output[:200]}...")
                self.warnings.append(f"{service_name} installation failed")
        
        # Install Prometheus manually
        print("   📈 Installing Prometheus...")
        prometheus_success = self.install_prometheus()
        if prometheus_success:
            installed_services += 1
            
        if installed_services >= 3:
            self.steps_completed += 1
            print(f"   ✅ Services installation completed ({installed_services}/4 services)")
            return True
        else:
            self.warnings.append(f"Only {installed_services}/4 services installed")
            return False

    def install_prometheus(self):
        """Install Prometheus manually."""
        try:
            prometheus_url = "https://github.com/prometheus/prometheus/releases/download/v2.48.0/prometheus-2.48.0.windows-amd64.zip"
            
            # Download
            success, _ = self.run_command(f'powershell -Command "Invoke-WebRequest -Uri {prometheus_url} -OutFile prometheus.zip"')
            if not success:
                return False
                
            # Extract
            success, _ = self.run_command('powershell -Command "Expand-Archive -Path prometheus.zip -DestinationPath C:\\prometheus -Force"')
            if not success:
                return False
                
            print("   ✅ Prometheus installed manually")
            return True
            
        except Exception as e:
            print(f"   ❌ Prometheus installation failed: {e}")
            return False

    def install_python_requirements(self):
        """Install Python requirements with fallback options."""
        print("\n🐍 Step 3: Installing Python Requirements")
        
        # Try different requirement files
        req_files = ["requirements-windows.txt", "requirements-minimal.txt", "requirements.txt"]
        
        for req_file in req_files:
            if Path(req_file).exists():
                print(f"   📋 Trying {req_file}...")
                success, output = self.run_command(f"pip install -r {req_file}", f"Installing from {req_file}")
                
                if success:
                    print(f"   ✅ Requirements installed from {req_file}")
                    self.steps_completed += 1
                    return True
                else:
                    print(f"   ⚠️ {req_file} failed, trying next...")
        
        # Fallback: essential packages
        print("   📦 Installing essential packages as fallback...")
        essential = [
            "fastapi", "uvicorn[standard]", "redis", "influxdb-client", 
            "prometheus-client", "python-dotenv", "psutil", "numpy==1.24.3"
        ]
        
        failed_packages = []
        for package in essential:
            success, _ = self.run_command(f"pip install {package}")
            if success:
                print(f"   ✅ {package}")
            else:
                print(f"   ❌ {package}")
                failed_packages.append(package)
        
        if len(failed_packages) < 3:  # Allow some failures
            self.steps_completed += 1
            print("   ✅ Essential packages installed")
            return True
        else:
            self.errors.append(f"Too many package failures: {failed_packages}")
            return False

    def setup_directories(self):
        """Create all required directories."""
        print("\n📁 Step 4: Setting up Directory Structure")
        
        directories = [
            POA_HOME / "logs" / "setup",
            POA_HOME / "logs" / "application", 
            POA_HOME / "logs" / "collectors",
            POA_HOME / "data" / "raw",
            POA_HOME / "data" / "processed",
            POA_HOME / "data" / "archive",
            POA_HOME / "config" / "services",
            POA_HOME / "config" / "collectors"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ {directory}")
        
        self.steps_completed += 1
        print("   ✅ Directory structure created")
        return True

    def configure_services(self):
        """Configure all native services."""
        print("\n⚙️ Step 5: Configuring Services")
        
        # Redis configuration
        redis_config = """
# Redis configuration for POA Trading Platform
port 6379
bind 127.0.0.1
maxmemory 512mb
maxmemory-policy allkeys-lru
save 60 1000
appendonly yes
appendfsync everysec
"""
        
        try:
            redis_config_path = POA_HOME / "config" / "services" / "redis.conf"
            redis_config_path.write_text(redis_config)
            print("   ✅ Redis configured")
        except Exception as e:
            print(f"   ⚠️ Redis config warning: {e}")
        
        # InfluxDB configuration
        influx_config = """
http-bind-address: ":8086"
storage-path: "{}/data/influxdb"
storage-wal-path: "{}/data/influxdb/wal"
storage-cache-max-memory-size: "512MB"
""".format(str(POA_HOME).replace('\\', '/'), str(POA_HOME).replace('\\', '/'))
        
        try:
            influx_config_path = POA_HOME / "config" / "services" / "influxdb.yml"
            influx_config_path.write_text(influx_config)
            print("   ✅ InfluxDB configured")
        except Exception as e:
            print(f"   ⚠️ InfluxDB config warning: {e}")
        
        self.steps_completed += 1
        print("   ✅ Service configuration completed")
        return True

    def validate_installation(self):
        """Validate the complete installation."""
        print("\n🏥 Step 6: Validating Installation")
        
        validation_results = {}
        
        # Check Python packages
        try:
            import fastapi, uvicorn, redis, influxdb_client, prometheus_client
            validation_results["python_packages"] = "✅ Healthy"
        except ImportError as e:
            validation_results["python_packages"] = f"❌ Missing: {e}"
        
        # Check directories
        if all(d.exists() for d in [POA_HOME/"logs", POA_HOME/"data", POA_HOME/"config"]):
            validation_results["directories"] = "✅ Healthy"
        else:
            validation_results["directories"] = "❌ Missing directories"
        
        # Check if services can be started (don't actually start them)
        service_check = self.run_command("sc query Redis", "Checking Redis service")[0]
        validation_results["redis_service"] = "✅ Available" if service_check else "⚠️ Not found"
        
        for check, status in validation_results.items():
            print(f"   • {check}: {status}")
        
        healthy_checks = sum(1 for status in validation_results.values() if "✅" in status)
        
        if healthy_checks >= 2:
            self.steps_completed += 1
            print("   ✅ Installation validation passed")
            return True
        else:
            self.warnings.append("Installation validation had issues")
            return False

    def print_summary(self):
        """Print setup completion summary."""
        setup_time = time.time() - self.setup_start
        
        print("\n" + "=" * 80)
        print("📋 SETUP COMPLETE - SUMMARY")
        print(f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST")
        print("-" * 80)
        
        print(f"🎉 Setup completed in {setup_time:.1f} seconds")
        print(f"✅ Steps Completed: {self.steps_completed}/6")
        
        if self.warnings:
            print(f"⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        if self.errors:
            print(f"❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"   • {error}")
        
        print("\n🌐 After Setup - Service URLs:")
        print("   • Redis: localhost:6379")
        print("   • InfluxDB: http://localhost:8086")
        print("   • Grafana: http://localhost:3000 (admin/admin)")
        print("   • Prometheus: http://localhost:9090")
        print("   • POA API: http://localhost:8000 (when running)")
        
        print(f"\n📄 Setup log: {LOG_DIR}/poa_setup_*.log")
        
        print("\n🚀 NEXT STEPS:")
        print("   1. Run: python poa_development.py (for development)")
        print("   2. Run: python poa_production.py (for production)")
        print("   3. Start services: ./setup_native_services.ps1")

    def run(self):
        """Run the complete setup process."""
        self.print_banner()
        
        try:
            # Execute all setup steps
            self.install_chocolatey()
            self.install_native_services()
            self.install_python_requirements()
            self.setup_directories()
            self.configure_services() 
            self.validate_installation()
            
        except KeyboardInterrupt:
            print("\n🛑 Setup interrupted by user")
            logger.info("Setup interrupted by user")
        except Exception as e:
            print(f"\n💥 Setup failed: {str(e)}")
            logger.error(f"Setup failed: {str(e)}")
            self.errors.append(str(e))
        finally:
            self.print_summary()

def main():
    setup = POASetup()
    setup.run()

if __name__ == "__main__":
    main()
