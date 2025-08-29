#!/usr/bin/env python3
"""
POA Trading Platform - Production Mode
Starts services and runs FastAPI production server
"""

import logging
import uvicorn
from poa_logging import setup_logging
from poa_services import NativeWindowsServices
from dotenv import load_dotenv

load_dotenv()

def main():
    setup_logging(log_level="INFO", component_name="production")
    logging.info("Starting POA Production Mode")
    
    # Initialize services
    services = NativeWindowsServices()
    
    try:
        # Start services
        logging.info("Starting production services...")
        service_results = services.start_native_services()
        
        # Start Redis (fallback to WSL if native fails)
        if service_results.get("Redis") != "running":
            logging.info("Attempting Redis WSL fallback...")
            services.start_redis_wsl()
            
        # Wait for Redis
        redis_client = services.wait_for_redis()
        
        # Print status
        status = services.get_service_status()
        logging.info(f"Production service status: {status}")
        
        print("\n🏭 Production server starting...")
        print("   • FastAPI: http://localhost:8000")
        print("   • Health:  http://localhost:8000/health")
        print("   • Metrics: http://localhost:8000/metrics")
        
        # Start FastAPI production server
        uvicorn.run(
            "src.main_native:app",
            host="0.0.0.0",
            port=8000,
            workers=4,
            reload=False,
            log_level="info"
        )
        
    except Exception as e:
        logging.error(f"Production mode failed: {e}")
        raise

if __name__ == "__main__":
    main()
