#!/usr/bin/env python3
"""
POA Trading Platform - Development Mode
Starts services and runs FastAPI development server
"""

import logging
import uvicorn
from poa_logging import setup_logging
from poa_services import NativeWindowsServices
from dotenv import load_dotenv

load_dotenv()

def main():
    setup_logging(log_level="DEBUG", component_name="development")
    logging.info("Starting POA Development Mode")
    
    # Initialize services
    services = NativeWindowsServices()
    
    try:
        # Start services
        logging.info("Starting native Windows services...")
        service_results = services.start_native_services()
        
        # Start Redis (fallback to WSL if native fails)
        if service_results.get("Redis") != "running":
            logging.info("Attempting Redis WSL fallback...")
            services.start_redis_wsl()
        
        # Wait for Redis
        redis_client = services.wait_for_redis()
        
        # Print status
        status = services.get_service_status()
        logging.info(f"Service status: {status}")
        
        print("\n🚀 Development server starting...")
        print("   • FastAPI: http://localhost:8000")
        print("   • Health:  http://localhost:8000/health")
        print("   • Docs:    http://localhost:8000/docs")
        
        # Start FastAPI development server
        uvicorn.run(
            "src.main_native:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="debug"
        )
        
    except Exception as e:
        logging.error(f"Development mode failed: {e}")
        raise

if __name__ == "__main__":
    main()
