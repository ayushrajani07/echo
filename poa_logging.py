import os
import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logging(log_level="INFO", component_name="poa"):
    """Centralized logging setup for all POA components"""
    try:
        # Expand environment variables
        log_dir_raw = os.getenv("POA_LOGS", r"C:\Users\ASUS\Documents\POA\logs")
        log_dir = Path(os.path.expandvars(log_dir_raw))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create component-specific log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"{component_name}_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler()
            ],
            force=True  # Override any existing configuration
        )
        
        logging.info(f"Logging initialized for {component_name}")
        logging.info(f"Log file: {log_file}")
        
    except Exception as e:
        print(f"Failed to setup logging: {e}", file=sys.stderr)
        sys.exit(1)
