import logging
from poa_services import start_all_services, status_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("🚀 Starting POA Development Mode")
    start_all_services()
    status_report()
    logger.info("✅ Development services initialized and ready.")