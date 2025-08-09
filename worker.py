import os
import time
import logging
from datetime import datetime
# Your analysis agent code lives in analysis_agent.py
import analysis_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# Interval in seconds (default 6 hours). Set RUNTIME_INTERVAL_SECONDS in Railway variables to override.
INTERVAL = int(os.getenv("RUNTIME_INTERVAL_SECONDS", "21600"))

def run_once():
    logging.info("Starting analysis cycle...")
    try:
        analysis_agent.demonstrate_agent()
        logging.info("Analysis cycle finished.")
    except Exception as exc:
        logging.exception("Analysis failed: %s", exc)

if __name__ == "__main__":
    # If RUN_ONCE=true, just execute once (useful for manual deploys/tests)
    run_once()
    if os.getenv("RUN_ONCE", "false").lower() in ("1","true","yes","y"):
        logging.info("RUN_ONCE enabled -> exiting after single run.")
    else:
        logging.info("Entering loop with interval %s seconds", INTERVAL)
        while True:
            time.sleep(INTERVAL)
            run_once()
