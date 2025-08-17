import os
import time
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

try:
    import analysis_agent
except Exception as e:
    analysis_agent = None
    logging.exception("analysis_agent not importable: %s", e)

INTERVAL = int(os.getenv("RUNTIME_INTERVAL_SECONDS", "21600"))  # default: cada 6 horas

def run_once():
    logging.info("Starting analysis cycle...")
    if analysis_agent is None:
        logging.error("analysis_agent not available")
        return
    try:
        if hasattr(analysis_agent, "demonstrate_agent"):
            try:
                analysis_agent.demonstrate_agent()
            except TypeError:
                # fallback si la firma no acepta kwargs
                analysis_agent.demonstrate_agent()
        else:
            logging.error("demonstrate_agent not found in analysis_agent")
    except Exception as exc:
        logging.exception("Analysis failed: %s", exc)
    logging.info("Analysis cycle finished.")

if __name__ == "__main__":
    run_once()
    if os.getenv("RUN_ONCE", "false").lower() in ("1", "true", "yes", "y"):
        logging.info("RUN_ONCE enabled -> exiting after single run.")
    else:
        logging.info("Entering loop with interval %s seconds", INTERVAL)
        while True:
            time.sleep(INTERVAL)
            run_once()
