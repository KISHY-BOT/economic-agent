# Economic Analysis Agent — Railway Background Worker

This repository is a ready-to-deploy **background worker** for Railway that runs your economic analysis agent.
It connects easily to GitHub: each push will re-deploy and run the worker.

## What it does
- Uses `analysis_agent.py` (your agent) to run forecasting (ARIMA, Random Forest), Monte Carlo simulations,
  and optional credit risk lookups (mocked if the API is unavailable).
- Runs once on boot, then repeats every **6 hours** by default (configurable).

## Files
- `analysis_agent.py` — your agent code (from your upload).
- `worker.py` — the background loop that calls `demonstrate_agent()` and logs output.
- `requirements.txt` — Python dependencies.
- `Procfile` — defines a `worker` process type.
- `railway.json` — sets the start command if you don't want to use the Procfile.
- `.env.example` — environment variables you can copy into Railway.

## Deploy steps (Railway + GitHub)

1. **Create GitHub repo**
   - Create a new GitHub repository and push these files (`git init`, `git add .`, `git commit -m "init"`, `git remote add origin ...`, `git push -u origin main`).

2. **Connect on Railway**
   - Go to Railway → *New Project* → **Deploy from GitHub repo**.
   - Authorize Railway and select your repository.
   - Railway will auto-detect Python and install dependencies from `requirements.txt`.

3. **Set variables (optional)**
   - In *Variables*, add:
     - `RUN_ONCE` = `false` (or `true` to run a single cycle per deploy)
     - `RUNTIME_INTERVAL_SECONDS` = `21600` (6 hours) or another interval

4. **Confirm the process**
   - Railway will use either `railway.json`'s `startCommand` or the `Procfile`.
   - Start command: `python worker.py`

5. **Logs**
   - Open your service → *Logs* to see the output from each run.

## Local test
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python worker.py  # runs once, then every 6 hours
```

## Notes
- The LSTM function in `analysis_agent.py` requires TensorFlow; it's provided as a stub and will be skipped if not installed.
- If you need the worker to respond to a schedule (e.g., only at a specific time of day), you can set `RUN_ONCE=true`
  and trigger a redeploy with **Railway Deploy Hooks** or GitHub Actions on a cron.
