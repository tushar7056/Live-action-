Deploy notes:

1) Set env var (optional):
   INITIAL_HISTORY_URL=/mnt/data/win1m_results.json
   (or a public https://raw.githubusercontent... link to your win1m_results.json)

2) Install deps:
   pip install -r requirements.txt

3) Run locally:
   uvicorn server:app --reload --host 0.0.0.0 --port 8000

4) On Replit:
   - Create a new Repl (Python)
   - Upload files and the 'static' folder
   - Add env var INITIAL_HISTORY_URL if you uploaded win1m_results.json outside repo
   - Set the Run command to: uvicorn server:app --host=0.0.0.0 --port=8000

5) On a VPS:
   Use systemd or screen to run the above uvicorn command; set automatic restart.

Websocket API: ws://<host>/ws
Stats API: https://<host>/api/stats
UI: https://<host>/
