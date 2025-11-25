# server.py
# -*- coding: utf-8 -*-
"""
FastAPI server that runs the auto-learning Python predictor in a background thread,
serves a small static UI and broadcasts realtime prediction/results via WebSocket.
Compatible with Replit / Fly / any Linux VM.
"""

import os
import json
import time
import threading
import asyncio
import requests
from datetime import datetime
from collections import Counter
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ---------------- Config ----------------
TRAINING_FILE = "predictor_state.json"
INITIAL_HISTORY_SOURCE = os.environ.get("INITIAL_HISTORY_URL", "/mnt/data/win1m_results.json")
API_ENDPOINTS = [
    "https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json",
    "https://api.daman.games/WinGo/WinGo_1M/GetHistoryIssuePage.json",
    "https://api.tc-lottery.com/WinGo/WinGo_1M/GetHistoryIssuePage.json"
]

HISTORY_LOAD = int(os.environ.get("HISTORY_LOAD", "200"))
RECENT_WINDOW = int(os.environ.get("RECENT_WINDOW", "20"))
MARKOV_MIN_TRANSITIONS = int(os.environ.get("MARKOV_MIN_TRANSITIONS", "8"))
CONFIDENCE_MIN = float(os.environ.get("CONFIDENCE_MIN", "0.015"))
SIZE_CONF_MIN = float(os.environ.get("SIZE_CONF_MIN", "0.03"))
WAIT_AFTER_RESULT = int(os.environ.get("WAIT_AFTER_RESULT", "10"))
RUN_INTERVAL = int(os.environ.get("RUN_INTERVAL", "20"))

# Thread-safe broadcast manager for websockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: set[WebSocket] = set()
        self.lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self.lock:
            self.active_connections.add(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self.lock:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        async with self.lock:
            websockets = list(self.active_connections)
        for ws in websockets:
            try:
                await ws.send_json(message)
            except:
                # ignore send errors
                pass

manager = ConnectionManager()

# ---------------- State ----------------
state_lock = threading.Lock()
state = {
    "correct_number": 0,
    "correct_size": 0,
    "total": 0,
    "history": [],        # list of tuples: (pred_number_or_None, actual_number_int)
    "transitions": [],    # list of tuples: (prev_int, curr_int)
    "freq": {},           # dict number->count
    "pending": None,      # current pending prediction dict
}

# ---------------- Persistence ----------------
def save_state():
    try:
        with state_lock:
            with open(TRAINING_FILE, "w") as f:
                json.dump(state, f, indent=2)
    except Exception as e:
        print("Save failed:", e)

def load_state():
    try:
        if os.path.exists(TRAINING_FILE):
            with open(TRAINING_FILE, "r") as f:
                data = json.load(f)
            with state_lock:
                state.update(data)
            print("[INFO] Loaded existing state file.")
    except Exception as e:
        print("Load failed:", e)

# ---------------- Fetch functions ----------------
def safe_last_digit(n):
    try:
        return int(n) % 10
    except:
        try:
            return int(str(n)[-1])
        except:
            return 0

def fetch_full_history_from_local(path):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        nums = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    n = item.get("premium") or item.get("number") or item.get("result")
                    if n is not None:
                        nums.append(safe_last_digit(n))
                else:
                    nums.append(safe_last_digit(item))
        return nums
    except Exception as e:
        print("[WARN] local history load failed:", e)
        return []

def fetch_full_history_from_api(limit=HISTORY_LOAD):
    ts = int(time.time()*1000)
    for api in API_ENDPOINTS:
        try:
            url = f"{api}?ts={ts}&page=1&pageSize={limit}"
            r = requests.get(url, timeout=10)
            data = r.json().get("data", {}).get("list", [])
            if not data:
                continue
            numbers = []
            for d in data:
                n = d.get("premium") or d.get("number")
                numbers.append(safe_last_digit(n))
            print(f"[INFO] Loaded {len(numbers)} history records from {api}")
            return numbers
        except Exception as e:
            print("[WARN] history api failed", e)
            continue
    print("[WARN] No API responded for history")
    return []

def fetch_full_history(limit=HISTORY_LOAD):
    src = (INITIAL_HISTORY_SOURCE or "").strip()
    if src:
        if os.path.exists(src):
            nums = fetch_full_history_from_local(src)
            if nums:
                return nums
        # try to download if http(s)
        if src.startswith("http://") or src.startswith("https://"):
            try:
                r = requests.get(src, timeout=15)
                data = r.json()
                nums = []
                for d in data:
                    if isinstance(d, dict):
                        n = d.get("premium") or d.get("number") or d.get("result")
                        if n is not None:
                            nums.append(safe_last_digit(n))
                    else:
                        nums.append(safe_last_digit(d))
                if nums:
                    print(f"[INFO] Downloaded {len(nums)} history from {src}")
                    return nums
            except Exception as e:
                print("[WARN] Could not download initial history:", e)
    return fetch_full_history_from_api(limit)

def fetch_latest():
    ts = int(time.time()*1000)
    for api in API_ENDPOINTS:
        try:
            url = f"{api}?ts={ts}&page=1&pageSize=1"
            r = requests.get(url, timeout=8)
            dlist = r.json().get("data", {}).get("list", [])
            if not dlist:
                continue
            d = dlist[0]
            n = d.get("premium") or d.get("number")
            return {
                "issue": d.get("issueNumber", ""),
                "number": safe_last_digit(n),
                "raw": d
            }
        except Exception:
            continue
    return None

# ---------------- Prediction engine functions ----------------
def markov_distribution(last):
    with state_lock:
        transitions = list(state["transitions"])
        freq = dict(state["freq"])
    counts = [0]*10
    total = 0
    for a,b in transitions:
        if a == last:
            counts[b] += 1
            total += 1
    if total < MARKOV_MIN_TRANSITIONS:
        s = sum(freq.values())
        if s>0:
            return {i: freq.get(i,0)/s for i in range(10)}
        return {i: 1/10 for i in range(10)}
    return {i: counts[i]/total for i in range(10)}

def trend_distribution(window=RECENT_WINDOW):
    with state_lock:
        recent = [a for _, a in state["history"][-window:]]
    if not recent:
        return {i: 1/10 for i in range(10)}
    L = len(recent)
    weights = list(range(1, L+1))
    counts = [0]*10
    for w,v in zip(weights, recent):
        counts[v] += w
    s = sum(counts)
    if s==0:
        return {i: 1/10 for i in range(10)}
    return {i: counts[i]/s for i in range(10)}

def frequency_distribution():
    with state_lock:
        freq = dict(state["freq"])
    s = sum(freq.values())
    if s==0:
        return {i: 1/10 for i in range(10)}
    return {i: freq.get(i,0)/s for i in range(10)}

def auto_learn_weights():
    with state_lock:
        transitions = list(state["transitions"])
        history_vals = [x[1] for x in state["history"]]
    repeats = sum(1 for a,b in transitions if a==b)
    markov_strength = repeats / max(1, len(transitions) or 1)
    recent = history_vals[-RECENT_WINDOW:]
    streaks = sum(1 for i in range(1,len(recent)) if recent[i]==recent[i-1])
    trend_strength = streaks / max(1, len(recent) or 1)
    freq = Counter(history_vals)
    total = sum(freq.values())
    even = sum(freq[n] for n in [0,2,4,6,8]) if total else 0
    freq_imbalance = abs(even - (total - even)) / max(1, total)
    w_markov = 0.40 + markov_strength * 0.40
    w_trend  = 0.25 + trend_strength  * 0.30
    w_freq   = 0.10 + freq_imbalance  * 0.35
    s = w_markov + w_trend + w_freq
    w_markov /= s; w_trend /= s; w_freq /= s
    return w_markov, w_trend, w_freq

def combine_distributions():
    w_markov, w_trend, w_freq = auto_learn_weights()
    last = state["history"][-1][1] if state["history"] else None
    markov = markov_distribution(last) if last is not None else {i:1/10 for i in range(10)}
    trend = trend_distribution()
    freq = frequency_distribution()
    combined = {}
    for i in range(10):
        combined[i] = markov[i]*w_markov + trend[i]*w_trend + freq[i]*w_freq
    s = sum(combined.values())
    if s<=0:
        return {i:1/10 for i in range(10)}
    for i in combined:
        combined[i] /= s
    return combined

def predict_number_and_size():
    dist = combine_distributions()
    sorted_nums = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    top_num, top_prob = sorted_nums[0]
    second_prob = sorted_nums[1][1] if len(sorted_nums)>1 else 0.0
    gap = top_prob - second_prob
    number_conf = (top_prob * 0.55) + (gap * 0.45) + 0.02
    size_probs = {"Big":0.0,"Small":0.0}
    for num,p in dist.items():
        size_probs["Big" if num>=5 else "Small"] += p
    pred_size = "Big" if size_probs["Big"] >= size_probs["Small"] else "Small"
    size_conf = abs(size_probs["Big"] - size_probs["Small"])
    if size_conf < SIZE_CONF_MIN:
        return None, 0.0, None, 0.0
    return int(top_num), float(number_conf), pred_size, float(size_conf)

# ---------------- Background predictor loop ----------------
async def broadcast_update():
    # send a snapshot of current stats
    with state_lock:
        s = {
            "pending": state.get("pending"),
            "stats": {
                "total": state.get("total", 0),
                "correct_number": state.get("correct_number", 0),
                "correct_size": state.get("correct_size", 0),
            },
            "last_history": state["history"][-10:]
        }
    await manager.broadcast(s)

def predictor_loop(stop_event: threading.Event):
    load_state()
    # initialize from history
    nums = fetch_full_history(HISTORY_LOAD)
    if nums:
        with state_lock:
            state["history"] = [(None, int(n)) for n in nums]
            state["transitions"] = [(nums[i-1], nums[i]) for i in range(1,len(nums))]
            state["freq"] = dict(Counter(nums))
        save_state()
    last_issue = None
    pending = None
    pending_size = None
    while not stop_event.is_set():
        try:
            # Polling interval
            time.sleep(RUN_INTERVAL)
            res = fetch_latest()
            if not res:
                continue
            issue = res.get("issue", "")
            if issue == last_issue:
                continue
            current_three = issue[-3:] if len(issue)>=3 else issue
            upcoming_serial = str((int(current_three)+1) % 1000).zfill(3) if current_three.isdigit() else "000"
            actual_num = int(res["number"])
            actual_size = "Big" if actual_num >= 5 else "Small"
            # Evaluate pending if existed
            if pending is not None:
                with state_lock:
                    state["total"] = state.get("total",0) + 1
                    if pending == actual_num:
                        state["correct_number"] = state.get("correct_number",0) + 1
                    if pending_size == actual_size:
                        state["correct_size"] = state.get("correct_size",0) + 1
                    state["history"].append((pending, actual_num))
                    if len(state["history"]) > 1000:
                        state["history"].pop(0)
                    if len(state["history"]) >= 2:
                        prev = state["history"][-2][1]
                        state["transitions"].append((prev, actual_num))
                    state["freq"][actual_num] = state["freq"].get(actual_num,0) + 1
                save_state()
                # broadcast result
                asyncio.run(broadcast_update())
                pending = None
                pending_size = None
            last_issue = issue
            # wait WAIT_AFTER_RESULT seconds before predicting next
            for i in range(WAIT_AFTER_RESULT):
                if stop_event.is_set():
                    return
                time.sleep(1)
            # predict now
            pred_number, num_conf, pred_size, size_conf = predict_number_and_size()
            if pred_number is None:
                # low confidence, skip
                with state_lock:
                    state["pending"] = None
                asyncio.run(broadcast_update())
                continue
            with state_lock:
                pending = pred_number
                pending_size = pred_size
                state["pending"] = {
                    "number": pending,
                    "number_conf": num_conf,
                    "size": pending_size,
                    "size_conf": size_conf,
                    "serial": upcoming_serial,
                    "time": datetime.utcnow().isoformat() + "Z"
                }
            # broadcast new pending prediction
            asyncio.run(broadcast_update())
        except Exception as e:
            print("Predictor loop error:", e)
            time.sleep(3)

# ---------------- FastAPI app ----------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def index():
    return FileResponse("static/index.html")

@app.get("/api/stats")
async def api_stats():
    with state_lock:
        return {
            "total": state.get("total",0),
            "correct_number": state.get("correct_number",0),
            "correct_size": state.get("correct_size",0),
            "pending": state.get("pending"),
            "last_history": state["history"][-20:]
        }

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        # initial push
        with state_lock:
            await ws.send_json({
                "pending": state.get("pending"),
                "stats": {
                    "total": state.get("total",0),
                    "correct_number": state.get("correct_number",0),
                    "correct_size": state.get("correct_size",0),
                },
                "last_history": state["history"][-10:]
            })
        while True:
            # keep connection open; expect no messages from client
            await ws.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(ws)
    except Exception:
        await manager.disconnect(ws)

# ---------------- Startup / run ----------------
stop_event = threading.Event()
pred_thread = threading.Thread(target=predictor_loop, args=(stop_event,), daemon=True)

@app.on_event("startup")
async def startup_event():
    print("[STARTUP] starting predictor thread")
    if not pred_thread.is_alive():
        pred_thread.start()

@app.on_event("shutdown")
async def shutdown_event():
    print("[SHUTDOWN] stopping predictor")
    stop_event.set()
    pred_thread.join(timeout=5)
    save_state()

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
