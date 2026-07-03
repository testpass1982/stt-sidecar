#!/usr/bin/env python3
"""stt-sidecar: локальный VOSK STT сервер + Rich TUI.

Первый запуск автоматически скачивает модель (~1.8GB).
Подбирает свободный порт начиная с 8081.
После запуска: ws://localhost:PORT/ws/transcribe + http://localhost:PORT/health
Выход: Ctrl+C
"""
import asyncio, json, logging, os, shutil, socket, sys, uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.request import urlretrieve

from fastapi import FastAPI, WebSocket
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
import uvicorn

# ── config ──────────────────────────────────────────────────────────
PORT = int(os.getenv("STT_PORT", "0"))  # 0 = auto
API_KEY = os.getenv("STT_API_KEY", "dev-key-123")
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "vosk-model-ru-0.42"

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger("stt-sidecar")

# ── shared state ────────────────────────────────────────────────────
class State:
    connections = 0
    last_text = ""
    total_chars = 0
    model_ready = False
    actual_port = 8081
    error = ""

state = State()

# ── auto-port ───────────────────────────────────────────────────────
def find_free_port(start=8081, end=8086):
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    return start  # если все заняты — пусть uvicorn упадёт

# ── model download ──────────────────────────────────────────────────
def _progress(blocknr, blocksize, totalsize):
    if totalsize > 0:
        downloaded = blocknr * blocksize
        pct = min(100, downloaded / totalsize * 100)
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        mb_dl = downloaded / 1024 / 1024
        mb_total = totalsize / 1024 / 1024
        print(f"\r  [{bar}] {pct:.0f}% ({mb_dl:.0f}/{mb_total:.0f} MB)", end="", flush=True)

def download_model():
    if MODEL_PATH.exists():
        state.model_ready = True
        return
    zip_path = MODEL_DIR / "vosk-model-ru-0.42.zip"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  ⬇ Скачиваю модель VOSK (big, ~1.8GB)...\n")
    try:
        urlretrieve(MODEL_URL, str(zip_path), _progress)
        print(f"\n  📦 Распаковываю...")
        shutil.unpack_archive(str(zip_path), str(MODEL_DIR))
        zip_path.unlink()
        print(f"  ✅ Модель готова: {MODEL_PATH}\n")
        state.model_ready = True
    except Exception as e:
        state.error = f"Ошибка загрузки модели: {e}"
        print(f"\n  ❌ {state.error}\n")

# ── VOSK engine (lazy load) ─────────────────────────────────────────
def get_recognizer():
    from vosk import Model, KaldiRecognizer
    if not hasattr(get_recognizer, "_model"):
        get_recognizer._model = Model(str(MODEL_PATH))
    return KaldiRecognizer(get_recognizer._model, 16000)

# ── FastAPI ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_app):
    async def _dl():
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, download_model)
    asyncio.create_task(_dl())
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "ok" if state.model_ready else "loading", "connections": state.connections, "port": state.actual_port}

@app.websocket("/ws/transcribe")
async def transcribe(ws: WebSocket):
    token = ws.query_params.get("token")
    if not token or token != API_KEY:
        await ws.close(code=4001)
        return
    await ws.accept()
    state.connections += 1
    logger.info("client connected (%d active)", state.connections)
    try:
        rec = get_recognizer()
        while True:
            data = await ws.receive_bytes()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                if text:
                    state.last_text = text
                    state.total_chars += len(text)
                    await ws.send_json({"text": text})
            else:
                partial = json.loads(rec.PartialResult()).get("partial", "")
                if partial:
                    await ws.send_json({"text": partial})
    except Exception:
        pass
    finally:
        state.connections -= 1
        logger.info("client disconnected (%d active)", state.connections)

# ── Rich TUI ────────────────────────────────────────────────────────
def make_layout():
    layout = Layout()
    layout.split_column(Layout(name="header", size=3), Layout(name="body"), Layout(name="footer", size=3))
    return layout

def render():
    url = f"ws://localhost:{state.actual_port}"
    hdr = Panel(Text("🎤 STT Sidecar — локальное распознавание речи", style="bold cyan"), box=box.ROUNDED)
    t = Table(box=box.SIMPLE)
    t.add_column("Параметр", style="bold")
    t.add_column("Значение")
    t.add_row("WebSocket", url)
    t.add_row("Модель", "✅ загружена" if state.model_ready else "⏳ загрузка...")
    t.add_row("Подключения", str(state.connections))
    t.add_row("Распознано символов", str(state.total_chars))
    t.add_row("Последний текст", state.last_text[:80] if state.last_text else "—")
    if state.error:
        t.add_row("Ошибка", Text(state.error, style="red"))
    body = Panel(t, title="📊 Статус", box=box.ROUNDED)
    ftr = Panel(Text(f"⬆ Открой tasks.webworx.ru → микрофон работает через {url}", style="dim"), box=box.ROUNDED)
    return hdr, body, ftr

async def tui_loop():
    layout = make_layout()
    try:
        with Live(layout, refresh_per_second=4, screen=True) as live:
            while True:
                h, b, f = render()
                layout["header"].update(h)
                layout["body"].update(b)
                layout["footer"].update(f)
                await asyncio.sleep(0.25)
    except asyncio.CancelledError:
        pass

# ── main ────────────────────────────────────────────────────────────
async def main():
    port = PORT if PORT else find_free_port()
    state.actual_port = port
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", ws_ping_interval=20)
    server = uvicorn.Server(config)
    tasks = [asyncio.create_task(server.serve()), asyncio.create_task(tui_loop())]
    _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("\n👋 До свидания!")
