#!/usr/bin/env python3
"""stt-sidecar: локальный VOSK STT сервер + Rich TUI.

При первом запуске спрашивает small (40MB) или big (1.8GB) модель.
Подбирает свободный порт начиная с 8081.
После запуска: ws://localhost:PORT/ws/transcribe + http://localhost:PORT/health
Выход: Ctrl+C
"""
import asyncio, json, logging, os, shutil, socket, sys, uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.request import urlretrieve

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from rich.console import Console
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
MODEL_DIR = Path(__file__).parent / "models"

MODELS = {
    "small": {
        "name": "vosk-model-small-ru-0.22",
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip",
        "size": "40 MB",
    },
    "big": {
        "name": "vosk-model-ru-0.42",
        "url": "https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip",
        "size": "1.8 GB",
    },
}

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger("stt-sidecar")
console = Console()

# ── shared state ────────────────────────────────────────────────────
class State:
    connections = 0
    last_text = ""
    total_chars = 0
    model_ready = False
    model_name = ""
    actual_port = 8081
    error = ""

state = State()

# ── auto-port ───────────────────────────────────────────────────────
def find_free_port(start=8081, end=8086):
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    return start

# ── model download ──────────────────────────────────────────────────
def _progress(blocknr, blocksize, totalsize):
    if totalsize > 0:
        downloaded = blocknr * blocksize
        pct = min(100, downloaded / totalsize * 100)
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        mb_dl = downloaded / 1024 / 1024
        mb_total = totalsize / 1024 / 1024
        print(f"\r  [{bar}] {pct:.0f}% ({mb_dl:.0f}/{mb_total:.0f} MB)", end="", flush=True)

def ensure_model():
    # Проверяем, есть ли уже одна из моделей
    for key, cfg in MODELS.items():
        p = MODEL_DIR / cfg["name"]
        if p.exists():
            state.model_ready = True
            state.model_name = cfg["name"]
            return key, cfg

    # Нет модели — спрашиваем (до TUI, обычный input)
    console.print()
    console.print("[bold cyan]🎤 STT Sidecar[/bold cyan] — выберите модель:", style="bold")
    console.print("  [1] small  (vosk-model-small-ru-0.22, 40 MB) — [green]быстрая загрузка[/green]")
    console.print("  [2] big    (vosk-model-ru-0.42, 1.8 GB) — точнее, но долго качать")
    console.print()
    choice = console.input("[bold]Ваш выбор [1/2]: [/bold]").strip()
    key = "big" if choice == "2" else "small"
    cfg = MODELS[key]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = MODEL_DIR / f"{cfg['name']}.zip"
    model_path = MODEL_DIR / cfg["name"]

    console.print(f"\n  ⬇ Скачиваю {cfg['name']} ({cfg['size']})...\n")
    try:
        urlretrieve(cfg["url"], str(zip_path), _progress)
        console.print(f"\n  📦 Распаковываю...")
        shutil.unpack_archive(str(zip_path), str(MODEL_DIR))
        zip_path.unlink()
        console.print(f"  ✅ Модель готова: {model_path}\n")
    except Exception as e:
        console.print(f"  ❌ Ошибка: {e}")
        sys.exit(1)

    state.model_ready = True
    state.model_name = cfg["name"]
    return key, cfg

# ── VOSK engine (lazy load) ─────────────────────────────────────────
def get_recognizer():
    from vosk import Model, KaldiRecognizer
    if not hasattr(get_recognizer, "_model"):
        model_path = MODEL_DIR / state.model_name
        get_recognizer._model = Model(str(model_path))
    return KaldiRecognizer(get_recognizer._model, 16000)

# ── FastAPI ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_app):
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

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
    t.add_row("Модель", f"✅ {state.model_name}" if state.model_ready else "⏳ загрузка...")
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
        ensure_model()  # спрашивает и качает ДО TUI
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("\n👋 До свидания!")
