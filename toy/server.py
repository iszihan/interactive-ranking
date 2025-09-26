# server.py
from pathlib import Path
import shutil
from typing import List, Optional
import asyncio
import base64
import json
import random
import time
import os
import pathlib


from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from io import BytesIO
from PIL import Image, ImageDraw


FRONTEND_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = FRONTEND_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SLOTS = 6
SLOTS_DIR = OUTPUT_DIR / "slots"
SLOTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

# helpers ------------------------------------------------------------


def _make_rank_png_bytes(step: int, i: int, name: str) -> bytes:
    """Sync helper (runs in thread) to build the PNG bytes."""
    time.sleep(1.0)
    img = Image.new("RGB", (200, 200), (220, 240, 255))
    d = ImageDraw.Draw(img)
    d.text((20, 80), f"Step {step} Rank {i+1}: {name}", fill=(0, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


class Engine:
    """Stateful engine holding variables across steps.

    Replace stub methods with your logic. Ensure images for the UI
    are written into OUTPUT_DIR so the frontend can load them.
    """

    def __init__(self, outputs_dir: Path) -> None:
        self.outputs_dir = outputs_dir
        print(f"Outputs dir: {self.outputs_dir}")
        self.step: int = 0
        self._events = asyncio.Queue()

        self.last_selected_basename = None

    def clear_outputs(self) -> None:
        for old in self.outputs_dir.glob("*"):
            try:
                if old.is_file():
                    old.unlink()
            except Exception:
                pass
        for old in SLOTS_DIR.glob("*"):
            try:
                if old.is_file():
                    old.unlink()
            except Exception:
                pass
        print("Cleared outputs.")

    def start(self) -> None:
        self.step = 0
        # init_images = glob.glob(os.path.join(self.outputs_dir, 'init*.png'))
        # if(len(init_images) < self.num_observations):
        #     Print('Waiting for initiation to finish...')

        print('Starting with initial images...')
        self.clear_outputs()
        # Simple copy example: copy init*.png from sibling repo into outputs
        src_dir = Path(
            '/scratch/ondemand29/chenxil/code/mood-board/search_benchmark/pair_experiments_0704/_s00/')
        if src_dir.exists():
            for p in sorted(src_dir.glob("init*.png")):
                try:
                    shutil.copy2(p, self.outputs_dir / p.name)
                except Exception:
                    pass

    async def next(self, ranking_basenames: list[str], round_id: int | None = None, limit: int | None = None) -> None:
        n = min(limit or len(ranking_basenames), SLOTS)
        if round_id is None:
            round_id = int(asyncio.get_running_loop().time() * 1000)
        await self._events.put(("begin", {"round": round_id, "n": n}))
        for i, name_full in enumerate(ranking_basenames[:n]):
            name = Path(name_full).stem
            data = await asyncio.to_thread(_make_rank_png_bytes, self.step, i, name)
            out_path = SLOTS_DIR / f"slot-{i}.png"
            await asyncio.to_thread(out_path.write_bytes, data)
            await self._events.put(("slot", {"round": round_id, "slot": i}))
        await self._events.put(("done", {"round": round_id}))
        self.step += 1


engine = Engine(OUTPUT_DIR)

app.mount("/static", StaticFiles(directory="."), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


@app.get("/")
def serve_index():
    # Serve index.html from the same folder as this file
    return FileResponse("index.html")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


def _list_image_urls() -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    images = [p for p in OUTPUT_DIR.iterdir() if p.suffix.lower()
              in exts and p.is_file()]
    images.sort()
    return [f"/outputs/{p.name}" for p in images]


@app.get("/api/images")
def images() -> JSONResponse:
    images = _list_image_urls()
    return JSONResponse({"images": images}, headers={
        "Cache-Control": "no-store, no-cache, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    })


@app.post("/api/start")
def start() -> JSONResponse:
    engine.start()

    MIN_COUNT = 1          # how many images make a “batch”
    WAIT_TIMEOUT = 120.0
    deadline = time.monotonic() + WAIT_TIMEOUT
    while time.monotonic() < deadline:
        images = _list_image_urls()
        if len(images) >= MIN_COUNT:
            print(f"Returning {len(images)} images from {OUTPUT_DIR}")
            return JSONResponse({"images": images}, headers={"Cache-Control": "no-store"})
        time.sleep(0.2)  # small sleep to avoid busy-wait

    # timed out (engine failed or took too long)
    return JSONResponse({"status": "pending", "images": []}, status_code=202)


class NextRequest(BaseModel):
    ranking: List[str]
    n: Optional[int] = None


@app.post("/api/next")
async def next_step(req: NextRequest) -> JSONResponse:
    # (optional) clear previous outputs without blocking the loop
    await asyncio.to_thread(engine.clear_outputs)

    # Convert URLs like /outputs/foo.png?v=... -> 'foo.png'
    basenames: list[str] = []
    for url in req.ranking:
        name = url.split("/outputs/")[-1].split("?")[0]
        if name:
            basenames.append(name)

    n = min(req.n or len(basenames) or SLOTS, SLOTS)

    # make a round id usable as a cache-buster
    loop = asyncio.get_running_loop()
    round_id = int(loop.time() * 1000)

    print("Ranking received:", basenames, "n:", n, "round:", round_id)

    # fire-and-forget the async generation; it will emit SSE begin/slot/done
    # ensure your Engine.next accepts (ranking_basenames, round_id, limit)
    asyncio.create_task(engine.next(basenames, round_id=round_id, limit=n))

    # no images list returned anymore—client listens on /api/events
    return JSONResponse({
        "round": round_id,
        "n": n,
        "accepted_ranking": basenames,
        "selected_basename": getattr(engine, "last_selected_basename", None),
    })


@app.get("/api/events")
async def events():
    async def gen():
        while True:
            kind, payload = await engine._events.get()
            yield f"event: {kind}\n"
            yield f"data: {json.dumps(payload)}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
