# server.py
from urllib.parse import urlparse
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
from typing import List, Optional
import asyncio
import json
import time
import os
import yaml

import pytorch_lightning as pl
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np
import uvicorn
from diffusers.utils import load_image
import pySequentialLineSearch

from helper.infer import infer
from engine import obj_sim, infer_image_img2img, prepare_init_obs

CONFIG_FILE = Path(__file__).parent.resolve() / "config.yml"

FRONTEND_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = FRONTEND_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SLOTS_DIR = OUTPUT_DIR / "slots"
SLOTS_DIR.mkdir(parents=True, exist_ok=True)

WORKING_DIR = OUTPUT_DIR / "work"
WORKING_DIR.mkdir(parents=True, exist_ok=True)

DESCRIPTION_DIR = FRONTEND_DIR / "description"
DESCRIPTION_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

# helpers ------------------------------------------------------------


def _make_rank_png_bytes(step: int, i: int, name: str) -> bytes:
    """Sync helper (runs in thread) to build the PNG bytes."""
    time.sleep(3.0)
    img = Image.new("RGB", (200, 200), (220, 240, 255))
    d = ImageDraw.Draw(img)
    d.text((20, 80), f"Step {step} Rank {i+1}: {name}", fill=(0, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def _make_generation(engine, x: np.ndarray):
    sim_val, image_path = engine.f(x)
    # Read image and return bytes
    data = Path(image_path).read_bytes()
    return data, x


class Engine:
    """Stateful engine holding variables across steps.

    Replace stub methods with your logic. Ensure images for the UI
    are written into OUTPUT_DIR so the frontend can load them.
    """

    def __init__(self, config_path: Path, outputs_dir: Path) -> None:
        self.outputs_dir = outputs_dir
        print(f"Outputs dir: {self.outputs_dir}")
        self.step: int = 0
        self._events = asyncio.Queue()

        self.last_selected_basename = None

        # Read config_path (yml)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.num_observations = config.get('num_observations', 10)
        self.init_dir = config.get('init_dir', None)
        self.seed = config.get('seed', 0)
        self.gt_config = config.get('gt_config', '')
        self.max_num_observations = config.get('max_num_observations', 20)

        self.use_sdxl = True
        self.negative_prompt = config.get('negative_prompt', '')
        self.infer_width = config.get('infer_width', 1024)
        self.infer_height = config.get('infer_height', 1024)
        self.infer_steps = config.get('infer_steps', 30)
        self.num_observations_per_step = config.get(
            'num_observations_per_step', 5)

        pl.seed_everything(self.seed)

        # Parse prompt and components
        if '@' in self.gt_config:
            self.prompt, self.gt_config = self.gt_config.split('@')
        self.prompt = self.prompt.replace("'", '')
        self.components = self.gt_config.split(',')
        self.component_weights = []
        for component in self.components:
            comp, weight = component.split(':')
            self.component_weights.append(
                [comp.strip(), float(weight.strip())])

        control_img_path = os.path.join(WORKING_DIR, 'control.png')
        if not os.path.exists(control_img_path):
            # infer an image with baseline model as control
            print(f'control image inference, {self.prompt}')
            random_seed = 194850943985
            images = infer(
                None,  # no lora
                self.prompt,
                ' ',
                random_seed,
                30,
                7,
                self.infer_width,
                self.infer_height,
                use_sdxl=self.use_sdxl,
            )
            self.control_img = np.array(images[0])
            images[0].save(control_img_path)
            print(f'Saved control image to {control_img_path}')
        else:
            self.control_img = np.array(load_image(control_img_path))
            print(f'Loaded control image from {control_img_path}')

        self.x_range = (0.0, 1.0)
        self.gt_image_path = None
        # self.sim_model, self.preprocess = dreamsim(
        #     pretrained=True, device=device)

        # Generate the target image based on the LoRA component weights
        self.weights_str = ','.join(
            [f"{c[1]:.2f}" for c in self.component_weights])
        self.gt_image_path = os.path.join(WORKING_DIR,
                                          f'gt_{self.weights_str}.png')
        if not os.path.exists(self.gt_image_path):
            image_path, self.gt_img = infer_image_img2img(
                self.component_weights,
                self.prompt,
                self.negative_prompt,
                self.infer_steps,
                infer_width=self.infer_width,
                infer_height=self.infer_height,
                image_path=self.gt_image_path,
                control_img=self.control_img)
            # self.gt_img = self.preprocess(gt_img).to(device)
            print(f"Generated GT image: {image_path}")
        else:
            self.gt_img = load_image(self.gt_image_path)  # .to(device)
            print(f"Loaded GT image from {self.gt_image_path}")

        def f(x): return obj_sim(self.gt_image_path, self.component_weights, x, weight_idx=1,
                                 infer_image_func=self.infer_img_func,
                                 output_dir=OUTPUT_DIR, to_vis=True, is_init=False, control_img=self.control_img)

        self.f = f

    def infer_img_func(self, component_weights, image_path=None, control_img=None):
        return infer_image_img2img(component_weights,
                                   self.prompt,
                                   self.negative_prompt,
                                   self.infer_steps,
                                   infer_width=self.infer_width,
                                   infer_height=self.infer_height,
                                   image_path=image_path,
                                   control_img=control_img)

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
        if self.init_dir is not None:
            # Simple copy example: copy init*.png from sibling repo into outputs
            src_dir = Path(self.init_dir)
            # Path('/scratch/ondemand29/chenxil/code/mood-board/search_benchmark/pair_experiments_0704/_s00/')
        else:
            src_dir = WORKING_DIR

        def f_init(x): return obj_sim(self.gt_image_path, self.component_weights, x, weight_idx=1,
                                      infer_image_func=self.infer_img_func,
                                      output_dir=src_dir, to_vis=True, is_init=True, control_img=self.control_img)
        self.f_init = f_init
        x_observations, x_record = prepare_init_obs(self.num_observations, len(
            self.component_weights), self.x_range, self.f_init, seed=self.seed)
        self.x_record = x_record

        if src_dir.exists():
            for p in sorted(src_dir.glob("init*.png")):
                try:
                    shutil.copy2(p, OUTPUT_DIR / p.name)
                except Exception:
                    pass
        self.init_dir = src_dir

    async def next(
        self,
        ranking_basenames: list[str],
        round_id: int | None = None,
        limit: int | None = None,
    ) -> None:
        # pick the user's top choice
        self.last_selected_basename = ranking_basenames[0]
        img_name = Path(self.last_selected_basename).name
        print('X record:', self.x_record)
        assert img_name in self.x_record, f"Image {img_name} not found in record."
        
        self.clear_outputs()

        # --- init / feedback ---
        if self.step == 0:
            pl.seed_everything(self.seed)

            self.path = [self.x_record[img_name][0]]
            self.path[0] = self.path[0].reshape(1, -1)

            def generate_initial_slider(num_dims: int):
                end_0 = self.path[0].flatten()
                end_1 = np.random.uniform(low=0.0, high=1.0, size=(num_dims,))
                return end_0, end_1

            # (sync, short) – okay on the loop
            self.optimizer = pySequentialLineSearch.SequentialLineSearchOptimizer(
                num_dims=self.path[0].shape[1],
                initial_query_generator=generate_initial_slider,
            )
            self.optimizer.set_hyperparams(
                kernel_signal_var=0.50,
                kernel_length_scale=0.10,
                kernel_hyperparams_prior_var=0.10,
            )
        else:
            self.path.append(self.x_record[img_name][0])
            # This may do non-trivial math; offload to a thread so the loop stays responsive.
            print(
                f"Submitting feedback {self.x_record[img_name][1]} for {img_name}")
            self.optimizer.submit_feedback_data(self.x_record[img_name][1])
            # Clear any slot- entries from previous round x_record
            self.x_record = {
                k: v for k, v in self.x_record.items() if not k.startswith("slot-")}

        # --- slider setup ---
        n = min(limit or self.num_observations_per_step,
                self.num_observations_per_step)
        # If this is heavy, you can also offload it:
        # slider_ends = await asyncio.to_thread(self.optimizer.get_slider_ends)
        slider_ends = self.optimizer.get_slider_ends()
        print(f"Slider ends: {slider_ends}")
        x_curr = slider_ends[0]
        direction = slider_ends[1] - slider_ends[0]
        slider_positions = np.linspace(0, 1, n)

        # Do NOT clear outputs here; you overwrite slot files in place to avoid 404s
        # self.clear_outputs()  # <- leave out to keep placeholders visible

        if round_id is None:
            round_id = int(asyncio.get_running_loop().time() * 1000)

        # Tell clients a new round started
        await self._events.put(("begin", {"round": round_id, "n": n}))

        for idx, lmbd in enumerate(slider_positions):
            # Prepare trial point (pure NumPy = sync/fast)
            x_trial = x_curr + lmbd * direction
            print(x_curr)
            x_trial = np.clip(
                x_trial,
                [self.x_range[0]] * x_curr.shape[0],
                [self.x_range[1]] * x_curr.shape[0],
            )

            print(
                f"Round {self.step} generating slot {x_trial} with λ={lmbd:.3f}")

            # Heavy generation (PIL/torch/etc.) – run in worker thread
            # NOTE: pass `self` (engine) into _make_generation; avoid global `engine`
            # <-- await
            data, x = await asyncio.to_thread(_make_generation, self, x_trial)

            # Write PNG bytes to the fixed slot path (file I/O in thread)
            out_path = SLOTS_DIR / f"slot-{idx}.png"
            await asyncio.to_thread(out_path.write_bytes, data)  # <-- await

            # Notify this slot is ready
            # <-- await
            await self._events.put(("slot", {"round": round_id, "slot": idx}))

            # Record mapping for later lookups (tiny critical section)
            out_name = out_path.name
            # If other tasks read/write x_record concurrently, protect with a lock:
            # async with self.x_lock:
            self.x_record[out_name] = (x, float(lmbd))

        # Tell clients the round finished
        await self._events.put(("done", {"round": round_id}))  # <-- await

        self.step += 1

engine = Engine(CONFIG_FILE, OUTPUT_DIR)

app.mount("/static", StaticFiles(directory="."), name="static")
app.mount("/description", StaticFiles(directory=str(DESCRIPTION_DIR)), name="description")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/slots", StaticFiles(directory=str(SLOTS_DIR)), name="slots")


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


# helper: get "foo.png" from any string, ignore blob:


def extract_basename(s: str) -> str | None:
    if not s:
        return None
    # already a bare filename?
    if "/" not in s and any(s.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif")):
        return s.split("?")[0]
    u = urlparse(s)
    if u.scheme == "blob":
        return None
    name = os.path.basename(u.path or "")
    return name.split("?")[0] if name else None


@app.post("/api/next")
async def next_step(req: NextRequest) -> JSONResponse:
    # IMPORTANT: do NOT clear the slots; optional: clear only historical outputs
    # If you want to keep a cleanup, make sure it doesn't touch SLOTS_DIR.
    # await asyncio.to_thread(engine.clear_outputs, keep_slots=True)

    # Normalize ranking -> basenames
    basenames = [b for b in (extract_basename(x) for x in req.ranking) if b]

    # decide how many to generate
    per_step = getattr(engine, "num_observations_per_step",
                       engine.num_observations_per_step)
    n = min(req.n or len(basenames) or per_step, per_step)

    # cache-buster round id
    round_id = int(asyncio.get_running_loop().time() * 1000)

    print("Ranking received:", basenames, "n:", n, "round:", round_id)

    # fire-and-forget generation; it must emit ("begin"/"slot"/"done") to SSE
    asyncio.create_task(engine.next(basenames, round_id=round_id, limit=n))

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
            kind, payload = await engine._events.get()   # must be a 2-tuple
            yield f"event: {kind}\n".encode()
            yield f"data: {json.dumps(payload)}\n\n".encode()
            await asyncio.sleep(0)  # flush
    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache",
                 "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
