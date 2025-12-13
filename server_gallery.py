# server_bo.py
import argparse
import signal
import atexit
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
import copy
import sys
from itertools import accumulate, combinations

import multiprocessing

import pytorch_lightning as pl
import torch
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np
import uvicorn
from diffusers.utils import load_image
import pySequentialLineSearch
import pysps

sys.path.append('/scratch/ondemand29/chenxil/code/mood-board')
from helper.infer import infer
from helper.sampler import sample_dirichlet_simplex
from helper.build_clusters import build_cluster_hierarchy, find_images_for_folders, build_tree
from search_benchmark.multi_solvers import break_clusters_even
from search_benchmark.comparison_solvers import fit_gpytorch_pair_model, pairwise_to_single_task, generate_comparisons, generate_comparisons_index
from engine import (obj_sim, infer_image_img2img,
                    prepare_init_obs, prepare_init_obs_simplex, infer_image)

from async_multi_gpu_pool import MultiGPUInferPool
from serialize import export_engine_state, apply_engine_state, save_engine_state, load_engine_state

CONFIG_FILE = Path(__file__).parent.resolve() / "config_gallery.yml"

FRONTEND_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = FRONTEND_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SLOTS_DIR = OUTPUT_DIR / "slots"
SLOTS_DIR.mkdir(parents=True, exist_ok=True)

WORKING_DIR = OUTPUT_DIR / "work"
WORKING_DIR.mkdir(parents=True, exist_ok=True)

STATE_PATH_OVERRIDE: Path | None = None
STATE_SAVE_PATH_OVERRIDE: Path | None = None

_env_state_override = os.environ.get("ENGINE_STATE_PATH_OVERRIDE")
if _env_state_override:
    try:
        STATE_PATH_OVERRIDE = Path(_env_state_override)
        print(f"[env] Using engine state path override: {STATE_PATH_OVERRIDE}")
    except Exception as exc:
        print(f"[env] Failed to apply ENGINE_STATE_PATH_OVERRIDE: {exc}")

_env_state_save_override = os.environ.get("ENGINE_STATE_SAVE_PATH_OVERRIDE")
if _env_state_save_override:
    try:
        STATE_SAVE_PATH_OVERRIDE = Path(
            _env_state_save_override)
        print(
            f"[env] Using engine save state path override: {STATE_SAVE_PATH_OVERRIDE}")
    except Exception as exc:
        print(
            f"[env] Failed to apply ENGINE_STATE_SAVE_PATH_OVERRIDE: {exc}")

app = FastAPI()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# helpers ------------------------------------------------------------

def ray_box_steps(p, v, low=0.0, high=1.0):
    p = np.asarray(p, dtype=float).flatten()
    v = np.asarray(v, dtype=float).flatten()
    # print(f'p: {p.shape}, v: {v.shape}, low: {low}, high: {high}')
    assert p.shape == v.shape

    # Steps for +v
    t_to_high = np.where(v > 0, (high - p) / v, np.inf)
    t_to_low = np.where(v < 0, (low - p) / v,  np.inf)
    t_plus = np.minimum(t_to_high, t_to_low).min()

    # Steps for -v
    t_to_high_m = np.where(v < 0, (high - p) / (-v), np.inf)
    t_to_low_m = np.where(v > 0, (p - low) / v,    np.inf)
    t_minus = np.minimum(t_to_high_m, t_to_low_m).min()

    return float(-t_minus), float(t_plus)

@app.on_event("startup")
async def _init_engine_once():
    global engine, STATE_PATH_OVERRIDE, STATE_SAVE_PATH_OVERRIDE
    if engine is None:
        engine = Engine(
            CONFIG_FILE,
            OUTPUT_DIR,
            state_path=STATE_PATH_OVERRIDE,
            save_state_path=STATE_SAVE_PATH_OVERRIDE,
        )


@app.on_event("shutdown")
async def _shutdown_engine_pool():
    if engine:
        engine.save_state(reason="app-shutdown")
    _shutdown_gpu_pool(save_state=False)


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
    return data, x, sim_val


def ranking2pairs(ranked_indices):
    # Given a ranked_indices list (e.g., [2,0,1]) from best to worse, generate all comparison pairs indicated by the input indices.
    comp_pairs = []
    # Using combinations
    for i, j in combinations(range(len(ranked_indices)), 2):
        comp_pairs.append(
            (ranked_indices[i], ranked_indices[j])
        )

    return comp_pairs


class Engine:
    """Stateful engine holding variables across steps.

    Replace stub methods with your logic. Ensure images for the UI
    are written into OUTPUT_DIR so the frontend can load them.
    """

    def __init__(self, config_path: Path, outputs_dir: Path, *, state_path: Path | None = None,
                 save_state_path: Path | None = None) -> None:
        self.outputs_dir = outputs_dir
        print(f"Outputs dir: {self.outputs_dir}")

        self.config_path = Path(config_path).resolve()
        self.state_path: Path | None = None
        self.save_state_path: Path | None = None
        self.autosave_enabled = True
        self.autoload_state = True

        self.step: int = 0
        self.cur_step: int = 0
        self._events = asyncio.Queue()

        self.last_selected_basename = None
        self.gpu_pool: MultiGPUInferPool | None = None
        self.worker_state_template: dict | None = None
        self._pool_warmed = False

        self._reset_runtime_state()

        # Read config_path (yml)
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        if not isinstance(config, dict):
            raise ValueError("Config file must be a mapping.")

        self.autosave_enabled = bool(config.get('autosave_enabled', True))
        self.autoload_state = bool(config.get('autoload_state', True))

        load_path = state_path
        if load_path:
            self.state_path = Path(load_path)
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Engine state path: {self.state_path}")

        save_path = save_state_path
        if save_path:
            self.save_state_path = Path(save_path)
            self.save_state_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Engine save state path: {self.save_state_path}")

        self.num_observations = config.get('num_observations', 10)
        self.num_candidates = config.get('num_candidates', 3)
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

        # Support for multi-GPU inference
        # gpu_ids: 0,1
        gpu_ids = config.get('gpu_ids', '')
        gpu_ids = [int(gid.strip())
                   for gid in str(gpu_ids).split(',') if gid.strip().isdigit()]
        if gpu_ids:
            self.gpu_pool = MultiGPUInferPool(
                gpu_ids=gpu_ids,
                module_name="engine_worker")

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

        control_img_path = WORKING_DIR / 'control.png'
        if not control_img_path.exists():
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
            self.control_img = np.array(load_image(str(control_img_path)))
            print(f'Loaded control image from {control_img_path}')
        self.control_img_path = str(control_img_path)

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
        self.worker_state_template = self._make_worker_state_template()

    def infer_img_func(self, component_weights, image_path=None, control_img=None):
        return infer_image_img2img(component_weights,
                                   self.prompt,
                                   self.negative_prompt,
                                   self.infer_steps,
                                   infer_width=self.infer_width,
                                   infer_height=self.infer_height,
                                   image_path=image_path,
                                   control_img=control_img)

    def infer_image_func_packed(self, input_component_weights, input_prompt, image_path):
        return infer_image(
            input_component_weights,
            input_prompt,
            self.negative_prompt,
            self.infer_steps,
            infer_width=self.infer_width,
            infer_height=self.infer_height,
            image_path=image_path)[0]

    def _reset_runtime_state(self) -> None:
        self.stage_index = 0
        self.x_record = {}
        self.x_observations = None
        self.train_X = None
        self.Y = None
        
        self.path = []

        self.train_dataset_history: list[dict] = []
        self.train_dataset_version: int = 0

        self.num_warmup = 0
        self.MC_SAMPLES = 0
        self.last_past_indices = []
        self.ranking_history: list[dict] = []
        self.last_round_context: dict | None = None

    def _make_worker_state_template(self) -> dict:
        return {
            "gt_image_path": self.gt_image_path,
            "component_weights": copy.deepcopy(self.component_weights),
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "infer_steps": self.infer_steps,
            "infer_width": self.infer_width,
            "infer_height": self.infer_height,
            "output_dir": str(self.outputs_dir),
            "control_img_path": self.control_img_path,
            "weight_idx": 1,
        }
    
    def _build_worker_payload(self, x_vector: np.ndarray) -> dict:
        lora_dim = len(self.component_weights)
        w_full = np.zeros(lora_dim)
        # Scale each LoRA by its parent dimension's scalar weight
        w_vec = x_vector.reshape(-1)
        scale = w_vec[np.asarray(self.dim_index_per_lora, dtype=int)]
        w_full[self.lora_merge_indices] = np.asarray(
            self.lora_merge_weights) * scale

        state = copy.deepcopy(self.worker_state_template or {})
        return {
            "state": state,
            "x": x_vector.tolist(),
            "w": w_full.tolist(),
        }

    def _warmup_gpu_pool(self) -> None:
        if not self.gpu_pool or self._pool_warmed:
            return

        print("Warming up GPU inference workers...")
        warm_dir = self.outputs_dir / "warmup"
        warm_dir.mkdir(parents=True, exist_ok=True)

        base_weights = np.full(len(self.component_weights), 1.0 /
                               max(1, len(self.component_weights)), dtype=np.float32)

        pending = set()
        for _ in range(len(self.gpu_pool.procs)):
            # Randomly alter weights slightly per GPU to avoid caching effects
            base_weights_gpu = base_weights + \
                np.random.normal(0, 0.01, size=base_weights.shape)
            base_weights_gpu = np.clip(base_weights_gpu, 0, 1)
            base_weights_gpu = base_weights_gpu / np.sum(base_weights_gpu)

            payload = self._build_worker_payload(base_weights_gpu)
            payload["state"]["output_dir"] = str(warm_dir)
            job_id = self.gpu_pool.submit(
                "worker_make_generation", {"payload": payload})
            pending.add(job_id)

        try:
            while pending:
                job_id, _, err = self.gpu_pool.get_result()
                if job_id not in pending:
                    # could be unrelated task; requeue by storing
                    # For now, ignore since warmup happens before other tasks
                    continue
                pending.remove(job_id)
                if err is not None:
                    raise err
        finally:
            shutil.rmtree(warm_dir, ignore_errors=True)

        self._pool_warmed = True

    async def _finalize_slot(self, slot_idx: int, idx: int, data: bytes, x_vector: np.ndarray, round_id: int, iteration: int, *, emit_events: bool = True):
        out_path = SLOTS_DIR / f"slot-{slot_idx}.png"
        await asyncio.to_thread(out_path.write_bytes, data)
        if emit_events:
            await self._events.put(("slot", {
                "round": round_id,
                "slot": slot_idx,
                "iteration": int(iteration),
            }))
        self.x_record[out_path.name] = (x_vector, int(idx))

    async def _generate_with_pool(self, new_x: torch.Tensor, new_I: List[int], new_y: List[float], round_id: int, iteration: int, *, emit_events: bool = True):
        pending: dict[int, tuple[int, int]] = {}
        for i in range(new_x.shape[0]):
            x_trial = new_x[i]
            if isinstance(x_trial, torch.Tensor):
                x_trial = x_trial.detach().cpu().numpy()
            idx = new_I[i]
            payload = self._build_worker_payload(x_trial)
            job_id = self.gpu_pool.submit(
                "worker_make_generation", {"payload": payload})
            pending[job_id] = (i, idx)

        while pending:
            job_id, result, err = await asyncio.to_thread(self.gpu_pool.get_result)
            slot_idx, idx = pending.pop(job_id)
            if err is not None:
                raise err
            data = result["data"]
            if isinstance(data, memoryview):
                data = data.tobytes()
            x_vec = np.array(result["x"], dtype=np.float64)
            new_y[slot_idx] = float(result["sim_val"])
            await self._finalize_slot(slot_idx, idx, data, x_vec, round_id, iteration, emit_events=emit_events)

    async def _generate_in_process(self, new_x: torch.Tensor, new_I: List[int], new_y: List[float], round_id: int, iteration: int, *, emit_events: bool = True):
        for i in range(new_x.shape[0]):
            x_trial = new_x[i]
            if isinstance(x_trial, torch.Tensor):
                x_trial = x_trial.detach().cpu().numpy()
            idx = new_I[i]
            data, x_vec, y = await asyncio.to_thread(_make_generation, self, x_trial)
            new_y[i] = float(np.asarray(y).flatten()[0])
            await self._finalize_slot(i, idx, data, x_vec, round_id, iteration, emit_events=emit_events)

    async def recall_last_round(self, *, emit_events: bool = False) -> bool:
        ctx = self.last_round_context or {}
        if not ctx:
            print("No last round context available to recall.")
            return False

        new_x_payload = ctx.get("new_x")
        new_I = ctx.get("new_I")
        round_id = ctx.get("round")
        if new_x_payload is None or new_I is None or round_id is None:
            print("Incomplete round context; cannot recall generation.")
            return False

        new_x_tensor = torch.tensor(
            new_x_payload, dtype=torch.double).to(device)
        new_I_list = [int(i) for i in new_I]
        n = int(ctx.get("n", new_x_tensor.shape[0]))

        iteration = ctx.get("iteration")
        if iteration is None:
            base_step = ctx.get("step")
            if base_step is None:
                iteration = int(self.step) if self.step else 0
            else:
                iteration = int(base_step) + 1
        else:
            iteration = int(iteration)

        regen_scores = ctx.get("new_y")
        if isinstance(regen_scores, list) and len(regen_scores) == new_x_tensor.shape[0]:
            new_y = [float(v) for v in regen_scores]
        else:
            new_y = [0.0 for _ in range(new_x_tensor.shape[0])]

        if emit_events:
            await self._events.put(("begin", {
                "round": round_id,
                "n": n,
                "iteration": iteration,
            }))

        if self.gpu_pool:
            await self._generate_with_pool(new_x_tensor, new_I_list, new_y, round_id, iteration)
        else:
            await self._generate_in_process(new_x_tensor, new_I_list, new_y, round_id, iteration)

        if emit_events:
            await self._events.put(("done", {"round": round_id, "iteration": iteration}))

        if self.last_round_context is not None:
            self.last_round_context["new_y"] = [float(val) for val in new_y]

        return True

    def get_gt_image_url(self) -> str | None:
        if not self.gt_image_path:
            return None
        try:
            rel = Path(self.gt_image_path).resolve(
            ).relative_to(self.outputs_dir.resolve())
        except ValueError:
            return None
        # normalize to forward slashes for URLs
        return f"/outputs/{rel.as_posix()}"

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

    async def start(self) -> None:
        """Initialize the engine and generate the first 3x3 grid."""
        pl.seed_everything(self.seed)
        pysps.set_seed(self.seed)

        self.step = 0
        self.cur_step = 0
        self.last_selected_basename = None
        self._reset_runtime_state()

        restored = False
        if self.autoload_state:
            restored = self.load_state()

        self.clear_outputs()

        if restored:
            print(f"Restored engine state from {self.state_path}")
            if self.gpu_pool and not self._pool_warmed:
                await asyncio.to_thread(self._warmup_gpu_pool)
            if self.last_round_context:
                try:
                    await self.recall_last_round()
                except Exception as exc:
                    print(f"Failed to recall last round during start: {exc}")
            else:
                print("No last round context to recall; outputs remain unchanged.")
            return

        print('Starting with initial images...')
        if self.init_dir is not None:
            src_dir = Path(self.init_dir)
        else:
            src_dir = WORKING_DIR
            
        def f_init(x): return obj_sim(self.gt_image_path, self.component_weights, x, weight_idx=1,
                                      infer_image_func=self.infer_img_func,
                                      output_dir=src_dir, to_vis=True, is_init=True, control_img=self.control_img)
        self.f_init = f_init
        x_observations, x_record, best_x = prepare_init_obs(self.num_observations, len(
            self.component_weights), self.x_range, self.f_init, return_best=True, seed=self.seed)
        self.x_record = x_record
        
        self.optimizer = pysps.Optimizer(len(self.component_weights), 
                                          best_x, 
                                          use_map_hyperparams=True)
        
        if src_dir.exists():
            for p in sorted(src_dir.glob("init*.png")):
                try:
                    shutil.copy2(p, OUTPUT_DIR / p.name)
                except Exception:
                    pass
        self.init_dir = src_dir
        os.makedirs(self.init_dir, exist_ok=True)

        print("Warming up GPU pool...")
        if self.gpu_pool and not self._pool_warmed:
            await asyncio.to_thread(self._warmup_gpu_pool)
        
        # retrieve the first search plane for the first UI interaction
        new_x = []
        plane = self.optimizer.retrieve_search_plane()
        radius = int((self.num_candidates - 1) / 2)
        x_range = self.x_range
        x_center = plane.center
        
        for i in range(self.num_candidates):
            for j in range(self.num_candidates):
                cell_index = (i-radius, j-radius)
                x = plane.calc_grid_parameters(cell_index, self.num_candidates, 0.5, [])
                # Check if any element of x is outside [0, 1]
                if x.min() < x_range[0] or x.max() > x_range[1]:
                    x_0 = x.copy()
                    # Make sure x is within the hyper-cube
                    direction = x - x_center
                    _, lambda_high = ray_box_steps(
                        x_center, direction, low=0.0, high=1.0)
                    x = x_center + lambda_high * direction
                    
                    # Clip to [0, 1]
                    x = np.clip(x, 0.0, 1.0)
                new_x.append(x)
        
        new_x = np.array(new_x)
        new_x = torch.from_numpy(new_x).double().to(device)
        new_I = [i for i in range(new_x.shape[0])]
        new_y = [0 for _ in range(new_x.shape[0])]
        
        round_id = int(time.time() * 1000)
        iteration = 0
        
        if self.gpu_pool:
            await self._generate_with_pool(new_x, new_I, new_y, round_id, iteration)
        else:
            await self._generate_in_process(new_x, new_I, new_y, round_id, iteration)
                
        new_y_results = list(new_y)


    def f_to_np(self, x):
        """Convert input tensor to numpy array for the objective function."""
        return torch.from_numpy(self.f(x.detach().cpu().numpy())).to(device=x.device)

    async def next(
        self,
        ranking_basenames: list[str],
        round_id: int | None = None,
        limit: int | None = None,
    ) -> None:
        
        # pick the user's top choice, TODO: need to be updated to a grid interface input 
        self.last_selected_basename = ranking_basenames[0]
        img_name = Path(self.last_selected_basename).name
        print('X record:', self.x_record)
        assert img_name in self.x_record, f"Image {img_name} not found in record."

        self.clear_outputs()
        self.path.append(self.x_record[img_name][0])
        print(
            f"Submitting feedback {self.x_record[img_name][1]} for {img_name}")
        self.optimizer.submit_feedback_data(self.x_record[img_name][1])
        
        
        

engine: Engine | None = None

def _require_engine() -> Engine:
    if engine is None:
        raise RuntimeError("Engine is not initialized in this process.")
    return engine

app.mount("/static", StaticFiles(directory="gallery"), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/slots", StaticFiles(directory=str(SLOTS_DIR)), name="slots")


@app.get("/")
def serve_index():
    # Serve index.html from the same folder as this file
    index_path = FRONTEND_DIR / "gallery" / "index.html"
    if not index_path.exists():
        raise RuntimeError(f"Frontend file missing: {index_path}")
    return FileResponse(str(index_path))

@app.get("/api/health")
def health() -> dict:
    return {"ok": True}

@app.post("/api/state/save")
def api_save_state(payload: dict | None = Body(default=None)) -> JSONResponse:
    eng = _require_engine()
    data = payload or {}
    path = data.get("path")
    reason = data.get("reason") or "api"
    force = bool(data.get("force", False))
    saved_path = eng.save_state(reason=reason, path=path, force=force)
    if not saved_path:
        return JSONResponse({
            "saved": False,
            "path": str(path) if path else None,
        }, status_code=400)
    return JSONResponse({"saved": True, "path": str(saved_path)})


@app.post("/api/state/load")
async def api_load_state(payload: dict | None = Body(default=None)) -> JSONResponse:
    eng = _require_engine()
    data = payload or {}
    path = data.get("path")
    force = bool(data.get("force", False))
    reset = data.get("reset")
    if reset is None:
        reset = True
    loaded = eng.load_state(path=path, reset=bool(reset), force=force)
    if loaded and eng.gpu_pool:
        eng._warmup_gpu_pool()

    recalled = False
    if loaded and data.get("recall"):
        emit_events = bool(data.get("emit_events", False))
        if eng.gpu_pool and not eng._pool_warmed:
            eng._warmup_gpu_pool()
        recalled = await eng.recall_last_round(emit_events=emit_events)

    status = 200 if loaded else 404
    return JSONResponse({
        "loaded": bool(loaded),
        "path": str(path or eng.state_path),
        "recalled": bool(recalled),
    }, status_code=status)


@app.post("/api/state/recall")
async def api_recall_state(payload: dict | None = Body(default=None)) -> JSONResponse:
    eng = _require_engine()
    data = payload or {}
    emit_events = bool(data.get("emit_events", False))
    if eng.gpu_pool and not eng._pool_warmed:
        eng._warmup_gpu_pool()
    recalled = await eng.recall_last_round(emit_events=emit_events)
    status = 200 if recalled else 404
    return JSONResponse({
        "recalled": bool(recalled),
    }, status_code=status)


def _list_image_urls() -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    slots = [p for p in SLOTS_DIR.iterdir()
             if p.suffix.lower() in exts and p.is_file()]
    outputs = [p for p in OUTPUT_DIR.iterdir()
               if p.suffix.lower() in exts and p.is_file()]

    # When restoring state, slot images represent the active round; fall back to
    # the historical outputs directory only if no slot images exist.
    images = slots or outputs
    images.sort()

    base = "/slots" if images is slots else "/outputs"
    return [f"{base}/{p.name}" for p in images]


@app.get("/api/images")
def images() -> JSONResponse:
    images = _list_image_urls()
    return JSONResponse({"images": images}, headers={
        "Cache-Control": "no-store, no-cache, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    })


@app.post("/api/start")
async def start() -> JSONResponse:
    eng = _require_engine()
    await eng.start()
    gt_url = eng.get_gt_image_url()
    iteration = int(getattr(eng, "step", 0))

    MIN_COUNT = 1          # how many images make a “batch”
    WAIT_TIMEOUT = 120.0
    deadline = time.monotonic() + WAIT_TIMEOUT
    while time.monotonic() < deadline:
        images = _list_image_urls()
        if len(images) >= MIN_COUNT:
            print(f"Returning {len(images)} images from {OUTPUT_DIR}")
            return JSONResponse({
                "images": images,
                "gt_image": gt_url,
                "iteration": iteration,
                "stage": eng.stage_status(),
            }, headers={"Cache-Control": "no-store"})
        time.sleep(0.2)  # small sleep to avoid busy-wait

    # timed out (engine failed or took too long)
    return JSONResponse({
        "status": "pending",
        "images": [],
        "gt_image": gt_url,
        "iteration": iteration,
        "stage": eng.stage_status(),
    }, status_code=202)


class NextRequest(BaseModel):
    ranking: List[str]
    n: Optional[int] = None


class StageAdvanceRequest(BaseModel):
    force: bool = False


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
    eng = _require_engine()
    # IMPORTANT: do NOT clear the slots; optional: clear only historical outputs
    # If you want to keep a cleanup, make sure it doesn't touch SLOTS_DIR.
    # await asyncio.to_thread(engine.clear_outputs, keep_slots=True)

    # Normalize ranking -> basenames
    basenames = [b for b in (extract_basename(x) for x in req.ranking) if b]

    # decide how many to generate
    per_step = getattr(eng, "num_observations_per_step",
                       eng.num_observations_per_step)
    per_step += 1
    n = min(req.n or len(basenames) or per_step, per_step)

    # cache-buster round id
    round_id = int(asyncio.get_running_loop().time() * 1000)

    print("Ranking received:", basenames, "n:", n, "round:", round_id)

    # fire-and-forget generation; it must emit ("begin"/"slot"/"done") to SSE
    asyncio.create_task(eng.next(basenames, round_id=round_id, limit=n))

    return JSONResponse({
        "round": round_id,
        "n": n,
        "accepted_ranking": basenames,
        "selected_basename": getattr(eng, "last_selected_basename", None),
    })


@app.get("/api/stage/status")
async def api_stage_status() -> JSONResponse:
    eng = _require_engine()
    return JSONResponse({
        "stage": eng.stage_status(),
        "images": _list_image_urls(),
    })


@app.post("/api/stage/next")
async def api_stage_next(req: StageAdvanceRequest | None = None) -> JSONResponse:
    eng = _require_engine()
    force_flag = bool(getattr(req, "force", False)) if req else False
    advanced = await asyncio.to_thread(eng.next_stage_start, force=force_flag)
    if advanced:
        await eng.emit_stage_update()

    reason = None
    if not advanced:
        if not eng.stage_boundaries:
            reason = "no-stages"
        elif eng.stage_index >= len(eng.stage_boundaries):
            reason = "completed"
        elif not eng.is_stage_ready():
            reason = "not-ready"
        else:
            reason = "blocked"

    payload = {
        "advanced": bool(advanced),
        "stage": eng.stage_status(),
        "images": _list_image_urls(),
    }
    if reason:
        payload["reason"] = reason

    status_code = 200 if advanced else 409
    return JSONResponse(payload, status_code=status_code)


@app.get("/api/events")
async def events():
    eng = _require_engine()

    async def gen():
        while True:
            kind, payload = await eng._events.get()   # must be a 2-tuple
            yield f"event: {kind}\n".encode()
            yield f"data: {json.dumps(payload)}\n\n".encode()
            await asyncio.sleep(0)  # flush
    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache",
                 "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )

# --- Robust GPU pool shutdown handlers (atexit + signals) ---


def _autosave_engine_state(reason: str) -> None:
    if engine is None:
        return
    engine.save_state(reason=reason)


def _shutdown_gpu_pool(from_signal: bool = False, *, save_state: bool = True):
    """Safely shutdown the MultiGPUInferPool if initialized.

    Args:
        from_signal: whether invoked from a signal handler (affects log message).
    """
    if save_state:
        reason = "signal" if from_signal else "shutdown"
        _autosave_engine_state(reason)
    if engine is None:
        return
    pool = engine.gpu_pool
    if pool is not None:
        try:
            pool.shutdown()
            print('[shutdown] MultiGPUInferPool terminated.' +
                  (' (signal)' if from_signal else ''))
        except Exception as e:
            print(f'[shutdown] Failed to terminate GPU pool: {e}')
        finally:
            engine.gpu_pool = None  # prevent double shutdown


def _register_shutdown_handlers():
    # atexit covers normal interpreter termination
    atexit.register(_shutdown_gpu_pool)

    # Handle Ctrl+C and SIGTERM for early interruption
    def _handler(signum, frame):
        print(f'[signal] Caught signal {signum}; shutting down GPU pool...')
        _shutdown_gpu_pool(from_signal=True)
        raise SystemExit(128 + signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except Exception as e:
            print(f'[signal] Could not register handler for {sig}: {e}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive ranking server")
    parser.add_argument("--state-path", dest="state_path", type=str,
                        help="Override path used to load engine state")
    parser.add_argument("--save-state-path", dest="save_state_path", type=str,
                        help="Override path used when saving engine state")
    args = parser.parse_args()

    if args.state_path:
        STATE_PATH_OVERRIDE = Path(args.state_path).expanduser()
        os.environ["ENGINE_STATE_PATH_OVERRIDE"] = str(STATE_PATH_OVERRIDE)
        print(f"[cli] Using engine state path override: {STATE_PATH_OVERRIDE}")

    if args.save_state_path:
        STATE_SAVE_PATH_OVERRIDE = Path(args.save_state_path).expanduser()
        os.environ["ENGINE_STATE_SAVE_PATH_OVERRIDE"] = str(
            STATE_SAVE_PATH_OVERRIDE)
        print(
            f"[cli] Using engine save state path override: {STATE_SAVE_PATH_OVERRIDE}")

    _register_shutdown_handlers()
    # uvicorn.run("server_bo:app", host="127.0.0.1", port=8000, reload=True)
    uvicorn.run("server_bo:app", host="0.0.0.0", port=8000, reload=True)
