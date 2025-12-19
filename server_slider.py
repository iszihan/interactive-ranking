# server_slider.py
import argparse
import signal
import atexit
from urllib.parse import urlparse
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
from typing import List, Optional, Any, Dict
import asyncio
import json
import time
import os
import yaml
import copy
from itertools import combinations

import multiprocessing

import pytorch_lightning as pl
import torch
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import uvicorn
from diffusers.utils import load_image
from PIL import Image

from helper.infer import infer
from engine import (obj_sim, infer_image_img2img, infer_image, check_nsfw_images)
from demographics import Demographics

from async_multi_gpu_pool import MultiGPUInferPool
from serialize import save_slider_state, load_slider_state, apply_slider_engine_state

CONFIG_FILE = Path(__file__).parent.resolve() / "config_slider.yml"

FRONTEND_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = FRONTEND_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SLOTS_DIR = OUTPUT_DIR / "slots"
SLOTS_DIR.mkdir(parents=True, exist_ok=True)

WORKING_DIR = OUTPUT_DIR / "work"
WORKING_DIR.mkdir(parents=True, exist_ok=True)

DESCRIPTION_DIR = FRONTEND_DIR / "description"
DESCRIPTION_DIR.mkdir(parents=True, exist_ok=True)

TUTORIAL_DIR = FRONTEND_DIR / "tutorial"
TUTORIAL_DIR.mkdir(parents=True, exist_ok=True)

DEMO_DIR = OUTPUT_DIR / "demographics"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

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
DEMO_ENABLED: bool = False
DEMO_PARTICIPANT_ID: str | None = None

# Allow demographics to be driven via environment (mirrors state-path handling)
_env_demo_enabled = os.environ.get("DEMOGRAPHIC_ENABLED")
_env_demo_participant = os.environ.get("DEMOGRAPHIC_PARTICIPANT_ID")
if _env_demo_enabled:
    DEMO_ENABLED = str(_env_demo_enabled).lower() in {"1", "true", "yes", "on"}
if _env_demo_participant:
    DEMO_PARTICIPANT_ID = _env_demo_participant
    DEMO_ENABLED = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# helpers ------------------------------------------------------------


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
        _save_slider_checkpoint(engine, reason="app-shutdown")
    _shutdown_gpu_pool(save_state=False)


def _make_generation(engine, x: np.ndarray):
    sim_val, image_path = engine.f(x)
    # Read image and return bytes
    data = Path(image_path).read_bytes()
    return data, x, sim_val


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
        self._events = asyncio.Queue()
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

        self.init_dir = config.get('init_dir', None)
        self.seed = config.get('seed', 0)
        self.gt_config = config.get('gt_config', '')

        self.use_sdxl = True
        self.negative_prompt = config.get('negative_prompt', '')
        example_dir = config.get('example_path')
        self.example_base_dir: Path | None = None
        self._example_cache: dict[str, Path | None] = {}
        if example_dir:
            try:
                resolved = Path(str(example_dir)).expanduser()
                if not resolved.is_absolute():
                    resolved = (self.config_path.parent / resolved).resolve()
                if resolved.exists():
                    self.example_base_dir = resolved
                else:
                    print(f"[config] example_path not found: {resolved}")
            except Exception as exc:
                print(
                    f"[config] Failed to resolve example_path {example_dir}: {exc}")
        self.infer_width = config.get('infer_width', 1024)
        self.infer_height = config.get('infer_height', 1024)
        self.infer_steps = config.get('infer_steps', 30)

        alpha_config = config.get('alpha_config', None)
        self.alpha_config_base = None
        if alpha_config:
            try:
                with open(alpha_config, 'r') as f_cfg:
                    alpha_raw = yaml.safe_load(f_cfg) or {}
            except Exception as exc:
                print(f"Failed to load alpha_config {alpha_config}: {exc}")
                alpha_raw = None
            if isinstance(alpha_raw, dict):
                self.alpha_config_base = alpha_raw.get(
                    'alpha_search', alpha_raw)

        # Support for multi-GPU inference
        # gpu_ids: 0,1
        gpu_ids = config.get('gpu_ids', '')
        gpu_ids = [int(gid.strip())
                   for gid in str(gpu_ids).split(',') if gid.strip().isdigit()]
        # Only keeps the first GPU since we don't run batch inference here
        gpu_ids = [gpu_ids[0]]
        if gpu_ids:
            self.gpu_pool = MultiGPUInferPool(
                gpu_ids=gpu_ids,
                module_name="engine_worker")

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

        self.alpha_context = None
        if self.alpha_config_base:
            self.alpha_context = copy.deepcopy(self.alpha_config_base)
            cache_dir_override = self.alpha_context.get(
                'cache_dir') or 'alpha_cache'
            cache_dir_resolved = str(
                Path(str(cache_dir_override)).expanduser())
            os.makedirs(cache_dir_resolved, exist_ok=True)
            self.alpha_context['cache_dir'] = cache_dir_resolved
            lora_meta = self.alpha_context.setdefault('loras', {})
            for comp in self.component_weights:
                comp_path = comp[0]
                lora_id = Path(comp_path).stem
                meta = lora_meta.setdefault(lora_id, {})
                meta.setdefault('path', str(comp_path))
                if 'image_dir' not in meta:
                    meta['image_dir'] = str(
                        Path(comp_path).with_suffix(''))

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
        self.x_record = {}
        self.last_round_context: dict | None = None
        self.slider_history: list[dict] = []

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

    def resolve_example_image(self, model_id: str) -> Path | None:
        if not model_id or not self.example_base_dir:
            return None
        if model_id in self._example_cache:
            return self._example_cache[model_id]

        base_dir = self.example_base_dir / model_id
        candidates = [
            base_dir / f"ver_{model_id}_000.png",
            base_dir / f"ver_{model_id}_000.jpg",
            base_dir / f"ver_{model_id}_000.jpeg",
            base_dir / f"ver_{model_id}_000.webp",
            base_dir / f"ver_{model_id}_000.gif",
            base_dir / f"ver_{model_id}_000.py",
        ]

        for candidate in candidates:
            if candidate.exists():
                self._example_cache[model_id] = candidate
                return candidate

        self._example_cache[model_id] = None
        return None

    def _record_slider_history(self, payload: dict) -> None:
        if not payload:
            return
        x_vals = payload.get("x")
        if not isinstance(x_vals, list):
            return
        entry = {
            "x": [float(v) for v in x_vals],
            "image": payload.get("image"),
            "similarity": payload.get("similarity"),
            "timestamp": float(payload.get("timestamp") or time.time()),
        }
        self.slider_history.insert(0, entry)
        self.get_slider_iteration()

    def get_slider_iteration(self) -> int:
        """Return the zero-based iteration index inferred from history/state."""
        history_len = len(self.slider_history) if isinstance(self.slider_history, list) else 0
        computed = max(0, history_len - 1) if history_len else 0
        current = int(getattr(self, "step", 0) or 0)
        if computed > current:
            self.step = computed
            current = computed
        return current

    def get_slider_history_payload(self) -> list[dict]:
        history_payload: list[dict] = []
        for entry in self.slider_history:
            x_vals = entry.get("x")
            if not isinstance(x_vals, list):
                continue
            history_payload.append({
                "x": [float(v) for v in x_vals],
                "image": entry.get("image"),
                "similarity": entry.get("similarity"),
                "timestamp": entry.get("timestamp"),
            })
        return history_payload

    def _build_worker_payload(self, x_vector: np.ndarray) -> dict:
        state = copy.deepcopy(self.worker_state_template or {})
        return {
            "state": state,
            "x": x_vector.tolist(),
            "w": x_vector.tolist(),
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

    async def _finalize_slot(self, slot_idx: int, idx: int, data: bytes, x_vector: np.ndarray, round_id: int, iteration: int):
        out_path = SLOTS_DIR / f"slot-{slot_idx}.png"
        await asyncio.to_thread(out_path.write_bytes, data)
        await self._events.put(("slot", {
            "round": round_id,
            "slot": slot_idx,
            "iteration": int(iteration),
        }))
        self.x_record[out_path.name] = (x_vector, int(idx))

    async def _generate_with_pool(self, new_x: torch.Tensor, new_I: List[int], new_y: List[float], round_id: int, iteration: int):
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
            await self._finalize_slot(slot_idx, idx, data, x_vec, round_id, iteration)

    async def _generate_in_process(self, new_x: torch.Tensor, new_I: List[int], new_y: List[float], round_id: int, iteration: int):
        for i in range(new_x.shape[0]):
            x_trial = new_x[i]
            if isinstance(x_trial, torch.Tensor):
                x_trial = x_trial.detach().cpu().numpy()
            idx = new_I[i]
            data, x_vec, y = await asyncio.to_thread(_make_generation, self, x_trial)
            new_y[i] = float(np.asarray(y).flatten()[0])
            await self._finalize_slot(i, idx, data, x_vec, round_id, iteration)

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

    def start(self) -> None:
        self.step = 0
        self._reset_runtime_state()

        restored = False
        if self.autoload_state:
            restored = _load_slider_checkpoint(self)

        # Initialize mcmc cache regardless of restore state
        # get_mcmc_from_cache()

        self.clear_outputs()

        if restored:
            print(f"Restored engine state from {self.state_path}")
            self._warmup_gpu_pool()
            if self.last_round_context:
                try:
                    asyncio.run(self.recall_last_round())
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

        if src_dir.exists():
            for p in sorted(src_dir.glob("init*.png")):
                try:
                    shutil.copy2(p, OUTPUT_DIR / p.name)
                except Exception:
                    pass
        self.init_dir = src_dir
        os.makedirs(self.init_dir, exist_ok=True)

        print("Warming up GPU pool...")
        self._warmup_gpu_pool()

        print("Engine started.")
        _save_slider_checkpoint(self, reason="start")

    def get_slider_metadata(self) -> dict:
        labels = []
        labels: list[str] = []
        thumbnails: list[str | None] = []
        model_ids: list[str] = []
        for idx, comp in enumerate(self.component_weights):
            comp_path = Path(str(comp[0]))
            model_id = comp_path.stem
            labels.append(f"Slider {idx + 1}")
            model_ids.append(model_id)
            example_path = self.resolve_example_image(model_id)
            thumbnails.append(
                f"/api/slider/example/{model_id}" if example_path else None)
        dim = len(labels)
        defaults = [0.0 for _ in range(dim)]
        return {
            "dimension": dim,
            "labels": labels,
            "range": list(self.x_range),
            "default": defaults,
            "thumbnails": thumbnails,
            "model_ids": model_ids,
        }

    def evaluate_slider_vector(self, values: List[float], *, record_history: bool = True,
                               autosave: bool = True) -> dict:
        dim = len(self.component_weights)
        if dim == 0:
            raise ValueError("No components configured for slider evaluation.")
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != dim:
            raise ValueError(
                f"Expected {dim} values, received {arr.shape[0] if arr.ndim == 1 else 'invalid shape'}.")

        low, high = self.x_range
        arr = np.nan_to_num(arr, nan=low, posinf=high, neginf=low)
        arr = np.clip(arr, low, high)

        sim_val, image_path = self.f(arr)
        sim_scalar = float(np.asarray(sim_val).flatten()[0])
        out_url = _relative_output_url(image_path)

        if out_url is None and image_path:
            out_url = str(image_path)

        is_safe: bool | None = None
        if image_path:
            try:
                with Image.open(image_path) as img:
                    result = check_nsfw_images([img.convert("RGB")])
                    if result:
                        is_safe = not bool(result[0])
            except Exception as exc:
                print(f"[safety] slider eval safety check failed: {exc}")
        # default to False (blur) if checker failed to return
        if is_safe is None:
            is_safe = False
            
        print(f'is_safe: {is_safe}')

        payload = {
            "x": arr.tolist(),
            "image": out_url,
            "similarity": sim_scalar,
            "timestamp": time.time(),
            "is_safe": is_safe,
        }
        if record_history:
            self._record_slider_history(payload)
        if autosave:
            _save_slider_checkpoint(self, reason="slider-eval")
        payload["iteration"] = self.get_slider_iteration()
        return payload


engine: Engine | None = None

def _require_engine() -> Engine:
    if engine is None:
        raise RuntimeError("Engine is not initialized in this process.")
    return engine


def _save_slider_checkpoint(eng: Engine, reason: str | None = None, path: Path | str | None = None,
                            *, force: bool = False) -> Path | None:
    target = path or eng.save_state_path
    if target is None:
        print("No save-state path configured; skipping save.")
        return None
    target = Path(target)
    if not force and not eng.autosave_enabled and eng.save_state_path and target == eng.save_state_path:
        print("Autosave disabled; skipping save.")
        return None
    try:
        saved_path = save_slider_state(eng, target, reason=reason)
        print(
            f"Saved engine state to {saved_path} ({reason or 'unspecified'})")
        return saved_path
    except Exception as exc:
        print(f"Failed to save engine state to {target}: {exc}")
        return None


def _load_slider_checkpoint(eng: Engine, *, path: Path | str | None = None,
                            reset: bool = False, force: bool = False) -> bool:
    target = path or eng.state_path
    if target is None:
        print("No state path configured; skipping load.")
        return False
    target = Path(target)
    if not target.exists():
        print(f"State file {target} does not exist.")
        return False
    if reset:
        eng._reset_runtime_state()
    if not force and not eng.autoload_state and eng.state_path and target == eng.state_path:
        print("Autoload disabled; skipping load.")
        return False
    try:
        payload = load_slider_state(target)
        saved_meta = payload.get("metadata", {}) if isinstance(
            payload, dict) else {}
        saved_config = saved_meta.get("config_path")
        if saved_config:
            try:
                saved_path = Path(saved_config).resolve()
                if saved_path != eng.config_path:
                    print(
                        f"Warning: loading state from {saved_path} while current config is {eng.config_path}")
            except Exception:
                pass
        apply_slider_engine_state(eng, payload, torch.device(device))
        print(f"Loaded engine state from {target}")
        return True
    except Exception as exc:
        print(f"Failed to load engine state from {target}: {exc}")
        return False


app.mount(
    "/static", StaticFiles(directory=str(FRONTEND_DIR / "sliders")), name="static")
app.mount("/description", StaticFiles(directory=str(DESCRIPTION_DIR)), name="description")
app.mount("/tutorial", StaticFiles(directory=str(TUTORIAL_DIR)), name="tutorial")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/slots", StaticFiles(directory=str(SLOTS_DIR)), name="slots")


@app.get("/")
def serve_index():
    # Serve index.html from the same folder as this file
    index_path = FRONTEND_DIR / "sliders" / "index.html"
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
    saved_path = _save_slider_checkpoint(
        eng, reason=reason, path=path, force=force)
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
    loaded = _load_slider_checkpoint(
        eng, path=path, reset=bool(reset), force=force)
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


def _list_image_urls() -> List[Dict[str, Any]]:
    exts = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    slots = [p for p in SLOTS_DIR.iterdir() if p.suffix.lower() in exts and p.is_file()]
    outputs = [p for p in OUTPUT_DIR.iterdir() if p.suffix.lower() in exts and p.is_file()]

    # Include both current slots (active round) and historical outputs so the
    # slider UI can resolve safety for latest renders and history entries.
    images = sorted(set(slots + outputs))
    if not images:
        return []

    def base_for(path: Path) -> str:
        return "/slots" if path.parent == SLOTS_DIR else "/outputs"

    safety_map: dict[Path, Optional[bool]] = {}
    eval_paths: list[Path] = []
    pil_batch: list[Image.Image] = []

    for path in images:
        try:
            with Image.open(path) as img:
                pil_batch.append(img.convert("RGB"))
            eval_paths.append(path)
        except Exception as exc:
            print(f"[safety] Failed to load {path}: {exc}")
            safety_map[path] = None

    if pil_batch:
        try:
            nsfw_hits = check_nsfw_images(pil_batch)
            # nsfw_hits = [False] * len(pil_batch)  # Placeholder: assume all safe
            if len(nsfw_hits) != len(eval_paths):
                print(f"[safety] Warning: expected {len(eval_paths)} safety results, got {len(nsfw_hits)}")
            for idx, path in enumerate(eval_paths):
                if idx < len(nsfw_hits):
                    safety_map[path] = not bool(nsfw_hits[idx])
                else:
                    safety_map[path] = None
        except Exception as exc:
            print(f"[safety] NSFW check failed: {exc}")
            for path in eval_paths:
                safety_map[path] = True
        finally:
            for img in pil_batch:
                try:
                    img.close()
                except Exception:
                    pass

    payloads: List[Dict[str, object]] = []

    for path in images:
        state = safety_map.get(path, True)
        payloads.append({
            "url": f"{base_for(path)}/{path.name}",
            "is_safe": state,
            "basename": path.name,
        })

    return payloads


def _relative_output_url(path: str | Path | None) -> str | None:
    if not path:
        return None
    try:
        rel = Path(path).resolve().relative_to(OUTPUT_DIR.resolve())
    except Exception:
        return None
    return f"/outputs/{rel.as_posix()}"


def _guess_media_type(path: Path) -> str:
    ext = path.suffix.lower()
    mapping = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".py": "text/plain",
    }
    return mapping.get(ext, "application/octet-stream")


@app.get("/api/images")
def images() -> JSONResponse:
    images = _list_image_urls()
    return JSONResponse({"images": images}, headers={
        "Cache-Control": "no-store, no-cache, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    })


@app.get("/api/slider/example/{model_id}")
def slider_example(model_id: str):
    eng = _require_engine()
    path = eng.resolve_example_image(model_id)
    if not path:
        return JSONResponse({"error": "Example image not found."}, status_code=404)
    media_type = _guess_media_type(path)
    return FileResponse(str(path), media_type=media_type)


@app.post("/api/start")
def start() -> JSONResponse:
    eng = _require_engine()
    eng.start()
    gt_url = eng.get_gt_image_url()
    slider_meta = eng.get_slider_metadata()
    history_payload = eng.get_slider_history_payload()
    latest_image = None

    if history_payload:
        latest_entry = history_payload[0]
        restored_vector = latest_entry.get("x")
        if isinstance(restored_vector, list) and restored_vector:
            sanitized_vector = [float(v) for v in restored_vector]
            slider_meta["default"] = sanitized_vector
            try:
                restored = eng.evaluate_slider_vector(
                    sanitized_vector,
                    record_history=False,
                    autosave=False,
                )
                latest_image = restored.get("image")
                refreshed_timestamp = restored.get("timestamp")
                refreshed_similarity = restored.get("similarity")
                if latest_image:
                    latest_entry["image"] = latest_image
                if refreshed_timestamp is not None:
                    latest_entry["timestamp"] = refreshed_timestamp
                if refreshed_similarity is not None:
                    latest_entry["similarity"] = refreshed_similarity
                if eng.slider_history:
                    eng.slider_history[0]["image"] = latest_entry.get("image")
                    if refreshed_timestamp is not None:
                        eng.slider_history[0]["timestamp"] = refreshed_timestamp
                    if refreshed_similarity is not None:
                        eng.slider_history[0]["similarity"] = refreshed_similarity
            except Exception as exc:
                print(f"[start] Failed to restore latest slider render: {exc}")

    if latest_image is None:
        zero_vector = list(slider_meta.get("default") or [])
        if slider_meta.get("dimension", 0) > 0 and zero_vector:
            try:
                zero_result = eng.evaluate_slider_vector(zero_vector)
                latest_image = zero_result.get("image")
                slider_meta["default"] = zero_result.get("x", zero_vector)
                history_payload = eng.get_slider_history_payload()
            except Exception as exc:
                print(f"[start] Failed zero-vector render: {exc}")

    if latest_image is None:
        images = _list_image_urls()
        if images:
            latest_image = images[-1]

    iteration = eng.get_slider_iteration()
    return JSONResponse({
        "gt_image": gt_url,
        "iteration": iteration,
        "slider": slider_meta,
        "latest_image": latest_image,
        "history": history_payload,
    }, headers={"Cache-Control": "no-store"})


@app.post("/api/demographics")
def save_demographics(payload: Demographics) -> JSONResponse:
    if not DEMO_ENABLED:
        return JSONResponse({"ok": False, "error": "Demographics disabled"}, status_code=404)
    record = payload.to_storage_dict()
    if DEMO_PARTICIPANT_ID:
        record["participant_id"] = DEMO_PARTICIPANT_ID
    record["server"] = "slider"
    suffix = record.get("participant_id") or record.get("timestamp_ms")
    out_path = DEMO_DIR / f"demo_{suffix}.json"
    try:
        out_path.write_text(json.dumps(record, indent=2))
    except Exception as exc:  # pragma: no cover
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    return JSONResponse({"ok": True})


@app.get("/api/demographics/config")
def demographics_config() -> JSONResponse:
    return JSONResponse({
        "enabled": DEMO_ENABLED,
        "participant_id": DEMO_PARTICIPANT_ID,
    })


class NextRequest(BaseModel):
    ranking: List[str]
    n: Optional[int] = None


class SliderEvalRequest(BaseModel):
    vector: List[float]
    record_history: bool = True


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


@app.post("/api/slider/eval")
async def api_slider_eval(req: SliderEvalRequest) -> JSONResponse:
    eng = _require_engine()
    if eng.gpu_pool and not eng._pool_warmed:
        eng._warmup_gpu_pool()

    try:
        result = await asyncio.to_thread(
            eng.evaluate_slider_vector,
            req.vector,
            record_history=req.record_history,
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

    return JSONResponse(result, headers={"Cache-Control": "no-store"})


@app.get("/api/slider/history")
def api_slider_history() -> JSONResponse:
    eng = _require_engine()
    history = eng.get_slider_history_payload()
    return JSONResponse({"history": history}, headers={"Cache-Control": "no-store"})


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
    _save_slider_checkpoint(engine, reason=reason)


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
    parser.add_argument("--demographic", dest="demographic", type=str,
                        help="Participant ID to enable demographic collection")
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

    if args.demographic:
        DEMO_ENABLED = True
        DEMO_PARTICIPANT_ID = args.demographic
        os.environ["DEMOGRAPHIC_ENABLED"] = "1"
        os.environ["DEMOGRAPHIC_PARTICIPANT_ID"] = DEMO_PARTICIPANT_ID
        print(f"[cli] Demographic collection enabled for participant: {DEMO_PARTICIPANT_ID}")

    _register_shutdown_handlers()
    # uvicorn.run("server_slider:app", host="127.0.0.1", port=8000, reload=True)
    uvicorn.run("server_slider:app", host="0.0.0.0", port=8000, reload=True)
