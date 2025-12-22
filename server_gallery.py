# server_bo.py
from serialize import apply_engine_state, save_engine_state, load_engine_state
from async_multi_gpu_pool import MultiGPUInferPool
from engine import (obj_sim, infer_image_img2img,
                    prepare_init_obs_simplex, infer_image, check_nsfw_images)
from demographics import Demographics
from helper.sampler import sample_dirichlet_simplex
from helper.infer import infer
import pysps
from diffusers.utils import load_image
import uvicorn
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi import FastAPI, Body
import torch
import pytorch_lightning as pl
import argparse
import signal
import atexit
from urllib.parse import urlparse
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
from typing import List, Optional, Dict
import asyncio
import json
import time
import os
import yaml
import copy
from itertools import combinations
import sys
sys.path.append('/scratch/ondemand29/chenxil/code/mood-board')


def _bootstrap_config_override(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    try:
        idx = argv.index("--config")
        if idx + 1 < len(argv):
            os.environ.setdefault("CONFIG_PATH_OVERRIDE", argv[idx + 1])
    except ValueError:
        return


_bootstrap_config_override()

CONFIG_FILE = Path(__file__).parent.resolve() / "config_gallery.yml"
_env_config_override = os.environ.get("CONFIG_PATH_OVERRIDE")
if _env_config_override:
    try:
        CONFIG_FILE = Path(_env_config_override).expanduser().resolve()
        print(f"[env] Using config override: {CONFIG_FILE}")
    except Exception as exc:
        print(f"[env] Failed to apply CONFIG_PATH_OVERRIDE: {exc}")


def _resolve_output_dir(config_path: Path, default_dir: Path) -> Path:
    try:
        with open(config_path, "r") as f_cfg:
            cfg = yaml.safe_load(f_cfg) or {}
    except Exception as exc:
        print(f"[config] Failed reading {config_path}: {exc}")
        return default_dir
    if not isinstance(cfg, dict):
        return default_dir
    override = cfg.get("output_dir")
    if not override:
        return default_dir
    try:
        candidate = Path(str(override)).expanduser()
        if not candidate.is_absolute():
            candidate = (config_path.parent / candidate).resolve()
        else:
            candidate = candidate.resolve()
        return candidate
    except Exception as exc:
        print(f"[config] Failed to resolve output_dir {override}: {exc}")
        return default_dir


FRONTEND_DIR = Path(__file__).parent.resolve()
DEFAULT_OUTPUT_DIR = FRONTEND_DIR / "outputs"
OUTPUT_DIR = _resolve_output_dir(CONFIG_FILE, DEFAULT_OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREINIT_DIR = OUTPUT_DIR / "preinit"
PREINIT_DIR.mkdir(parents=True, exist_ok=True)

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

# Allow demographics to be set via environment (mirrors state-path handling)
_env_demo_enabled = os.environ.get("DEMOGRAPHIC_ENABLED")
_env_demo_participant = os.environ.get("DEMOGRAPHIC_PARTICIPANT_ID")
if _env_demo_enabled:
    DEMO_ENABLED = str(_env_demo_enabled).lower() in {"1", "true", "yes", "on"}
if _env_demo_participant:
    DEMO_PARTICIPANT_ID = _env_demo_participant
    DEMO_ENABLED = True

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
        self.clear_outputs()

        self.config_path = Path(config_path).resolve()
        self.config_snapshot: dict | None = None
        self.state_path: Path | None = None
        self.save_state_path: Path | None = None
        self.autosave_enabled = True
        self.autoload_state = True
        self.state_loaded = False

        self.step: int = 0
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
        self.config_snapshot = copy.deepcopy(config)

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

        self.target_dim = config.get('target_dim', 8)
        self.sample_past = config.get('sample_past', 2)
        self.beta = config.get('beta', 9)

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
            prompt = self.prompt
            control_prompt = prompt.replace('drawing', 'photo')
            print(f'control image inference, {control_prompt}')
            random_seed = 194850943985
            images = infer(
                None,  # no lora
                control_prompt,
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
        self.x_observations = None
        self.train_X = None
        self.Y = None
        self.path = []
        self.path_train_X = None
        self.path_Y = None
        self.num_warmup = 0
        self.MC_SAMPLES = 0
        self.I = []
        self.last_past_indices = []
        self.ranking_history: list[dict] = []
        self.last_round_context: dict | None = None
        self.init_ready_timestamp: float | None = None

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

    def save_state(self, reason: str | None = None, path: Path | str | None = None, *, force: bool = False) -> Path | None:
        target = path or self.save_state_path
        if target is None:
            print("No save-state path configured; skipping save.")
            return None
        target = Path(target)
        if not force and not self.autosave_enabled and self.save_state_path and target == self.save_state_path:
            print("Autosave disabled; skipping save.")
            return None
        try:
            saved_path = save_engine_state(self, target, reason=reason)
            print(
                f"Saved engine state to {saved_path} ({reason or 'unspecified'})")
            return saved_path
        except Exception as exc:
            print(f"Failed to save engine state to {target}: {exc}")
            return None

    def load_state(self, path: Path | str | None = None, *, reset: bool = False, force: bool = False) -> bool:
        target = path or self.state_path
        if target is None:
            print("No state path configured; skipping load.")
            return False
        target = Path(target)
        if not target.exists():
            print(f"State file {target} does not exist.")
            return False
        if reset:
            self._reset_runtime_state()
        if not force and not self.autoload_state and target == self.state_path:
            print("Autoload disabled; skipping load.")
            return False
        try:
            payload = load_engine_state(target)
            saved_meta = payload.get("metadata", {}) if isinstance(
                payload, dict) else {}
            saved_config = saved_meta.get("config_path")
            if saved_config:
                try:
                    saved_path = Path(saved_config).resolve()
                    if saved_path != self.config_path:
                        print(
                            f"Warning: loading state from {saved_path} while current config is {self.config_path}"
                        )
                except Exception:
                    pass
            apply_engine_state(self, payload, torch.device(device))
            print(f"Loaded engine state from {target}")
            return True
        except Exception as exc:
            print(f"Failed to load engine state from {target}: {exc}")
            return False

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

    async def start(self) -> None:
        self.step = 0
        self.last_selected_basename = None
        self._reset_runtime_state()

        restored = False
        if self.autoload_state:
            restored = self.load_state()

        # # Initialize mcmc cache regardless of restore state
        # get_mcmc_from_cache()

        self.clear_outputs()

        if restored:
            print(f"Restored engine state from {self.state_path}")
            self._warmup_gpu_pool()
            if self.last_round_context:
                try:
                    await self.recall_last_round()
                except Exception as exc:
                    print(f"Failed to recall last round during start: {exc}")
            else:
                print("No last round context to recall; outputs remain unchanged.")
            return

        print(f'Starting with {self.num_observations} initial images...')
        if self.init_dir is not None:
            src_dir = Path(self.init_dir)
        else:
            src_dir = WORKING_DIR

        def f_preinit(x):
            return obj_sim(self.gt_image_path, self.component_weights, x, weight_idx=1,
                           infer_image_func=self.infer_img_func,
                           output_dir=src_dir / 'preinit', to_vis=True, is_init=True, control_img=self.control_img)

        def f_init(x):
            return obj_sim(self.gt_image_path, self.component_weights, x, weight_idx=1,
                           infer_image_func=self.infer_img_func,
                           output_dir=src_dir, to_vis=True, is_init=True, control_img=self.control_img)

        self.f_init = f_init
        self.f_preinit = f_preinit
        if not os.path.exists(src_dir / 'preinit'):
            os.makedirs(src_dir / 'preinit', exist_ok=True)

        init_observations, x_record = prepare_init_obs_simplex(self.num_observations, len(
            self.component_weights), self.f_preinit, seed=self.seed, sparse_threshold=None, sampler=sample_dirichlet_simplex)

        x_observations_score = init_observations[1]
        best_x = init_observations[0][np.argmax(x_observations_score)]
        self.pysps_optimizer = pysps.Optimizer(len(self.component_weights),
                                               best_x,
                                               use_map_hyperparams=True)
        self.pysps_num_candidates = 3  # TODO: add this as a config var
        self.pysps_radius = int((self.pysps_num_candidates - 1) / 2)
        self.pysps_x_range = [0.0, 1.0]
        self.plane = self.pysps_optimizer.retrieve_search_plane()
        x_center = self.plane.get_center()
        X = []
        for i in range(self.pysps_num_candidates):
            for j in range(self.pysps_num_candidates):
                cell_index = (i - self.pysps_radius, j - self.pysps_radius)
                x = self.plane.calc_grid_parameters(
                    cell_index, self.pysps_num_candidates, 0.5, [])

                # x = (x + 1.0) / 2.0  # normalize to [0, 1]
                # Check if any element of x is outside [0, 1]
                if x.min() < self.pysps_x_range[0] or x.max() > self.pysps_x_range[1]:
                    x_0 = x.copy()
                    # Make sure x is within the hyper-cube
                    direction = x - x_center
                    direction = direction / np.linalg.norm(direction)
                    _, lambda_high = ray_box_steps(
                        x_center, direction, low=0.0, high=1.0)

                    x = x_center + lambda_high * direction

                    # Clip to [0, 1]
                    x = np.clip(x, 0.0, 1.0)
                    print(f'Clipping x: {x_0} (centered at {x_center}) to {x}')
                print(f'Cell index: {cell_index}, x: {x}')
                X.append(x)

        X = torch.tensor(np.array(X), dtype=torch.double).to(device)

        async def prepare_init_pysps_plane(train_X, f):
            """Prepare initial observations for the optimization."""
            yy = []
            round_id = 0
            for i in range(train_X.shape[0]):
                sim_val, image_path = f(
                    train_X[i].reshape(1, -1).detach().cpu().numpy())
                yy.append(sim_val)
                data = Path(image_path).read_bytes()
                await self._finalize_slot(i, i, data, train_X[i].detach().cpu().numpy(), round_id, 0)
                round_id += 1

            Y = torch.tensor(yy).double().reshape(-1, 1)

            return (train_X.detach().cpu().numpy(), Y.detach().cpu().numpy())

        self.x_observations = await prepare_init_pysps_plane(X, self.f_init)
        print("Done generating the initial plane observations.")

        print("Warming up GPU pool...")
        self._warmup_gpu_pool()

        self.init_ready_timestamp = time.time()

        print("Engine started.")
        self.save_state(reason="start")

    def f_to_np(self, x):
        """Convert input tensor to numpy array for the objective function."""
        return torch.from_numpy(self.f(x.detach().cpu().numpy())).to(device=x.device)

    async def next(
        self,
        selection_basenames: list[str],
        round_id: int | None = None,
        limit: int | None = None,
    ) -> None:

        # Pick the user's chosen image (first entry of selection list)
        self.last_selected_basename = selection_basenames[0]
        img_name = Path(self.last_selected_basename).name
        # print('X record:', self.x_record)
        assert img_name in self.x_record, f"Image {img_name} not found in record."
        print(img_name)

        # submit user's choice for pysps optimizer
        x_selected_original = self.x_record[img_name][0]
        self.pysps_optimizer.submit_data(
            x_selected_original, self.plane.get_vertices())

        # Add the current selecton to path
        selected_x = self.x_record[img_name][0]
        if isinstance(selected_x, np.ndarray):
            selected_x = torch.from_numpy(selected_x).double().to(device)
        self.path.append(selected_x)

        self.clear_outputs()

        # retrieve the next search plane based on user feedback

        # --- init / feedback ---
        if self.step == 0:
            pl.seed_everything(self.seed)

            train_X, Y = self.x_observations
            self.train_X = torch.from_numpy(train_X).double().to(device)
            self.Y = torch.from_numpy(Y).double().to(device)

            self.num_warmup = train_X.shape[0]
            self.MC_SAMPLES = 256
            self.path_train_X = self.train_X.clone()
            self.path_Y = self.Y.clone()
            self.path = []
            self.I = [0 for _ in range(self.train_X.shape[0])]

            self.last_past_indices = []

        print(f'self.train_X: {self.train_X}')
        debug_indx = self.x_record[img_name][1]
        print(
            f'Selected: {self.train_X[debug_indx]}, score: {self.Y[debug_indx]}')

        ranked_indices = [
            self.x_record[Path(img_name).name][1] for img_name in selection_basenames]
        print(f'ranked_indices: {ranked_indices}')

        # retrieve the next x on next search plane
        self.plane = self.pysps_optimizer.retrieve_search_plane()
        x_center = self.plane.get_center()
        print(f'x_center: {x_center}')
        new_X = []
        for i in range(self.pysps_num_candidates):
            for j in range(self.pysps_num_candidates):
                cell_index = (i - self.pysps_radius, j - self.pysps_radius)
                x = self.plane.calc_grid_parameters(
                    cell_index, self.pysps_num_candidates, 0.5, [])

                print(f'Cell index: {cell_index}, x before clipping: {x}')

                # x = (x + 1.0) / 2.0  # normalize to [0, 1]
                # Check if any element of x is outside [0, 1]
                if x.min() < self.pysps_x_range[0] or x.max() > self.pysps_x_range[1]:
                    x_0 = x.copy()
                    # Make sure x is within the hyper-cube
                    direction = x - x_center
                    direction = direction / np.linalg.norm(direction)
                    _, lambda_high = ray_box_steps(
                        x_center, direction, low=0.0, high=1.0)

                    x = x_center + lambda_high * direction

                    # Clip to [0, 1]
                    x = np.clip(x, 0.0, 1.0)
                    print(f'Clipping x: {x_0} (centered at {x_center}) to {x}')
                new_X.append(x)

        new_x = torch.tensor(np.array(new_X), dtype=torch.double).to(device)
        new_I = [self.train_X.shape[0] +
                 i for i in range(self.pysps_num_candidates**2)]
        self.train_X = torch.cat([self.train_X, new_x], dim=0)

        # --- slider setup ---
        n = min(limit or self.num_observations_per_step + 1,
                self.num_observations_per_step + 1)

        # Do NOT clear outputs here; you overwrite slot files in place to avoid 404s
        # self.clear_outputs()  # <- leave out to keep placeholders visible

        if round_id is None:
            round_id = int(asyncio.get_running_loop().time() * 1000)
        iteration = int(self.step) + 1

        self.ranking_history.append({
            "step": int(self.step),
            "round": int(round_id),
            # keep field name for backward compat
            "ranking": list(selection_basenames),
            "indices": [int(x) for x in ranked_indices],
            "selected": self.last_selected_basename,
            "saved_at": time.time(),
        })

        ctx_new_x = new_x.detach().cpu().tolist()
        self.last_round_context = {
            "step": int(self.step),
            "round": int(round_id),
            "n": int(n),
            "new_x": ctx_new_x,
            "new_I": [int(i) for i in new_I],
            "selection": list(selection_basenames),
            "timestamp": time.time(),
            "iteration": int(iteration),
        }

        # Tell clients a new round started
        await self._events.put(("begin", {
            "round": round_id,
            "n": n,
            "iteration": int(iteration),
        }))

        # Iterate and generate each slot given new_x
        new_y = [0 for _ in range(new_x.shape[0])]
        if self.gpu_pool:
            await self._generate_with_pool(new_x, new_I, new_y, round_id, iteration)
        else:
            await self._generate_in_process(new_x, new_I, new_y, round_id, iteration)
        new_y_results = list(new_y)
        if self.last_round_context is not None:
            self.last_round_context["new_y"] = [
                float(val) for val in new_y_results]

        feedback_values = new_y_results[1::]
        self.Y = torch.cat([self.Y, torch.from_numpy(
            np.array(feedback_values).reshape(-1, 1)).double().to(device)], dim=0)

        # Tell clients the round finished
        await self._events.put(("done", {
            "round": round_id,
            "iteration": int(iteration),
        }))  # <-- await

        self.step += 1
        self.save_state(reason=f"step-{self.step}")


engine: Engine | None = None


def _require_engine() -> Engine:
    if engine is None:
        raise RuntimeError("Engine is not initialized in this process.")
    return engine


app.mount("/static", StaticFiles(directory="gallery"), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/slots", StaticFiles(directory=str(SLOTS_DIR)), name="slots")
app.mount("/description", StaticFiles(directory=str(DESCRIPTION_DIR)),
          name="description")
app.mount("/tutorial", StaticFiles(directory=str(TUTORIAL_DIR)), name="tutorial")


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


@app.get("/api/stage/status")
async def api_stage_status() -> JSONResponse:
    # Gallery workflow does not use stages; return empty stage state for parity.
    return JSONResponse({
        "stage": None,
        "images": _list_image_urls(),
    })


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


def _list_image_urls() -> List[Dict[str, object]]:
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
    if not images:
        return []

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
            print(f'[safety] Checking NSFW for {eval_paths}...')
            nsfw_hits = check_nsfw_images(pil_batch)
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
            "url": f"{base}/{path.name}",
            "is_safe": state,
            "basename": path.name,
        })

    return payloads


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
            }, headers={"Cache-Control": "no-store"})
        time.sleep(0.2)  # small sleep to avoid busy-wait

    # timed out (engine failed or took too long)
    return JSONResponse({
        "status": "pending",
        "images": [],
        "gt_image": gt_url,
        "iteration": iteration,
    }, status_code=202)


@app.post("/api/demographics")
def save_demographics(payload: Demographics) -> JSONResponse:
    if not DEMO_ENABLED:
        return JSONResponse({"ok": False, "error": "Demographics disabled"}, status_code=404)
    record = payload.to_storage_dict()
    if DEMO_PARTICIPANT_ID:
        record["participant_id"] = DEMO_PARTICIPANT_ID
    record["server"] = "gallery"
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
    selection: Optional[str] = None  # single chosen image (preferred)
    ranking: Optional[List[str]] = None  # backward-compatible
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
    eng = _require_engine()

    # Normalize selection (preferred) or fall back to ranking[0]
    chosen = None
    if req.selection:
        chosen = extract_basename(req.selection)
    if not chosen and req.ranking:
        # use the first entry as the chosen one
        for x in req.ranking:
            b = extract_basename(x)
            if b:
                chosen = b
                break

    if not chosen:
        return JSONResponse(
            {"error": "No selection provided"},
            status_code=400,
        )

    basenames = [chosen]

    # decide how many to generate
    per_step = getattr(eng, "num_observations_per_step",
                       eng.num_observations_per_step)
    per_step += 1
    n = min(req.n or per_step, per_step)

    # cache-buster round id
    round_id = int(asyncio.get_running_loop().time() * 1000)

    print("Selection received:", basenames, "n:", n, "round:", round_id)

    # fire-and-forget generation; it must emit ("begin"/"slot"/"done") to SSE
    asyncio.create_task(eng.next(basenames, round_id=round_id, limit=n))

    return JSONResponse({
        "round": round_id,
        "n": n,
        "accepted_selection": chosen,
        "selected_basename": getattr(eng, "last_selected_basename", None),
    })


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
    parser.add_argument("--demographic", dest="demographic", type=str,
                        help="Participant ID to enable demographic collection")
    parser.add_argument("--config", dest="config_path", type=str,
                        help="Path to config file override")
    parser.add_argument("--port", dest="port", type=int, default=8000,
                        help="Port to bind the server (default: 8000)")
    parser.add_argument("--ssh", action="store_true",
                        help="Enable SSH tunneling (not implemented)")
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

    if args.config_path:
        config_override = Path(args.config_path).expanduser()
        os.environ["CONFIG_PATH_OVERRIDE"] = str(config_override)
        CONFIG_FILE = config_override.resolve()
        print(f"[cli] Using config override: {CONFIG_FILE}")

    _register_shutdown_handlers()
    host = "127.0.0.1" if not args.ssh else "0.0.0.0"
    uvicorn.run("server_gallery:app", host=host, port=args.port, reload=False)
