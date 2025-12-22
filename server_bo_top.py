# server_bo.py
from serialize import export_engine_state, apply_engine_state, save_engine_state, load_engine_state
from async_multi_gpu_pool import MultiGPUInferPool
from helper.sampler import qmc_simplex_generator, get_mcmc_from_cache
from search_benchmark.acq_prior import (
    qSimplexUpperConfidenceBound,
    stick_breaking_transform, inverse_stick_breaking_transform
)
from botorch.optim.optimize import gen_batch_initial_conditions
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition import (
    qUpperConfidenceBound)
from demographics import Demographics
from engine import (obj_sim, infer_image_img2img,
                    prepare_init_obs, prepare_init_obs_simplex, infer_image,
                    check_nsfw_images)
from search_benchmark.comparison_solvers import fit_gpytorch_pair_model, pairwise_to_single_task, generate_comparisons, generate_comparisons_index
from search_benchmark.multi_solvers import break_clusters_even
from search_benchmark.util import sample_shuffled_unshuffled
from helper.build_clusters import build_cluster_hierarchy, find_images_for_folders, build_tree
from helper.sampler import sample_dirichlet_simplex
from helper.infer import infer
import argparse
import signal
import atexit
from urllib.parse import urlparse
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
from typing import List, Optional, Dict, Any
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
sys.path.append('/scratch/ondemand29/chenxil/code/mood-board')


def _bootstrap_config_override(argv: list[str] | None = None) -> None:
    """Capture --config early so CONFIG_FILE and derived paths honor it."""
    if argv is None:
        argv = sys.argv[1:]
    try:
        idx = argv.index("--config")
        if idx + 1 < len(argv):
            os.environ.setdefault("CONFIG_PATH_OVERRIDE", argv[idx + 1])
    except ValueError:
        return


_bootstrap_config_override()

CONFIG_FILE = Path(__file__).parent.resolve() / "config.yml"
_env_config_override = os.environ.get("CONFIG_PATH_OVERRIDE")
if _env_config_override:
    try:
        CONFIG_FILE = Path(_env_config_override).expanduser().resolve()
        print(f"[env] Using config override: {CONFIG_FILE}")
    except Exception as exc:
        print(f"[env] Failed to apply CONFIG_PATH_OVERRIDE: {exc}")


def _resolve_output_dir(config_path: Path, default_dir: Path) -> Path:
    """Return the output_dir from config if present, else default."""
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
        self.prev_pairwise_gp = None

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
        self.sample_past_n = config.get('sample_past_n', 0)
        self.top_k = config.get('top_k', 5)

        self.use_sdxl = True
        self.negative_prompt = config.get('negative_prompt', '')
        self.infer_width = config.get('infer_width', 1024)
        self.infer_height = config.get('infer_height', 1024)
        self.infer_steps = config.get('infer_steps', 30)
        self.num_observations_per_step = config.get(
            'num_observations_per_step', 5)

        # Stage iterations
        raw_stage_iterations = config.get('stage_iterations', [])
        if isinstance(raw_stage_iterations, list):
            cleaned_stage_iterations: list[int] = []
            for value in raw_stage_iterations:
                try:
                    int_value = int(value)
                except (TypeError, ValueError):
                    continue
                if int_value > 0:
                    cleaned_stage_iterations.append(int_value)
            self.stage_iterations = cleaned_stage_iterations
        else:
            self.stage_iterations = []
        self.stage_boundaries = list(
            accumulate(self.stage_iterations)) if self.stage_iterations else []
        self.stage_index: int = 0

        self.target_dim = config.get('target_dim', 8)
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

        self.worker_state_template = self._make_worker_state_template()
        self.clear_outputs()

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

        # Pairwise comparisons accumulated across rounds
        self.comp_pairs = torch.empty((0, 2), dtype=torch.long, device=device)

        self.train_dataset_history: list[dict] = []
        self.train_dataset_version: int = 0

        self.num_warmup = 0
        self.MC_SAMPLES = 0
        self.last_past_indices = []
        self.ranking_history: list[dict] = []
        self.last_round_context: dict | None = None

    def _archive_current_train_dataset(self, reason: str | None = None) -> None:
        """Persist the active train_X/Y tensors before rebuilding them."""
        if self.train_X is None or self.Y is None:
            return
        snapshot = {
            "version": int(self.train_dataset_version),
            "stage_index": int(self.stage_index),
            "step": int(self.step),
            "reason": reason,
            "train_X": self.train_X.detach().cpu().clone(),
            "Y": self.Y.detach().cpu().clone(),
            "lora_merge_indices": [int(i) for i in getattr(self, "lora_merge_indices", [])],
            "lora_merge_weights": [float(w) for w in getattr(self, "lora_merge_weights", [])],
            "dim_index_per_lora": [int(i) for i in getattr(self, "dim_index_per_lora", [])],
        }
        self.train_dataset_history.append(snapshot)

    def _begin_new_train_dataset_version(self) -> None:
        self.train_dataset_version += 1

    def is_stage_ready(self) -> bool:
        if not self.stage_boundaries:
            return False
        if self.stage_index < 0:
            return False
        if self.stage_index >= len(self.stage_boundaries):
            return False
        target_step = self.stage_boundaries[self.stage_index]
        return int(self.step) >= int(target_step)

    def stage_status(self) -> dict:
        boundaries = list(self.stage_boundaries)
        has_stages = bool(boundaries)
        total_stages = len(boundaries) + 1 if has_stages else 1
        clamped_index = max(0, min(self.stage_index, len(boundaries)))
        current_stage = clamped_index + 1
        ready = bool(
            has_stages and clamped_index < len(boundaries) and int(self.step) >= int(boundaries[clamped_index]))
        next_stage_number = current_stage + \
            1 if clamped_index < len(boundaries) else None
        next_stage_step = boundaries[clamped_index] if clamped_index < len(
            boundaries) else None
        remaining = None
        if next_stage_step is not None:
            remaining = max(0, int(next_stage_step) - int(self.step))
        return {
            "iterations": list(self.stage_iterations),
            "boundaries": boundaries,
            "currentStage": int(current_stage),
            "totalStages": int(total_stages),
            "nextStageReady": bool(ready),
            "nextStageNumber": int(next_stage_number) if next_stage_number is not None else None,
            "nextStageStep": int(next_stage_step) if next_stage_step is not None else None,
            "remainingInStage": int(remaining) if remaining is not None else None,
            "stageIndex": int(clamped_index),
            "step": int(self.step),
            "hasStages": has_stages,
        }

    async def emit_stage_update(self) -> None:
        await self._events.put(("stage", {
            "stage": self.stage_status(),
            "images": _list_image_urls(),
        }))

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

    def start(self) -> None:
        pl.seed_everything(self.seed)

        self.step = 0
        self.cur_step = 0
        self.last_selected_basename = None
        self._reset_runtime_state()

        restored = False
        if self.autoload_state:
            restored = self.load_state()

        # Initialize mcmc cache regardless of restore state
        get_mcmc_from_cache()

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

        self.dim2loras = self.construct_search_space()
        self.update_search_space(src_dir)

        if src_dir.exists():
            for p in sorted(src_dir.glob("init*.png")):
                try:
                    shutil.copy2(p, OUTPUT_DIR / p.name)
                except Exception:
                    pass
        self.init_dir = src_dir
        os.makedirs(self.init_dir, exist_ok=True)

        self.construct_init_samples()

        print("Warming up GPU pool...")
        self._warmup_gpu_pool()

        print("Engine started.")
        self.save_state(reason="start")

        pl.seed_everything(self.seed)

    def next_stage_start(self, *, force: bool = False,
                         ranking_basenames: list[str] | None = None) -> bool:
        if not self.stage_boundaries:
            print('No stage iterations configured; skipping next stage request.')
            return False
        if not force and not self.is_stage_ready():
            print('Next stage not ready yet; skipping advance.')
            return False
        if self.stage_index >= len(self.stage_boundaries):
            print('All configured stages have already been started.')
            return False

        pl.seed_everything(self.seed)

        self._archive_current_train_dataset(
            reason=f"stage-{self.stage_index}-complete")

        if ranking_basenames:
            self.ranking_history.append({
                "event": "stage-advance",
                "stage_index": int(self.stage_index),
                "step": int(self.step),
                "round": None,
                "ranking": list(ranking_basenames),
                "indices": [],
                "selected": getattr(self, "last_selected_basename", None),
                "saved_at": time.time(),
                "train_version": int(self.train_dataset_version),
            })
        self._begin_new_train_dataset_version()

        self.stage_index += 1
        self.cur_step = 0
        self.last_round_context = None

        self.clear_outputs()

        src_dir = WORKING_DIR
        ranked_indices: list[int] = []
        if ranking_basenames:
            for img_name in ranking_basenames:
                rec = self.x_record.get(Path(img_name).name)
                if rec is None:
                    continue
                ranked_indices.append(int(rec[1]))

        best_idx: int | None = ranked_indices[0] if ranked_indices else None
        self.comp_pairs = torch.empty((0, 2), dtype=torch.long, device=device)

        self.dim2loras_before = copy.deepcopy(self.dim2loras)
        self.dim2loras = self.construct_search_space(best_idx)
        self.update_search_space(src_dir)
        seed_context = self.construct_init_samples(
            capture_context=True,
            context_iteration=int(self.step) + 1,
        )

        if seed_context:
            self.last_round_context = seed_context

        pl.seed_everything(self.seed)
        self.save_state(reason=f'stage-{self.stage_index}')

        return True

    def construct_search_space(self, best_idx=None):
        # Look at the current train_X and Y to determine which LoRAs are active
        dim2loras = {
            i: ([Path(c[0]).stem], 0) for i, c in enumerate(self.component_weights)}

        if self.train_X is None:
            return dim2loras

        # Pick the x with the largest y; fall back to best observed if not provided.
        if best_idx is None:
            if self.Y is None:
                return dim2loras
            y_array = self.Y.detach().cpu().numpy().reshape(-1)
            best_idx = int(np.argmax(y_array))

        x_best = self.train_X[best_idx].reshape(-1)

        print(f"x_best: {x_best}")
        active_dims = (x_best > 0).nonzero(as_tuple=True)[0]
        inactive_dims = (x_best == 0).nonzero(as_tuple=True)[0]

        print(
            f"Active dimensions: {active_dims}, Inactive dimensions: {inactive_dims}")

        dim2loras = {d: loras for d, loras in dim2loras.items()
                     if d in active_dims.tolist()}

        # Filter train_X and Y to only keep rows whose non-zero dims match active_dims
        train_X2 = self.train_X[torch.all(
            self.train_X[:, inactive_dims] == 0, dim=1), :]
        train_X2 = train_X2[:, active_dims]
        Y2 = self.Y[torch.all(self.train_X[:, inactive_dims] == 0, dim=1), :]

        sorted_indices = Y2.flatten().argsort(descending=True).squeeze()
        train_X2 = train_X2[sorted_indices, :]
        Y2 = Y2[sorted_indices, :]

        if len(train_X2.shape) == 1:
            train_X2 = train_X2.reshape(1, -1)
        if len(Y2.shape) == 1:
            Y2 = Y2.reshape(1, -1)

        self.train_X = train_X2
        self.Y = Y2

        return dim2loras

    def construct_init_samples(self, *, capture_context: bool = False,
                               context_iteration: int | None = None):
        inf_f = self.f_init if self.train_X is None else self.f
        self.x_observations, x_record = prepare_init_obs_simplex(self.num_observations, len(
            self.dim2loras), inf_f, seed=self.seed, sparse_threshold=0.1, sampler=sample_dirichlet_simplex)
        seed_context: dict | None = None
        context_iteration_value = context_iteration if context_iteration is not None else int(
            self.step)

        # Reuse the existing train_X and Y if available
        if self.train_X is not None and self.Y is not None:
            assert self.train_X.shape[0] == self.Y.shape[0]
            dist_threshold = 0.05
            init_observations2 = [self.train_X[0]]
            init_y2 = [self.Y[0]]
            for i in range(1, self.train_X.shape[0]):
                if len(init_observations2) >= self.max_num_observations/2:
                    break
                obs = self.train_X[i]
                if not any(torch.norm(obs - o) < dist_threshold for o in init_observations2):
                    init_observations2.append(obs)
                    init_y2.append(self.Y[i])
                else:
                    print(
                        f"Skipping observation {obs} as it is too close to existing ones.")

            x_observations2 = self.x_observations[0]
            y_observations2 = self.x_observations[1]
            for i in range(len(x_observations2)):
                sample = torch.from_numpy(
                    x_observations2[i]).double().to(device)
                y = torch.from_numpy(y_observations2[i]).double().to(device)
                if not any(torch.norm(sample - o) < dist_threshold for o in init_observations2):
                    init_observations2.append(sample)
                    init_y2.append(y)
                else:
                    print(
                        f"Skipping observation {sample} as it is too close to existing ones.")

            init_observations2 = torch.stack(init_observations2).double()

            if self.prev_pairwise_gp is not None:
                init_observations2, _ = self.sort_x(
                    self.prev_pairwise_gp, init_observations2, self.dim2loras_before, self.dim2loras)

            self.x_record = {}

            async def _seed_initial_slots() -> dict | None:
                new_y = [0.0 for _ in range(init_observations2.shape[0])]
                new_I = list(range(init_observations2.shape[0]))
                round_id = int(time.time() * 1000)
                iteration = int(context_iteration_value)
                if self.gpu_pool:
                    if not self._pool_warmed:
                        self._warmup_gpu_pool()
                    await self._generate_with_pool(init_observations2, new_I, new_y, round_id, iteration, emit_events=False)
                else:
                    await self._generate_in_process(init_observations2, new_I, new_y, round_id, iteration, emit_events=False)
                return {
                    "round": int(round_id),
                    "n": int(init_observations2.shape[0]),
                    "iteration": int(iteration),
                    "new_x": init_observations2.detach().cpu().tolist(),
                    "new_I": [int(i) for i in new_I],
                    "new_y": [float(val) for val in new_y],
                }

            seed_context = asyncio.run(_seed_initial_slots())
            self.x_observations = (
                np.array(seed_context["new_x"]), np.array(seed_context["new_y"]))

            print(f'In construct_init_samples: {self.x_record}')
        else:
            self.x_record = x_record
        if capture_context and seed_context:
            seed_context.setdefault("ranking", [])
            seed_context.setdefault("timestamp", time.time())
            seed_context.setdefault("step", int(self.step))
            seed_context.setdefault("stage_seed", True)
        else:
            seed_context = None if not capture_context else seed_context

        train_X, Y = self.x_observations
        Y = Y.reshape(-1, 1)
        self.train_X = torch.from_numpy(train_X).double().to(device)
        self.Y = torch.from_numpy(Y).double().to(device)

        return seed_context

    def update_search_space(self, init_src_dir):
        lora2canon_dim = {
            Path(c[0]).stem: i for i, c in enumerate(self.component_weights)}

        lora_dim = len(lora2canon_dim)
        self.lora_merge_indices = []
        self.lora_merge_weights = []
        dim_sorted = sorted(self.dim2loras.keys())
        print(f'lora2canon_dim: {lora2canon_dim}')

        for dim in dim_sorted:
            loras = self.dim2loras[dim][0]
            if len(loras) == 0:
                continue
            # All ones:
            weights = [1.0 for _ in loras]
            print(f'loras: {loras}, weight: {weights}')
            self.lora_merge_indices += [lora2canon_dim[lora] for lora in loras]
            self.lora_merge_weights += weights

        print(f'lora_merge_indices: {self.lora_merge_indices}')
        print(f'lora_merge_weights: {self.lora_merge_weights}')
        # exit()

        # Map each flattened LoRA entry to its dimension index (in dim_sorted order)
        self.dim_index_per_lora: list[int] = []
        for d_idx, dim in enumerate(dim_sorted):
            self.dim_index_per_lora.extend(
                [d_idx] * len(self.dim2loras[dim][0]))

        def f_init_full(x):
            return obj_sim(self.gt_image_path, self.component_weights, x, weight_idx=1,
                           infer_image_func=self.infer_img_func,
                           output_dir=init_src_dir, to_vis=True, is_init=True, control_img=self.control_img)

        def f_init(w):
            w_full = np.zeros(lora_dim)
            # Scale each LoRA by its parent dimension's scalar weight
            w_vec = w.reshape(-1)
            scale = w_vec[np.asarray(self.dim_index_per_lora, dtype=int)]
            w_full[self.lora_merge_indices] = np.asarray(
                self.lora_merge_weights) * scale
            print(
                f'Merging weights: {self.lora_merge_indices} -> {self.lora_merge_weights}')
            print(f'Scaling factors: {scale}')
            print(f'f_init_extended: {w} -> {w_full}')
            # exit()
            return f_init_full(w_full.reshape(1, -1))

        self.f_init = f_init

        def f_full(x): return obj_sim(self.gt_image_path, self.component_weights, x, weight_idx=1,
                                      infer_image_func=self.infer_img_func,
                                      output_dir=OUTPUT_DIR, to_vis=True, is_init=False, control_img=self.control_img)

        def f(w):
            w_full = np.zeros(lora_dim)
            # Scale each LoRA by its parent dimension's scalar weight
            w_vec = w.reshape(-1)
            scale = w_vec[np.asarray(self.dim_index_per_lora, dtype=int)]
            w_full[self.lora_merge_indices] = np.asarray(
                self.lora_merge_weights) * scale
            print(
                f'Merging weights: {self.lora_merge_indices} -> {self.lora_merge_weights}')
            print(f'Scaling factors: {scale}')
            print(f'f_extended: {w} -> {w_full}')
            # exit()
            return f_full(w_full.reshape(1, -1))

        self.f = f

    def build_tree(self):
        # Build LoRA tree
        folders = [str(c[0]).replace('.safetensors', '/')
                   for c in self.component_weights]
        # print(f'folders: {folders}')
        lora_images = find_images_for_folders(folders)
        res = build_cluster_hierarchy(
            lora_images=lora_images,
            # out_dir=str(args.dir),
            out_dir=None,
            pca_dim=16,
            linkage='average',
            # linkage='ward',
            metric='cosine',
            sanity_generate=False,
        )
        lora2canon_dim = {
            Path(c[0]).stem: i for i, c in enumerate(self.component_weights)}
        tree = res.get('tree') if isinstance(res, dict) else None
        Z_link = np.asarray(tree['linkage_matrix'], dtype=float)
        lora_tree = build_tree(Z_link)

        active_loras = [
            ([Path(c[0]).stem for c in self.component_weights], 0)]
        print(f'active_loras: {active_loras}')
        lora_clusters, dim2dims, cluster_counts = break_clusters_even(
            active_loras, self.target_dim, lora2canon_dim, lora_tree,
            alpha_config=self.alpha_context,
            infer_image_func_packed=self.infer_image_func_packed)
        dim2loras = {i: cluster for i,
                     cluster in enumerate(lora_clusters)}

        print(f'dim2loras: {dim2loras}')

        return lora2canon_dim, dim2loras, dim2dims

    def f_to_np(self, x):
        """Convert input tensor to numpy array for the objective function."""
        return torch.from_numpy(self.f(x.detach().cpu().numpy())).to(device=x.device)

    def acquire(self, gp):
        qmc_sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([self.MC_SAMPLES]))
        acq_function = qUpperConfidenceBound(
            model=gp,
            beta=self.beta,
            sampler=qmc_sampler       # reuse your existing QMC sampler
        )
        final_acq_function = qSimplexUpperConfidenceBound(
            model=gp,
            beta=self.beta,
            sampler=qmc_sampler
        )

        eps = 1e-4  # small buffer away from 0/1 in parameter space

        # --- 1. Coefficient-space bounds (sum x <= 1 handled elsewhere) ---
        # x_range is presumably [0.0, 1.0]
        coeff_bounds = torch.tensor(
            [self.x_range] * self.train_X.shape[1]).double().to(device)
        # shape (2, d), row 0 = lower, row 1 = upper
        coeff_bounds = coeff_bounds.T.to(self.train_X)

        # --- 2. Generate initial conditions in *coefficient space* as before ---
        num_restarts = 20
        raw_samples = 1024

        start_time = time.time()
        # Xinit_coeff = gen_batch_initial_conditions_l1(
        #     acq_function,                # this acq is defined on coefficients
        #     coeff_bounds,                # keep full [0,1] here
        #     q=BATCH_SIZE,
        #     num_restarts=num_restarts,
        #     raw_samples=raw_samples,
        # )

        def generator(n, q, seed):
            return qmc_simplex_generator(n, q, self.train_X.shape[1], seed)
        Xinit_coeff = gen_batch_initial_conditions(
            acq_function=acq_function,
            bounds=coeff_bounds,
            q=self.num_observations_per_step,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            generator=generator
        )
        # Inverse stick-breaking: coeffs -> parameter space V in [0,1]
        Xinit = inverse_stick_breaking_transform(Xinit_coeff)

        # Clamp initial conditions away from hard boundaries for stability
        Xinit = Xinit.clamp(min=eps, max=1 - eps)

        end_time = time.time()
        print(
            f"Initial condition generation time: {end_time - start_time:.2f} seconds")

        # --- 3. Parameter-space bounds for L-BFGS (shrunken) ---
        param_bounds = coeff_bounds.clone()
        param_bounds[0, :] = param_bounds[0, :] + \
            eps      # lower bounds += eps
        param_bounds[1, :] = param_bounds[1, :] - \
            eps      # upper bounds -= eps

        start_time = time.time()
        new_x_ei, _ = optimize_acqf(
            acq_function=final_acq_function,  # this one expects parameter-space X
            bounds=param_bounds,              # use shrunken bounds here
            q=self.num_observations_per_step,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            batch_initial_conditions=Xinit,
        )
        end_time = time.time()
        print(
            f"Acquisition function optimization time: {end_time - start_time:.2f} seconds")
        print(f"original new_x_ei: {new_x_ei}")

        # Clamp once more just in case optimizer wandered numerically
        # new_x_ei = new_x_ei.clamp(min=eps_round, max=1 - eps_round)
        eps_round = eps + 1e-6
        new_x_ei[new_x_ei <= eps_round] = 0
        new_x_ei[new_x_ei >= 1 - eps_round] = 1

        # --- 4. Map optimized parameters -> coefficients via stick breaking ---
        new_x_ei = stick_breaking_transform(new_x_ei)

        new_x_ei[new_x_ei <= eps_round] = 0
        new_x_ei[new_x_ei >= 1 - eps_round] = 1
        print(f"new_x_ei: {new_x_ei}")

        return new_x_ei

    def sort_x(self, pairwise_gp, new_x, dim2loras_before=None, dim2loras_after=None):
        new_x_orig = new_x.clone()
        if dim2loras_before is not None and dim2loras_after is not None:
            print(f'[sort_x] dim2loras_before: {dim2loras_before}')
            print(f'[sort_x] dim2loras_after: {dim2loras_after}')
            print(f'[sort_x] new_x: {new_x}')

            # Initialize mapped tensor with same batch size as new_x but old dimension count.
            new_x_orig = torch.zeros(
                new_x.shape[0], len(dim2loras_before), dtype=new_x.dtype, device=new_x.device)

            # Move the new_x after next start to the previous stage space
            dim_sorted_before = sorted(dim2loras_before.keys())
            lora_before2idx = {}
            for idx in range(len(dim_sorted_before)):
                dim = dim_sorted_before[idx]
                loras = dim2loras_before[dim][0]
                for lora in loras:
                    lora_before2idx[lora] = idx

            dim_sorted = sorted(dim2loras_after.keys())
            for i in range(len(dim_sorted)):
                dim = dim_sorted[i]
                loras = dim2loras_after[dim][0]
                for lora in loras:
                    if lora not in lora_before2idx:
                        raise ValueError(
                            f'LoRA {lora} not found in previous stage mapping.')
                    idx = lora_before2idx[lora]
                    # Map the entire column for this dimension into the old space column.
                    new_x_orig[:, idx] = new_x[:, i]

            # Debug output
            print(f'[sort_x] new_x mapped to original space: {new_x_orig}')

        pairwise_gp.eval()
        with torch.no_grad():
            # a MultivariateNormal
            post = pairwise_gp.posterior(new_x_orig)
            new_latent_Y = post.mean.squeeze(-1).unsqueeze(-1)  # shape (n,1)
        # Sort new_x based on estimated latent Y values (higher is better)
        # descending order
        print(f'new_latent_Y: {new_latent_Y}')
        ranked_new_indices = torch.argsort(-new_latent_Y, dim=0)
        ranked_new_indices = ranked_new_indices.flatten()
        new_x = new_x[ranked_new_indices]

        return new_x, ranked_new_indices

    def sample_past(self, latent_Y, comp_pairs, last_best, sample_past_n):
        # Sample some past observations based on the utility values
        past_indices = sample_shuffled_unshuffled(
            latent_Y, comp_pairs, sample_past_n+1, device)

        # Remove last_best if already present
        past_indices = past_indices[past_indices != last_best]
        past_indices = past_indices[:sample_past_n]

        # Add the most current one
        past_indices = torch.cat(
            [past_indices, torch.tensor([last_best], device=device)])

        # Uniqueify the past_indices
        past_indices = torch.unique(past_indices)
        print(f'past_indices: {past_indices}')

        return past_indices

    async def next(
        self,
        ranking_basenames: list[str],
        round_id: int | None = None,
        limit: int | None = None,
        all_basenames: list[str] | None = None,
    ) -> None:
        # pick the user's top choice
        self.last_selected_basename = ranking_basenames[0]
        img_name = Path(self.last_selected_basename).name
        print('X record:', self.x_record)
        assert img_name in self.x_record, f"Image {img_name} not found in record."

        self.clear_outputs()

        # --- init / feedback ---
        if self.cur_step == 0:
            train_X, Y = self.x_observations

            self.num_warmup = train_X.shape[0]
            self.MC_SAMPLES = 256

            self.last_past_indices = []

        ranked_indices = [
            self.x_record[Path(img_name).name][1] for img_name in ranking_basenames]
        additional_indices = [
            self.x_record[Path(img_name).name][1] for img_name in all_basenames if img_name not in ranking_basenames]
        new_comp = ranking2pairs(ranked_indices)

        # Add additional comparisons
        for idx in ranked_indices:
            for add_idx in additional_indices:
                new_comp.append([idx, add_idx])

        new_comp = torch.tensor(new_comp, dtype=torch.long).to(device)
        print(f'new_comp: {new_comp}')
        self.comp_pairs = torch.cat([self.comp_pairs, new_comp], dim=0).long()
        print(f'ranked_indices: {ranked_indices}')

        pairwise_gp = fit_gpytorch_pair_model(self.train_X, self.comp_pairs)
        gp, latent_Y = pairwise_to_single_task(
            self.train_X, self.Y, pairwise_gp)

        print(f'self.train_X: {self.train_X}')
        print(f'self.comp_pairs: {self.comp_pairs}')
        print(f'latent_Y: {latent_Y}')

        print('Acquiring GP...')
        new_x = self.acquire(gp)
        new_batch_size = new_x.shape[0]
        print(f'Acquired new_x: {new_x}')

        # Sample past
        if self.sample_past_n >= 0:
            last_past_indices = self.sample_past(
                latent_Y, self.comp_pairs, last_best=ranked_indices[0], sample_past_n=self.sample_past_n)
            last_past_indices = last_past_indices.tolist()
        else:
            last_past_indices = [ranked_indices[0]]
        print(f'last_past_indices: {last_past_indices}')

        # Estimate the latent Y values for new_x
        new_x, _ = self.sort_x(pairwise_gp, new_x)
        self.prev_pairwise_gp = pairwise_gp

        # Add new_x to train_X
        new_I = last_past_indices + \
            [self.train_X.shape[0] + i for i in range(new_x.shape[0])]
        self.train_X = torch.cat([self.train_X, new_x], dim=0)

        # Re-sort
        print(f'self.train_X: {self.train_X}')
        if self.sample_past_n >= 0:
            new_x = self.train_X[new_I]
            new_x, ranked_new_indices = self.sort_x(pairwise_gp, new_x)
            new_I = [new_I[i] for i in ranked_new_indices]
            print(f're-sorted new_x: {new_x}')
            print(f're-sorted new_I: {new_I}')

        # Add the current best to the path
        selected_x = self.x_record[img_name][0]
        if isinstance(selected_x, np.ndarray):
            selected_x = torch.from_numpy(selected_x).double().to(device)
        self.path.append(selected_x)

        # --- slider setup ---
        # Recompute n after past sampling/re-sorting so frontend placeholder count
        # matches the actual number of candidates being sent.
        effective_total = new_x.shape[0]
        n = effective_total

        # Do NOT clear outputs here; you overwrite slot files in place to avoid 404s
        # self.clear_outputs()  # <- leave out to keep placeholders visible

        if round_id is None:
            round_id = int(asyncio.get_running_loop().time() * 1000)
        iteration = int(self.step) + 1

        self.ranking_history.append({
            "step": int(self.step),
            "round": int(round_id),
            "ranking": list(ranking_basenames),
            "all_basenames": list(all_basenames) if all_basenames else None,
            "indices": [int(x) for x in ranked_indices],
            "selected": self.last_selected_basename,
            "saved_at": time.time(),
            "train_version": int(self.train_dataset_version),
        })

        ctx_new_x = new_x.detach().cpu().tolist()
        self.last_round_context = {
            "step": int(self.step),
            "round": int(round_id),
            "n": int(n),
            "new_x": ctx_new_x,
            "new_I": [int(i) for i in new_I],
            "ranking": list(ranking_basenames),
            "all_basenames": list(all_basenames) if all_basenames else None,
            "timestamp": time.time(),
            "iteration": int(iteration),
        }

        # Tell clients a new round started
        await self._events.put(("begin", {
            "round": round_id,
            "n": n,
            "iteration": int(iteration),
            "stage": self.stage_status(),
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

        self.Y = torch.cat([self.Y, torch.zeros(
            (new_batch_size, 1), dtype=torch.double).to(device)], dim=0)  # for selected
        self.Y[new_I] = torch.from_numpy(
            np.array([float(v) for v in new_y]).reshape(-1, 1)).double().to(device)
        print(f'Updated Y: {self.Y}')

        # Tell clients the round finished
        self.step += 1
        self.cur_step += 1
        self.save_state(reason=f"step-{self.step}")

        await self._events.put(("done", {
            "round": round_id,
            "iteration": int(iteration),
            "stage": self.stage_status(),
        }))


engine: Engine | None = None


def _require_engine() -> Engine:
    if engine is None:
        raise RuntimeError("Engine is not initialized in this process.")
    return engine


app.mount(
    "/static", StaticFiles(directory=str(FRONTEND_DIR / "bo_top")), name="static")
app.mount("/description", StaticFiles(directory=str(DESCRIPTION_DIR)),
          name="description")
app.mount("/tutorial", StaticFiles(directory=str(TUTORIAL_DIR)), name="tutorial")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/slots", StaticFiles(directory=str(SLOTS_DIR)), name="slots")


@app.get("/")
def serve_index():
    # Serve index.html from the same folder as this file
    index_path = FRONTEND_DIR / "bo_top" / "index.html"
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


def _list_image_urls() -> List[Dict[str, Any]]:
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
            nsfw_hits = check_nsfw_images(pil_batch)
            if len(nsfw_hits) != len(eval_paths):
                print(
                    f"[safety] Warning: expected {len(eval_paths)} safety results, got {len(nsfw_hits)}")
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


@app.post("/api/demographics")
def save_demographics(payload: Demographics) -> JSONResponse:
    if not DEMO_ENABLED:
        return JSONResponse({"ok": False, "error": "Demographics disabled"}, status_code=404)
    record = payload.to_storage_dict()
    if DEMO_PARTICIPANT_ID:
        record["participant_id"] = DEMO_PARTICIPANT_ID
    record["server"] = "bo"
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


@app.post("/api/start")
def start() -> JSONResponse:
    eng = _require_engine()
    eng.start()
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
                "top_k": getattr(eng, "top_k", None),
                "stage": eng.stage_status(),
            }, headers={"Cache-Control": "no-store"})
        time.sleep(0.2)  # small sleep to avoid busy-wait

    # timed out (engine failed or took too long)
    return JSONResponse({
        "status": "pending",
        "images": [],
        "gt_image": gt_url,
        "iteration": iteration,
        "top_k": getattr(eng, "top_k", None),
        "stage": eng.stage_status(),
    }, status_code=202)


class NextRequest(BaseModel):
    ranking: List[str]
    all_basenames: Optional[List[str]] = None
    n: Optional[int] = None


class StageAdvanceRequest(BaseModel):
    force: bool = False
    ranking: List[str]


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
    top_k = getattr(eng, "top_k", None)
    if top_k and top_k > 0:
        basenames = basenames[:top_k]

    all_basenames = [b for b in (extract_basename(x)
                                 for x in (req.all_basenames or [])) if b]

    # decide how many to generate
    per_step = getattr(eng, "num_observations_per_step",
                       eng.num_observations_per_step)
    per_step += 1
    n_requested = req.n if req and req.n else None
    n = min(n_requested or per_step, per_step)

    # cache-buster round id
    round_id = int(asyncio.get_running_loop().time() * 1000)

    print("Ranking received:", basenames, "all:",
          all_basenames, "n:", n, "round:", round_id)

    # fire-and-forget generation; it must emit ("begin"/"slot"/"done") to SSE
    asyncio.create_task(eng.next(basenames, round_id=round_id,
                        limit=n, all_basenames=all_basenames))

    return JSONResponse({
        "round": round_id,
        "n": n,
        "accepted_ranking": basenames,
        "all_basenames": all_basenames,
        "selected_basename": getattr(eng, "last_selected_basename", None),
    })


@app.get("/api/stage/status")
async def api_stage_status() -> JSONResponse:
    eng = _require_engine()
    return JSONResponse({
        "stage": eng.stage_status(),
        "images": _list_image_urls(),
        "top_k": getattr(eng, "top_k", None),
    })


@app.post("/api/stage/next")
async def api_stage_next(req: StageAdvanceRequest) -> JSONResponse:
    eng = _require_engine()
    if not req or not req.ranking:
        return JSONResponse({"error": "ranking required"}, status_code=400)

    force_flag = bool(getattr(req, "force", False))
    basenames = [b for b in (extract_basename(x) for x in req.ranking) if b]
    top_k = getattr(eng, "top_k", None)
    if top_k and top_k > 0:
        basenames = basenames[:top_k]
    advanced = await asyncio.to_thread(
        eng.next_stage_start, force=force_flag,
        ranking_basenames=basenames or None)
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
        "top_k": getattr(eng, "top_k", None),
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
    parser.add_argument("--demographic", dest="demographic", type=str,
                        help="Participant ID to enable demographic collection")
    parser.add_argument("--config", dest="config_path", type=str,
                        help="Path to config file override")
    parser.add_argument("--port", dest="port", type=int, default=8000,
                        help="Port to bind the server (default: 8000)")
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
        print(
            f"[cli] Demographic collection enabled for participant: {DEMO_PARTICIPANT_ID}")

    if args.config_path:
        config_override = Path(args.config_path).expanduser()
        os.environ["CONFIG_PATH_OVERRIDE"] = str(config_override)
        CONFIG_FILE = config_override.resolve()
        print(f"[cli] Using config override: {CONFIG_FILE}")

    _register_shutdown_handlers()
    uvicorn.run("server_bo_top:app", host="127.0.0.1",
                port=args.port, reload=False)
