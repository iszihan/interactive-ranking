# server_bo.py
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
from itertools import combinations

import pytorch_lightning as pl
import torch
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
from helper.sampler import sample_dirichlet_simplex
from helper.build_clusters import build_cluster_hierarchy, find_images_for_folders, build_tree
from search_benchmark.multi_solvers import break_clusters_even
from search_benchmark.comparison_solvers import fit_gpytorch_pair_model, pairwise_to_single_task, generate_comparisons, generate_comparisons_index
from engine import obj_sim, infer_image_img2img, prepare_init_obs, prepare_init_obs_simplex, infer_image

from botorch.acquisition import (
    qUpperConfidenceBound)
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.optim.optimize import gen_batch_initial_conditions
from search_benchmark.acq_prior import (
    qSimplexUpperConfidenceBound,
    stick_breaking_transform, inverse_stick_breaking_transform
)
from helper.sampler import qmc_simplex_generator
from helper.sampler import get_mcmc_from_cache

CONFIG_FILE = Path(__file__).parent.resolve() / "config.yml"

FRONTEND_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = FRONTEND_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SLOTS_DIR = OUTPUT_DIR / "slots"
SLOTS_DIR.mkdir(parents=True, exist_ok=True)

WORKING_DIR = OUTPUT_DIR / "work"
WORKING_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    def infer_image_func_packed(self, input_component_weights, input_prompt, image_path):
        return infer_image(
            input_component_weights,
            input_prompt,
            self.negative_prompt,
            self.infer_steps,
            infer_width=self.infer_width,
            infer_height=self.infer_height,
            image_path=image_path)[0]

    def get_gt_image_url(self) -> str | None:
        if not self.gt_image_path:
            return None
        try:
            rel = Path(self.gt_image_path).resolve().relative_to(self.outputs_dir.resolve())
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
        # x_observations, x_record = prepare_init_obs(self.num_observations, len(
        #     self.component_weights), self.x_range, self.f_init, seed=self.seed)

        if src_dir.exists():
            for p in sorted(src_dir.glob("init*.png")):
                try:
                    shutil.copy2(p, OUTPUT_DIR / p.name)
                except Exception:
                    pass
        self.init_dir = src_dir
        os.makedirs(self.init_dir, exist_ok=True)

        # Initialize mcmc cache
        get_mcmc_from_cache()

        # self.lora2canon_dim, self.dim2loras, self.dim2dims = self.build_tree()
        self.x_observations, x_record = prepare_init_obs_simplex(self.num_observations, len(
            self.component_weights), self.f_init, seed=self.seed, sparse_threshold=0.1, sampler=sample_dirichlet_simplex)
        self.x_record = x_record

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
        
        ranked_indices = [
            self.x_record[Path(img_name).name][1] for img_name in ranking_basenames]
        comp_pairs = ranking2pairs(ranked_indices)
        print(f'ranked_indices: {ranked_indices}')
        
        comp_pairs = torch.tensor(comp_pairs, dtype=torch.long).to(device)
        pairwise_gp = fit_gpytorch_pair_model(self.train_X, comp_pairs)
        gp, latent_Y = pairwise_to_single_task(
            self.train_X, self.Y, pairwise_gp)
         
        print(f'ranked_indices: {ranked_indices}')

        print('Acquiring GP...')
        new_x = self.acquire(gp)
        print(f'Acquired new_x: {new_x}')

        # Estimate the latent Y values for new_x
        pairwise_gp.eval()
        with torch.no_grad():
            # a MultivariateNormal
            post = pairwise_gp.posterior(new_x)
            new_latent_Y = post.mean.squeeze(-1).unsqueeze(-1)  # shape (n,1)
        # Sort new_x based on estimated latent Y values (higher is better)
        ranked_new_indices = torch.argsort(-new_latent_Y, dim=0)  # descending order
        ranked_new_indices = ranked_new_indices.flatten()
        new_x = new_x[ranked_new_indices]

        # Add new_x to train_X
        new_I = [ranked_indices[0]] + \
            [self.train_X.shape[0] + i for i in range(new_x.shape[0])]
        self.train_X = torch.cat([self.train_X, new_x], dim=0)
         
        # Add the current best to the beginning of new_x
        selected_x = self.x_record[img_name][0]
        if isinstance(selected_x, np.ndarray):
            selected_x = torch.from_numpy(selected_x).double().to(device)
        new_x = torch.vstack([selected_x, new_x])

        self.path.append(selected_x)
            
        # --- slider setup ---
        n = min(limit or self.num_observations_per_step + 1,
                self.num_observations_per_step + 1)

        # Do NOT clear outputs here; you overwrite slot files in place to avoid 404s
        # self.clear_outputs()  # <- leave out to keep placeholders visible

        if round_id is None:
            round_id = int(asyncio.get_running_loop().time() * 1000)

        # Tell clients a new round started
        await self._events.put(("begin", {"round": round_id, "n": n}))

        # Iterate and generate each slot given new_x
        new_y = [0 for _ in range(new_x.shape[0])]
        for i in range(new_x.shape[0]):
            # Prepare trial point (pure NumPy = sync/fast)
            x_trial = new_x[i]
            if isinstance(x_trial, torch.Tensor):
                x_trial = x_trial.detach().cpu().numpy()
            idx = new_I[i]

            # Heavy generation (PIL/torch/etc.) – run in worker thread
            # NOTE: pass `self` (engine) into _make_generation; avoid global `engine`
            # <-- await
            data, x, y = await asyncio.to_thread(_make_generation, self, x_trial)
            new_y[i] = y

            # Write PNG bytes to the fixed slot path (file I/O in thread)
            out_path = SLOTS_DIR / f"slot-{i}.png"
            await asyncio.to_thread(out_path.write_bytes, data)  # <-- await

            # Notify this slot is ready
            # <-- await
            await self._events.put(("slot", {"round": round_id, "slot": i}))

            # Record mapping for later lookups (tiny critical section)
            out_name = out_path.name
            # If other tasks read/write x_record concurrently, protect with a lock:
            # async with self.x_lock:
            self.x_record[out_name] = (x, int(idx))

        new_y = new_y[1::]
        self.Y = torch.cat([self.Y, torch.from_numpy(
            np.array(new_y)).double().to(device)], dim=0)

        # Tell clients the round finished
        await self._events.put(("done", {"round": round_id}))  # <-- await

        self.step += 1


engine = Engine(CONFIG_FILE, OUTPUT_DIR)

app.mount("/static", StaticFiles(directory="."), name="static")
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
    gt_url = engine.get_gt_image_url()

    MIN_COUNT = 1          # how many images make a “batch”
    WAIT_TIMEOUT = 120.0
    deadline = time.monotonic() + WAIT_TIMEOUT
    while time.monotonic() < deadline:
        images = _list_image_urls()
        if len(images) >= MIN_COUNT:
            print(f"Returning {len(images)} images from {OUTPUT_DIR}")
            return JSONResponse({"images": images, "gt_image": gt_url}, headers={"Cache-Control": "no-store"})
        time.sleep(0.2)  # small sleep to avoid busy-wait

    # timed out (engine failed or took too long)
    return JSONResponse({"status": "pending", "images": [], "gt_image": gt_url}, status_code=202)


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
    per_step += 1
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
    uvicorn.run("server_bo:app", host="127.0.0.1", port=8000, reload=True)
