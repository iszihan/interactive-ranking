from __future__ import annotations

import os
import shlex
import subprocess
from PIL import Image
from pathlib import Path
import shutil
import hashlib
import copy 
from typing import List
import math
from diffusers.utils import load_image
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
import yaml
import sys
import glob
import pySequentialLineSearch
import pytorch_lightning as pl
from dreamsim import dreamsim
sys.path.append('/scratch/year/zling/projects/lora-moodboard/mood-board')
from helper.infer import infer, temporary_directory, infer_img2img, infer_guidance
sys.path.append('/scratch/year/zling/projects/lora-moodboard/mood-board/search_benchmark')
from solvers import (prepare_init_obs,
                     random_direction_solve, cyclic_coordinate_descent_solve, random_coordinate_descent_solve,
                     bayesian_botorch_solve, bayesian_saasbo_solve, bayesian_saasbo_extra_solve, bayesian_saasbo_batch_solve,
                     multistage_solve, koyama_line_solve, koyama_plane_solve)
from benchmark_botorch import load_img_npz, save_img_npz, save_sim_val_npz, get_x_hash, load_sim_val_npz, find_triggers
import torch

FRONTEND_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = FRONTEND_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = '/scratch/year/zling/projects/lora-moodboard/mood-board/search_benchmark/pair_experiments/small_benchmark_0704_p1.yml'
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
infer_width, infer_height = 1024, 1024
app = FastAPI(title="Interactive Ranking Backend")

def infer_image_img2img(component_weights, 
                        prompt,
                        negative_prompt,
                        image_path=None, 
                        control_img=None,
                        steps=15,
                        ):
    # global steps, negative_prompt, prompt, use_sdxl

    lora_files = [file for file, _ in component_weights]
    weights = [weight for _, weight in component_weights]
    triggers = find_triggers(lora_files, weights)
    loras = []
    for i, lora_file in enumerate(lora_files):
        if weights[i] == 0:
            continue
        loras.append((lora_file, math.sqrt(weights[i])))

    if image_path != None:
        if os.path.exists(image_path):
            print(f"Image {image_path} already exists, skipping inference.")
            return image_path, Image.open(image_path)

    # Example: generate or load your image as a numpy array
    triggers_str = ', '.join(triggers)
    positive_prompt = f'{triggers_str}, {prompt}'
    seed = 184759827843959
    cfg = 7
    img2img_denoise = 0.8
    
    images = infer_img2img(loras, positive_prompt, negative_prompt,
                           seed, steps, cfg, infer_width, infer_height,
                           use_sdxl=True, image=control_img, img2img_denoise=img2img_denoise)
    # Convert to base64
    im = images[0]
    if image_path != None:
        im.save(image_path)
    # Convert im to PIL Image
    im = Image.fromarray(np.array(im))
    return image_path, im

class Engine:
    """Stateful engine holding variables across steps.

    Replace stub methods with your logic. Ensure images for the UI
    are written into OUTPUT_DIR so the frontend can load them.
    """

    def __init__(self, outputs_dir: Path) -> None:
        self.outputs_dir = outputs_dir
        print(f"Outputs dir: {self.outputs_dir}")
        self.step: int = 0
        
        self.last_selected_basename = None
        
        with open(CONFIG_PATH, 'r') as file:
            self.gt_configs = yaml.safe_load(file)[0]
        self.max_num_observations = 20
        self.use_sdxl = True
        self.seed = 10
        self.negative_prompt = "easynegative, 3d, realistic, badhandv4, (lowres,  bad quality),  (worst quality,  low quality:1.3),  blurry,  cropped,  out of frame,  border,  bad hands,  interlocked fingers,  mutated hands,  (Bad anatomy:1.4),  from afar,  warped,  (deformed,  disfigured:1.1),  twisted torso,  mutated limbs"
        print(f"Starting experiment with seed {self.seed} with config {self.gt_configs}")
        
        if '@' in self.gt_configs:
            self.prompt, self.gt_config = self.gt_configs.split('@')
        self.prompt = self.prompt.replace("'", '')
        self.components = self.gt_config.split(',')
        self.component_weights = []
        for component in self.components:
                comp, weight = component.split(':')
                self.component_weights.append(
                    [comp.strip(), float(weight.strip())])
        control_img_path = os.path.join(os.path.dirname(CONFIG_PATH), 'control.png')
        if not os.path.exists(control_img_path):
            # infer an image with baseline model as control 
            print(f'control image inference, {self.prompt}')
            random_seed = 194850943985
            images = infer(
                None, # no lora 
                self.prompt,
                ' ',
                random_seed, 
                30, 
                7, 
                self.infer_width, 
                self.infer_height,
                use_sdxl=True,
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
        self.gt_image_path = os.path.join(
                    os.path.dirname(CONFIG_PATH),
                    f'gt_{self.weights_str}.png')
        if not os.path.exists(self.gt_image_path):
            image_path, self.gt_img = infer_image_img2img(
                            self.component_weights, 
                            self.prompt,
                            self.negative_prompt,
                            image_path=self.gt_image_path, 
                            control_img=self.control_img)
            # self.gt_img = self.preprocess(gt_img).to(device)
            print(f"Generated GT image: {image_path}")
        else:
            self.gt_img = load_image(self.gt_image_path) #.to(device)
            print(f"Loaded GT image from {self.gt_image_path}")
            
        self.infer_img_func = infer_image_img2img
                        
        # def f_init(x): return self.obj_sim(self.gt_img, 
        #                                    self.component_weights, 
        #                                    x, 
        #                                    weight_idx=1,
        #                                    infer_image_func=self.infer_img_func,
        #                                    output_dir=self.outputs_dir, 
        #                                    to_vis=True, 
        #                                    is_init=True, 
        #                                    control_img=self.control_img)
        # # sample initial observations
        self.num_observations = 10
        # self.observations, self.start_obs = prepare_init_obs(
        #         self.num_observations, 
        #         len(self.component_weights), 
        #         self.x_range, f_init,
        #         seed=self.seed)
        # exit() 
        pl.seed_everything(self.seed)
        self.optimizer = pySequentialLineSearch.SequentialLineSearchOptimizer(
            num_dims=5)
        self.optimizer.set_hyperparams(kernel_signal_var=0.50,
                                        kernel_length_scale=0.10,
                                           kernel_hyperparams_prior_var=0.10)
        
    def clear_outputs(self) -> None:
        for old in self.outputs_dir.glob("*"):
            try:
                if old.is_file():
                    old.unlink()
            except Exception:
                pass

    def start(self) -> None:
        self.step = 0
        # init_images = glob.glob(os.path.join(self.outputs_dir, 'init*.png'))
        # if(len(init_images) < self.num_observations):
        #     Print('Waiting for initiation to finish...')
        
        # self.clear_outputs()
        # # Simple copy example: copy init*.png from sibling repo into outputs
        # src_dir = (FRONTEND_DIR / ".." / "lora-moodboard" / "mood-board" / "search_benchmark" /
        #            "pair_experiments" / "interactive_test_run" / "_s00").resolve()
        # if src_dir.exists():
        #     for p in sorted(src_dir.glob("init*.png")):
        #         try:
        #             shutil.copy2(p, self.outputs_dir / p.name)
        #         except Exception:
        #             pass

    def next(self, ranking_basenames: List[str]) -> None:
       # Capture left-most (best) image selection from UI ranking
        if ranking_basenames and self.step > 0:
            self.last_selected_basename = ranking_basenames[0]
            # parse the image name to get the chosen slider position 
            selected_lmdb = self.last_selected_basename.split('_')[-1][:-4].split('=')[-1]
            selected_lmdb = float(selected_lmdb)
            self.optimizer.submit_feedback_data(selected_lmdb)
            print(f"Selected image with slider position: {selected_lmdb}")
        slider_ends = self.optimizer.get_slider_ends()    
        # infer images for slider and ask the user the find the best one
        x_curr = slider_ends[0]
        direction = slider_ends[1] - slider_ends[0]
        direction_norm = np.linalg.norm(direction)
        direction_normalized = direction / direction_norm
        lambda_bounds = [0, direction_norm]
        slider_positions = np.linspace(0, direction_norm, 10)
        # Do not clear outputs to avoid 404s in the UI while generation runs
        self.clear_outputs()
        # infer new images 
        for idx, lmbd in enumerate(slider_positions):
            x_trial = x_curr + lmbd * direction_normalized
            x_trial = np.clip(x_trial, [0,0,0,0,0], [1,1,1,1,1])
            self.infer_image(x_trial, 
                             self.component_weights, 
                             1, 
                             self.infer_img_func,
                             output_dir=self.outputs_dir, 
                             to_vis=True, 
                             is_init=False, 
                             control_img=self.control_img,
                             attrs={'lmbd': float(lmbd)})
        self.step += 1
        
    def infer_image(self, x, component_weights, 
                    weight_idx, infer_image_func,
                    output_dir=None, to_vis=False, 
                    is_init=False, to_cache=True, 
                    control_img=None, attrs=None):
    
        print(f"Evaluating with weights: {x}")
        attrs = attrs or {}
        parts = []
        for k, v in attrs.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.6f}")
            else:
                parts.append(f"{k}={v}")
        img_name_attrs_str = '_'.join(parts)

        infer_component_weights = copy.deepcopy(component_weights)
        for i in range(len(infer_component_weights)):
            infer_component_weights[i][weight_idx] = x.flatten()[i]
        weights_str = ','.join(
            [f"{c[weight_idx]:.2f}" for c in infer_component_weights])
       
        # -- check cache
        cache_dir = None
        if output_dir is not None and to_cache:
            cache_dir = os.path.join(output_dir, 'cache')
            os.makedirs(cache_dir, exist_ok=True)
            x_hash = get_x_hash(x)
            cache_path = os.path.join(cache_dir, f"{x_hash}.npz")
            if os.path.exists(cache_path):
                img = load_img_npz(cache_path)
                print(f"loaded cached img from: {cache_path}")
                img = Image.fromarray(img)
                img.save(os.path.join(output_dir, f'img_{weights_str}_{img_name_attrs_str}.png'))

        img = None
        if is_init:
            image_path = os.path.join(
                output_dir, f'init_{weights_str}.png')
            if os.path.exists(image_path):
                print(f"image {image_path} already exists, skipping inference.")
                img = image.open(image_path)
        
        if img is None:
            if control_img is None:
                _, img = infer_image_func(infer_component_weights, 
                                          self.prompt,
                                          self.negative_prompt,
                                          image_path=None)
            else:
                _, img = infer_image_func(infer_component_weights, 
                                          self.prompt,
                                          self.negative_prompt,
                                          image_path=None, 
                                          control_img=self.control_img)
        if not is_init:
            image_path = os.path.join(
                output_dir, f'img_{weights_str}_{img_name_attrs_str}.png')
        if output_dir is not None and to_vis and not os.path.exists(image_path):
            img.save(image_path)
            print(f"saved image to: {image_path}")
        else:
            print(f'not saved: {image_path}')
        # -- save to cache
        if cache_dir is not None:
            save_img_npz(cache_path, x, np.array(img))
            print(f"saved result to cache: {cache_path}")
    
    def obj_sim(self, gt_img, component_weights, x, weight_idx, infer_image_func,
                output_dir=None, to_vis=False, is_init=False, to_cache=True, control_img=None):
    
        print(f"Evaluating with weights: {x}")

        # -- Check cache
        cache_dir = None
        if output_dir is not None and to_cache:
            cache_dir = os.path.join(output_dir, 'cache')
            os.makedirs(cache_dir, exist_ok=True)
            x_hash = get_x_hash(x)
            cache_path = os.path.join(cache_dir, f"{x_hash}.npz")
            if os.path.exists(cache_path):
                sim_val = load_sim_val_npz(cache_path)
                print(f"Loaded cached result {sim_val} from: {cache_path}")
                return sim_val

        if self.sim_model is None:
            self.sim_model, self.preprocess = dreamsim(pretrained=True, device=device)
        infer_component_weights = copy.deepcopy(component_weights)
        for i in range(len(infer_component_weights)):
            infer_component_weights[i][weight_idx] = x.flatten()[i]
        weights_str = ','.join(
            [f"{c[weight_idx]:.2f}" for c in infer_component_weights])
        img = None
        if is_init:
            image_path = os.path.join(
                output_dir, f'init_{weights_str}.png')
            if os.path.exists(image_path):
                print(f"Image {image_path} already exists, skipping inference.")
                img = Image.open(image_path)
        if img is None:
            if control_img is None:
                _, img = infer_image_func(infer_component_weights, 
                                          self.prompt,
                                          self.negative_prompt,
                                          image_path=None)
            else:
                _, img = infer_image_func(infer_component_weights, 
                                          self.prompt,
                                          self.negative_prompt,
                                          image_path=None, 
                                          control_img=self.control_img)
        img_preprocessed = self.preprocess(img).to(device)
        distance = self.sim_model(gt_img, img_preprocessed).item()
        sim_val = 1 - np.array([distance])

        if not is_init:
            image_path = os.path.join(
                output_dir, f'sim_{weights_str}_sim{(sim_val.item()):.3f}.png')
        if output_dir is not None and to_vis and not os.path.exists(image_path):
            img.save(image_path)
            print(f"Saved image to: {image_path}")
        else:
            print(f'Not saved: {image_path}')
        # -- Save to cache
        if cache_dir is not None:
            save_sim_val_npz(cache_path, x, sim_val)
            print(f"Saved result to cache: {cache_path}")
        return sim_val

engine = Engine(OUTPUT_DIR)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}

def _list_image_urls() -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    images = [p for p in OUTPUT_DIR.iterdir() if p.suffix.lower() in exts and p.is_file()]
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
    images = _list_image_urls()
    print(f"Returning {len(images)} images from {OUTPUT_DIR}")
    return JSONResponse({"images": images})

class NextRequest(BaseModel):
    ranking: List[str]

@app.post("/api/next")
def next_step(req: NextRequest) -> JSONResponse:
    # Convert URLs like /outputs/foo.png to basenames 'foo.png'
    basenames: List[str] = []
    for url in req.ranking:
        name = url.split("/outputs/")[-1].split("?")[0]
        if name:
            basenames.append(name)

    engine.next(basenames)

    images = _list_image_urls()
    return JSONResponse({"images": images, 
                         "accepted_ranking": basenames,
                         "selected_basename": engine.last_selected_basename})


# Serve generated outputs and frontend files
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=int(os.environ.get("PORT", "6006")),
        reload=True,
    )


