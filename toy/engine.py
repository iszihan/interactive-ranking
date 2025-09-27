import os
import copy
import glob
import math
from pathlib import Path

from PIL import Image
import numpy as np
import torch
from dreamsim import dreamsim
from botorch.utils.sampling import draw_sobol_samples
import pytorch_lightning as pl

from helper.infer import infer_img2img
from search_benchmark.util import (
    get_x_hash, save_sim_val_npz, load_sim_val_npz, save_img_npz, load_img_npz, find_triggers)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
sim_model, preprocess = None, None
gt_img = None


def infer_image_img2img(component_weights,
                        prompt,
                        negative_prompt,
                        steps,
                        infer_width=512,
                        infer_height=512,
                        image_path=None,
                        control_img=None):

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


def obj_sim(gt_img_path, component_weights, x, weight_idx, infer_image_func,
            output_dir=None, to_vis=False, is_init=False, to_cache=True, control_img=None):

    global sim_model, preprocess, device, gt_img
    print(f"Evaluating with weights: {x}")
    
    infer_component_weights = copy.deepcopy(component_weights)
    for i in range(len(infer_component_weights)):
        infer_component_weights[i][weight_idx] = x.flatten()[i]
    weights_str = ','.join(
        [f"{c[weight_idx]:.2f}" for c in infer_component_weights])

    # -- Check cache
    cache_dir = None
    if output_dir is not None and to_cache and not is_init:
        cache_dir = os.path.join(output_dir, 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        x_hash = get_x_hash(x)
        cache_path = os.path.join(cache_dir, f"{x_hash}.npz")
        img_cache_path = os.path.join(cache_dir, f"{x_hash}_img.npz")
        if os.path.exists(cache_path):
            sim_val = load_sim_val_npz(cache_path)
            img = load_img_npz(img_cache_path)
            print(f"Loaded cached result {sim_val} from: {cache_path}")

            img = Image.fromarray(img)
            image_path = os.path.join(
                output_dir, f'sim_{weights_str}_sim{(sim_val.item()):.3f}.png')
            img.save(image_path)

            return sim_val, image_path

    if sim_model is None:
        sim_model, preprocess = dreamsim(pretrained=True, device=device)

    img = None
    if is_init:
        image_path = os.path.join(
            output_dir, f'init_{weights_str}.png')
        if os.path.exists(image_path):
            print(f"Image {image_path} already exists, skipping inference.")
            img = Image.open(image_path)
        else:
            print(f"Image {image_path} does not exist, running inference.")

    if img is None:
        if control_img is None:
            _, img = infer_image_func(infer_component_weights, image_path=None)
        else:
            _, img = infer_image_func(
                infer_component_weights, image_path=None, control_img=control_img)

    img_preprocessed = preprocess(img).to(device)

    if gt_img is None:
        gt_img = Image.open(gt_img_path)
        gt_img = preprocess(gt_img).to(device)

    distance = sim_model(gt_img, img_preprocessed).item()
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
        save_img_npz(img_cache_path, x, np.array(img))
        print(f"Saved result to cache: {cache_path}")

    return sim_val, image_path


def prepare_init_obs(num_observations, num_dim, x_range, f,
                     seed=0):
    """Prepare initial observations for the optimization."""
    pl.seed_everything(seed)

    train_X = draw_sobol_samples(
        bounds=torch.tensor([x_range] * num_dim).T.double(),
        n=num_observations, q=1, seed=seed
    ).squeeze(1).double()
    x_record = {}
    for i in range(num_observations):
        sim_val, image_path = f(
            train_X[i].reshape(1, -1).detach().cpu().numpy())
        x_record[Path(image_path).name] = (
            train_X[i].detach().cpu().numpy(), -1)

    return train_X.detach().cpu().numpy(), x_record
