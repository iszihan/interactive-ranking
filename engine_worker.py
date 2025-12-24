"""Worker-side helpers for MultiGPU inference."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import numpy as np
from diffusers.utils import load_image

from engine import obj_sim, infer_image_img2img

_CONTROL_IMG_CACHE: Dict[str, np.ndarray] = {}


def _get_control_img(path: str | None):
  if not path:
    return None
  cached = _CONTROL_IMG_CACHE.get(path)
  if cached is None:
    _CONTROL_IMG_CACHE[path] = np.array(load_image(path))
  return _CONTROL_IMG_CACHE[path]


def worker_make_generation(payload: Dict[str, Any]):
  state = payload["state"]
  w = np.array(payload["w"], dtype=np.float32).reshape(1, -1)
  x = np.array(payload["x"], dtype=np.float32).reshape(1, -1)

  def infer_img_func(component_weights, image_path=None, control_img=None):
    return infer_image_img2img(
        component_weights,
        state["prompt"],
        state["negative_prompt"],
        state["infer_steps"],
        infer_width=state["infer_width"],
        infer_height=state["infer_height"],
        image_path=image_path,
        control_img=control_img,
    )

  is_init = bool(state.get("is_init", False))

  sim_val, image_path = obj_sim(
      state["gt_image_path"],
      copy.deepcopy(state["component_weights"]),
      w,
      weight_idx=state.get("weight_idx", 1),
      infer_image_func=infer_img_func,
        output_dir=state["output_dir"],
        to_vis=True,
        is_init=is_init,
      control_img=_get_control_img(state.get("control_img_path")),
  )

  data = Path(image_path).read_bytes()
  output = {
      "data": data,
      "x": x.flatten().tolist(),
      "sim_val": float(np.asarray(sim_val).flatten()[0]),
      "image_path": str(image_path),
  }
  if 'x_pysps' in payload:
    output['x_pysps'] = payload['x_pysps']
  return output
