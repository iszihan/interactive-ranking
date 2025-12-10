"""Engine state serialization helpers."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from server_bo import Engine

ENGINE_STATE_VERSION = 1


def _tensor_to_list(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return [_tensor_to_list(v) for v in value]
    return value


def _dict_to_serializable(record: Dict[str, Any]) -> Dict[str, Any]:
    serialized: Dict[str, Any] = {}
    for key, value in (record or {}).items():
        vec, idx = value
        serialized[key] = {
            "vector": _tensor_to_list(vec),
            "index": idx,
        }
    return serialized


def _dict_from_serialized(data: Dict[str, Any]) -> Dict[str, Any]:
    restored: Dict[str, Any] = {}
    for key, value in (data or {}).items():
        restored[key] = (
            np.asarray(value.get("vector", []), dtype=np.float64),
            int(value.get("index", -1))
        )
    return restored


def export_engine_state(engine: "Engine") -> Dict[str, Any]:
    x_obs = getattr(engine, "x_observations", None)
    if isinstance(x_obs, tuple) and len(x_obs) == 2:
        obs_train = _tensor_to_list(x_obs[0])
        obs_y = _tensor_to_list(x_obs[1])
    else:
        obs_train = None
        obs_y = None

    history_payload: list[Dict[str, Any]] = []
    for item in getattr(engine, "ranking_history", []) or []:
        if not isinstance(item, dict):
            continue
        history_payload.append({
            "step": int(item.get("step", 0)),
            "round": int(item.get("round", 0)),
            "ranking": list(item.get("ranking", [])),
            "indices": [int(x) for x in item.get("indices", [])],
            "selected": item.get("selected"),
            "saved_at": item.get("saved_at"),
        })

    state: Dict[str, Any] = {
        "version": ENGINE_STATE_VERSION,
        "metadata": {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "config_path": str(getattr(engine, "config_path", "")),
            "outputs_dir": str(engine.outputs_dir),
            "step": getattr(engine, "step", 0),
        },
        "engine": {
            "step": getattr(engine, "step", 0),
            "seed": getattr(engine, "seed", 0),
            "last_selected_basename": getattr(engine, "last_selected_basename", None),
            "component_weights": getattr(engine, "component_weights", []),
            "x_record": _dict_to_serializable(getattr(engine, "x_record", {})),
            "x_observations": {
                "train_X": obs_train,
                "Y": obs_y,
            },
            "train_X": _tensor_to_list(getattr(engine, "train_X", None)),
            "Y": _tensor_to_list(getattr(engine, "Y", None)),
            "path": _tensor_to_list(getattr(engine, "path", [])),
            "path_train_X": _tensor_to_list(getattr(engine, "path_train_X", None)),
            "path_Y": _tensor_to_list(getattr(engine, "path_Y", None)),
            "num_warmup": getattr(engine, "num_warmup", 0),
            "MC_SAMPLES": getattr(engine, "MC_SAMPLES", 0),
            "I": list(getattr(engine, "I", [])),
            "last_past_indices": list(getattr(engine, "last_past_indices", [])),
            "state_loaded": getattr(engine, "state_loaded", False),
            "ranking_history": history_payload,
            "last_round_context": getattr(engine, "last_round_context", None),
            "stage_index": getattr(engine, "stage_index", 0),
        }
    }
    return state


def save_engine_state(engine: "Engine", path: Path | str, reason: str | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    state = export_engine_state(engine)
    state.setdefault("metadata", {})["reason"] = reason or "manual"
    with p.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    return p


def load_engine_state(path: Path | str) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_tensor(data: Any, device: torch.device) -> torch.Tensor | None:
    if data is None:
        return None
    return torch.tensor(data, dtype=torch.double, device=device)


def apply_engine_state(engine: "Engine", payload: Dict[str, Any], device: torch.device) -> None:
    engine_payload = payload.get("engine", {})
    engine.step = int(engine_payload.get("step", 0))
    engine.seed = int(engine_payload.get("seed", engine.seed))
    engine.last_selected_basename = engine_payload.get(
        "last_selected_basename")
    engine.component_weights = engine_payload.get(
        "component_weights", engine.component_weights)

    engine.x_record = _dict_from_serialized(engine_payload.get("x_record", {}))

    xo = engine_payload.get("x_observations", {}) or {}
    train_obs = xo.get("train_X")
    y_obs = xo.get("Y")
    if train_obs is not None and y_obs is not None:
        engine.x_observations = (
            np.asarray(train_obs, dtype=np.float64),
            np.asarray(y_obs, dtype=np.float64),
        )

    engine.train_X = _to_tensor(engine_payload.get("train_X"), device)
    engine.Y = _to_tensor(engine_payload.get("Y"), device)

    path_list = engine_payload.get("path") or []
    engine.path = [
        _to_tensor(item, device) if item is not None else None
        for item in path_list
    ]

    engine.path_train_X = _to_tensor(
        engine_payload.get("path_train_X"), device)
    engine.path_Y = _to_tensor(engine_payload.get("path_Y"), device)

    engine.num_warmup = engine_payload.get("num_warmup", engine.num_warmup)
    engine.MC_SAMPLES = engine_payload.get("MC_SAMPLES", engine.MC_SAMPLES)
    engine.I = list(engine_payload.get("I", engine.I))
    engine.last_past_indices = list(engine_payload.get(
        "last_past_indices", engine.last_past_indices))
    engine.ranking_history = list(engine_payload.get(
        "ranking_history", engine.ranking_history))
    last_round_ctx = engine_payload.get("last_round_context")
    if isinstance(last_round_ctx, dict):
        normalized_ctx = dict(last_round_ctx)
        if "new_I" in normalized_ctx:
            normalized_ctx["new_I"] = [
                int(i) for i in normalized_ctx.get("new_I", [])]
        if "round" in normalized_ctx:
            normalized_ctx["round"] = int(normalized_ctx["round"])
        if "iteration" in normalized_ctx and normalized_ctx["iteration"] is not None:
            try:
                normalized_ctx["iteration"] = int(
                    normalized_ctx["iteration"])
            except (TypeError, ValueError):
                normalized_ctx.pop("iteration", None)
        engine.last_round_context = normalized_ctx
    else:
        engine.last_round_context = None
    engine.state_loaded = True

    meta = payload.get("metadata", {})
    stage_index_value = engine_payload.get("stage_index")
    if stage_index_value is not None:
        try:
            engine.stage_index = max(0, int(stage_index_value))
        except Exception:
            pass


def export_slider_history(engine: "Engine") -> list[Dict[str, Any]]:
    history: list[Dict[str, Any]] = []
    for entry in getattr(engine, "slider_history", []) or []:
        x_vals = entry.get("x") if isinstance(entry, dict) else None
        if not isinstance(x_vals, list):
            continue
        record: Dict[str, Any] = {
            "x": [float(v) for v in x_vals],
        }
        if entry.get("image") is not None:
            record["image"] = entry.get("image")
        if entry.get("similarity") is not None:
            try:
                record["similarity"] = float(entry.get("similarity"))
            except (TypeError, ValueError):
                pass
        if entry.get("timestamp") is not None:
            try:
                record["timestamp"] = float(entry.get("timestamp"))
            except (TypeError, ValueError):
                pass
        history.append(record)
    return history


def apply_slider_history(engine: "Engine", payload: list[Dict[str, Any]] | None) -> None:
    history: list[Dict[str, Any]] = []
    for entry in payload or []:
        x_vals = entry.get("x") if isinstance(entry, dict) else None
        if not isinstance(x_vals, list):
            continue
        record: Dict[str, Any] = {
            "x": [float(v) for v in x_vals],
            "image": entry.get("image"),
            "similarity": entry.get("similarity"),
            "timestamp": entry.get("timestamp"),
        }
        history.append(record)
    setattr(engine, "slider_history", history)


def export_slider_state(engine: "Engine") -> Dict[str, Any]:
    metadata = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(getattr(engine, "config_path", "")),
        "outputs_dir": str(getattr(engine, "outputs_dir", "")),
        "step": getattr(engine, "step", 0),
    }

    engine_payload: Dict[str, Any] = {
        "step": getattr(engine, "step", 0),
        "seed": getattr(engine, "seed", 0),
        "last_selected_basename": getattr(engine, "last_selected_basename", None),
        "component_weights": getattr(engine, "component_weights", []),
        "x_record": _dict_to_serializable(getattr(engine, "x_record", {})),
        "last_round_context": getattr(engine, "last_round_context", None),
        "stage_index": getattr(engine, "stage_index", 0),
        "slider_history": export_slider_history(engine),
    }

    return {
        "version": ENGINE_STATE_VERSION,
        "metadata": metadata,
        "engine": engine_payload,
    }


def save_slider_state(engine: "Engine", path: Path | str, reason: str | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    state = export_slider_state(engine)
    state.setdefault("metadata", {})["reason"] = reason or "manual"
    with p.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    return p


def load_slider_state(path: Path | str) -> Dict[str, Any]:
    return load_engine_state(path)


def apply_slider_engine_state(engine: "Engine", payload: Dict[str, Any], device: torch.device) -> None:
    _ = device  # unused for slider state but retained for API compatibility
    engine_payload = payload.get("engine", {}) if isinstance(payload, dict) else {}
    if not isinstance(engine_payload, dict):
        engine_payload = {}

    if "step" in engine_payload:
        try:
            engine.step = int(engine_payload.get("step", engine.step))
        except Exception:
            pass
    if "seed" in engine_payload:
        try:
            engine.seed = int(engine_payload.get("seed", engine.seed))
        except Exception:
            pass

    if "last_selected_basename" in engine_payload:
        engine.last_selected_basename = engine_payload.get("last_selected_basename")
    if "component_weights" in engine_payload:
        engine.component_weights = engine_payload.get("component_weights", [])

    x_record_payload = engine_payload.get("x_record")
    if isinstance(x_record_payload, dict):
        engine.x_record = _dict_from_serialized(x_record_payload)

    last_round_ctx = engine_payload.get("last_round_context")
    if isinstance(last_round_ctx, dict):
        normalized_ctx = dict(last_round_ctx)
        if "new_I" in normalized_ctx:
            normalized_ctx["new_I"] = [int(i) for i in normalized_ctx.get("new_I", [])]
        if "round" in normalized_ctx:
            try:
                normalized_ctx["round"] = int(normalized_ctx["round"])
            except Exception:
                normalized_ctx.pop("round", None)
        if "iteration" in normalized_ctx and normalized_ctx["iteration"] is not None:
            try:
                normalized_ctx["iteration"] = int(normalized_ctx["iteration"])
            except Exception:
                normalized_ctx.pop("iteration", None)
        engine.last_round_context = normalized_ctx
    elif last_round_ctx is None:
        engine.last_round_context = None

    slider_payload = engine_payload.get("slider_history")
    if slider_payload is None and isinstance(payload, dict):
        slider_payload = payload.get("slider_history")
    apply_slider_history(engine, slider_payload if isinstance(slider_payload, list) else None)

    if "stage_index" in engine_payload:
        try:
            engine.stage_index = max(0, int(engine_payload.get("stage_index", engine.stage_index)))
        except Exception:
            pass

    engine.state_loaded = True
