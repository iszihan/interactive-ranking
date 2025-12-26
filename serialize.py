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


def _safe_int(value: Any, default: int | None = None) -> int | None:
    """Cast to int when possible; otherwise return ``default``.

    Avoids ``int(None)`` errors during serialization when optional fields (e.g.,
    stage transitions without a round id) are present in history records.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_state_save_path(target: Path | str) -> Path:
    """Return a file path for saving, generating a timestamped name for dirs."""
    candidate = Path(target)
    if candidate.suffix:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    directory = candidate
    stem = directory.name or "state"
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return directory / f"{stem}_{timestamp}.json"


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
        vec = None
        idx = None
        image_path = None
        if isinstance(value, (list, tuple)):
            if len(value) >= 2:
                vec, idx = value[0], value[1]
            if len(value) >= 3:
                image_path = value[2]
        elif isinstance(value, dict):
            vec = value.get("vector")
            idx = value.get("index")
            image_path = value.get("image_path")
        if vec is None or idx is None:
            # Skip malformed entries rather than raising during save.
            continue
        entry = {
            "vector": _tensor_to_list(vec),
            "index": idx,
        }
        if image_path is not None:
            entry["image_path"] = str(image_path)
        serialized[key] = entry
    return serialized


def _dict_from_serialized(data: Dict[str, Any]) -> Dict[str, Any]:
    restored: Dict[str, Any] = {}
    for key, value in (data or {}).items():
        vector = np.asarray(value.get("vector", []), dtype=np.float64)
        index = _safe_int(value.get("index", -1), -1)
        image_path = value.get("image_path")
        if image_path is None:
            restored[key] = (vector, index)
        else:
            restored[key] = (vector, index, image_path)
    return restored


def export_engine_state(engine: "Engine") -> Dict[str, Any]:
    x_obs = getattr(engine, "x_observations", None)
    if isinstance(x_obs, tuple) and len(x_obs) == 2:
        obs_train = _tensor_to_list(x_obs[0])
        obs_y = _tensor_to_list(x_obs[1])
    else:
        obs_train = None
        obs_y = None

    init_ready_ts = getattr(engine, "init_ready_timestamp", None)
    try:
        init_ready_ts = float(
            init_ready_ts) if init_ready_ts is not None else None
    except (TypeError, ValueError):
        init_ready_ts = None

    history_payload: list[Dict[str, Any]] = []
    for item in getattr(engine, "ranking_history", []) or []:
        if not isinstance(item, dict):
            continue
        train_version_value = _safe_int(item.get("train_version", 0), 0)

        new_y_snapshot = item.get("new_y")
        if isinstance(new_y_snapshot, torch.Tensor):
            new_y_snapshot = _tensor_to_list(new_y_snapshot)

        result_images_snapshot = item.get("result_images")
        if isinstance(result_images_snapshot, (list, tuple)):
            result_images_snapshot = [str(x) for x in result_images_snapshot]

        ready_at_val = item.get("ready_at")
        try:
            ready_at_val = float(ready_at_val) if ready_at_val is not None else None
        except (TypeError, ValueError):
            ready_at_val = None

        indices_payload: list[int] = []
        for x in item.get("indices", []) or []:
            val = _safe_int(x, None)
            if val is not None:
                indices_payload.append(val)

        history_payload.append({
            "step": _safe_int(item.get("step"), 0) or 0,
            "round": _safe_int(item.get("round"), None),
            "ranking": list(item.get("ranking", [])),
            "indices": indices_payload,
            "selected": item.get("selected"),
            "saved_at": item.get("saved_at"),
            "ready_at": ready_at_val,
            "train_version": train_version_value,
            "new_y": new_y_snapshot,
            "result_images": result_images_snapshot,
        })

    train_history_payload: list[Dict[str, Any]] = []
    for snapshot in getattr(engine, "train_dataset_history", []) or []:
        if not isinstance(snapshot, dict):
            continue
        train_history_payload.append({
            "version": int(snapshot.get("version", 0)),
            "stage_index": int(snapshot.get("stage_index", 0)),
            "step": int(snapshot.get("step", 0)),
            "reason": snapshot.get("reason"),
            "train_X": _tensor_to_list(snapshot.get("train_X")),
            "Y": _tensor_to_list(snapshot.get("Y")),
            "lora_merge_indices": [int(i) for i in snapshot.get("lora_merge_indices", [])],
            "lora_merge_weights": [float(w) for w in snapshot.get("lora_merge_weights", [])],
            "dim_index_per_lora": [int(i) for i in snapshot.get("dim_index_per_lora", [])],
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
            "dim2loras": getattr(engine, "dim2loras", None),
            "lora_merge_indices": list(getattr(engine, "lora_merge_indices", [])),
            "lora_merge_weights": list(getattr(engine, "lora_merge_weights", [])),
            "dim_index_per_lora": list(getattr(engine, "dim_index_per_lora", [])),
            "comp_pairs": _tensor_to_list(getattr(engine, "comp_pairs", None)),
            "x_record": _dict_to_serializable(getattr(engine, "x_record", {})),
            "x_observations": {
                "train_X": obs_train,
                "Y": obs_y,
            },
            "train_X": _tensor_to_list(getattr(engine, "train_X", None)),
            "Y": _tensor_to_list(getattr(engine, "Y", None)),
            "train_dataset_history": train_history_payload,
            "train_dataset_version": getattr(engine, "train_dataset_version", 0),
            "path": _tensor_to_list(getattr(engine, "path", [])),
            "num_warmup": getattr(engine, "num_warmup", 0),
            "MC_SAMPLES": getattr(engine, "MC_SAMPLES", 0),
            "last_past_indices": list(getattr(engine, "last_past_indices", [])),
            "state_loaded": getattr(engine, "state_loaded", False),
            "ranking_history": history_payload,
            "last_round_context": getattr(engine, "last_round_context", None),
            "stage_index": getattr(engine, "stage_index", 0),
            "init_ready_timestamp": init_ready_ts,
        }
    }
    return state


def save_engine_state(engine: "Engine", path: Path | str, reason: str | None = None) -> Path:
    p = build_state_save_path(path)
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
    cpu_device = torch.device("cpu")
    engine.step = _safe_int(engine_payload.get("step", 0), 0) or 0
    engine.seed = _safe_int(engine_payload.get("seed", engine.seed), engine.seed) or engine.seed
    engine.last_selected_basename = engine_payload.get(
        "last_selected_basename")
    engine.component_weights = engine_payload.get(
        "component_weights", engine.component_weights)
    if "dim2loras" in engine_payload and engine_payload.get("dim2loras") is not None:
        engine.dim2loras = engine_payload.get("dim2loras")
    if "lora_merge_indices" in engine_payload:
        engine.lora_merge_indices = list(
            engine_payload.get("lora_merge_indices") or [])
    if "lora_merge_weights" in engine_payload:
        engine.lora_merge_weights = list(
            engine_payload.get("lora_merge_weights") or [])
    if "dim_index_per_lora" in engine_payload:
        engine.dim_index_per_lora = list(
            engine_payload.get("dim_index_per_lora") or [])

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

    comp_pairs_payload = engine_payload.get("comp_pairs")
    if comp_pairs_payload is not None:
        engine.comp_pairs = _to_tensor(comp_pairs_payload, device)
    else:
        engine.comp_pairs = torch.empty(
            (0, 2), dtype=torch.long, device=device)

    engine.train_dataset_version = _safe_int(
        engine_payload.get("train_dataset_version", getattr(engine, "train_dataset_version", 0)),
        getattr(engine, "train_dataset_version", 0)) or getattr(engine, "train_dataset_version", 0)
    dataset_history_payload = engine_payload.get("train_dataset_history") or []
    restored_history: list[Dict[str, Any]] = []
    for snapshot_payload in dataset_history_payload:
        if not isinstance(snapshot_payload, dict):
            continue
        restored_history.append({
            "version": _safe_int(snapshot_payload.get("version", 0), 0) or 0,
            "stage_index": _safe_int(snapshot_payload.get("stage_index", 0), 0) or 0,
            "step": _safe_int(snapshot_payload.get("step", 0), 0) or 0,
            "reason": snapshot_payload.get("reason"),
            "train_X": _to_tensor(snapshot_payload.get("train_X"), cpu_device),
            "Y": _to_tensor(snapshot_payload.get("Y"), cpu_device),
            "lora_merge_indices": [int(i) for i in snapshot_payload.get("lora_merge_indices", [])],
            "lora_merge_weights": [float(w) for w in snapshot_payload.get("lora_merge_weights", [])],
            "dim_index_per_lora": [int(i) for i in snapshot_payload.get("dim_index_per_lora", [])],
        })
    engine.train_dataset_history = restored_history

    path_list = engine_payload.get("path") or []
    engine.path = [
        _to_tensor(item, device) if item is not None else None
        for item in path_list
    ]

    engine.num_warmup = engine_payload.get("num_warmup", engine.num_warmup)
    engine.MC_SAMPLES = engine_payload.get("MC_SAMPLES", engine.MC_SAMPLES)
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
            round_val = _safe_int(normalized_ctx.get("round"), None)
            if round_val is None:
                normalized_ctx.pop("round", None)
            else:
                normalized_ctx["round"] = round_val
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

    init_ready_ts = engine_payload.get("init_ready_timestamp")
    try:
        engine.init_ready_timestamp = float(
            init_ready_ts) if init_ready_ts is not None else None
    except (TypeError, ValueError):
        engine.init_ready_timestamp = None

    meta = payload.get("metadata", {})
    stage_index_value = engine_payload.get("stage_index")
    stage_index_cast = _safe_int(stage_index_value, None)
    if stage_index_cast is not None:
        engine.stage_index = max(0, stage_index_cast)


def export_slider_history(engine: "Engine") -> list[Dict[str, Any]]:
    history: list[Dict[str, Any]] = []
    for entry in getattr(engine, "slider_history", []) or []:
        x_vals = entry.get("x") if isinstance(entry, dict) else None
        if not isinstance(x_vals, list):
            continue
        ready_at_val = entry.get("ready_at")
        try:
            ready_at_val = float(ready_at_val) if ready_at_val is not None else None
        except (TypeError, ValueError):
            ready_at_val = None
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
        if ready_at_val is not None:
            record["ready_at"] = ready_at_val
        history.append(record)
    return history


def apply_slider_history(engine: "Engine", payload: list[Dict[str, Any]] | None) -> None:
    history: list[Dict[str, Any]] = []
    for entry in payload or []:
        x_vals = entry.get("x") if isinstance(entry, dict) else None
        if not isinstance(x_vals, list):
            continue
        ready_at_val = entry.get("ready_at")
        try:
            ready_at_val = float(ready_at_val) if ready_at_val is not None else None
        except (TypeError, ValueError):
            ready_at_val = None
        record: Dict[str, Any] = {
            "x": [float(v) for v in x_vals],
            "image": entry.get("image"),
            "similarity": entry.get("similarity"),
            "timestamp": entry.get("timestamp"),
        }
        if ready_at_val is not None:
            record["ready_at"] = ready_at_val
        history.append(record)
    setattr(engine, "slider_history", history)


def export_slider_state(engine: "Engine") -> Dict[str, Any]:
    metadata = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(getattr(engine, "config_path", "")),
        "outputs_dir": str(getattr(engine, "outputs_dir", "")),
        "step": getattr(engine, "step", 0),
    }

    init_ready_ts = getattr(engine, "init_ready_timestamp", None)
    try:
        init_ready_ts = float(
            init_ready_ts) if init_ready_ts is not None else None
    except (TypeError, ValueError):
        init_ready_ts = None

    engine_payload: Dict[str, Any] = {
        "step": getattr(engine, "step", 0),
        "seed": getattr(engine, "seed", 0),
        "last_selected_basename": getattr(engine, "last_selected_basename", None),
        "component_weights": getattr(engine, "component_weights", []),
        "dim2loras": getattr(engine, "dim2loras", None),
        "lora_merge_indices": list(getattr(engine, "lora_merge_indices", [])),
        "lora_merge_weights": list(getattr(engine, "lora_merge_weights", [])),
        "dim_index_per_lora": list(getattr(engine, "dim_index_per_lora", [])),
        "x_record": _dict_to_serializable(getattr(engine, "x_record", {})),
        "last_round_context": getattr(engine, "last_round_context", None),
        "stage_index": getattr(engine, "stage_index", 0),
        "slider_history": export_slider_history(engine),
        "init_ready_timestamp": init_ready_ts,
    }

    return {
        "version": ENGINE_STATE_VERSION,
        "metadata": metadata,
        "engine": engine_payload,
    }


def save_slider_state(engine: "Engine", path: Path | str, reason: str | None = None) -> Path:
    p = build_state_save_path(path)
    state = export_slider_state(engine)
    state.setdefault("metadata", {})["reason"] = reason or "manual"
    with p.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    return p


def load_slider_state(path: Path | str) -> Dict[str, Any]:
    return load_engine_state(path)


def apply_slider_engine_state(engine: "Engine", payload: Dict[str, Any], device: torch.device) -> None:
    _ = device  # unused for slider state but retained for API compatibility
    engine_payload = payload.get(
        "engine", {}) if isinstance(payload, dict) else {}
    if not isinstance(engine_payload, dict):
        engine_payload = {}

    if "step" in engine_payload:
        new_step = _safe_int(engine_payload.get("step", engine.step), engine.step)
        if new_step is not None:
            engine.step = new_step
    if "seed" in engine_payload:
        new_seed = _safe_int(engine_payload.get("seed", engine.seed), engine.seed)
        if new_seed is not None:
            engine.seed = new_seed

    if "last_selected_basename" in engine_payload:
        engine.last_selected_basename = engine_payload.get(
            "last_selected_basename")
    if "component_weights" in engine_payload:
        engine.component_weights = engine_payload.get("component_weights", [])
    if "dim2loras" in engine_payload and engine_payload.get("dim2loras") is not None:
        engine.dim2loras = engine_payload.get("dim2loras")
    if "lora_merge_indices" in engine_payload:
        engine.lora_merge_indices = list(
            engine_payload.get("lora_merge_indices") or [])
    if "lora_merge_weights" in engine_payload:
        engine.lora_merge_weights = list(
            engine_payload.get("lora_merge_weights") or [])
    if "dim_index_per_lora" in engine_payload:
        engine.dim_index_per_lora = list(
            engine_payload.get("dim_index_per_lora") or [])

    x_record_payload = engine_payload.get("x_record")
    if isinstance(x_record_payload, dict):
        engine.x_record = _dict_from_serialized(x_record_payload)

    last_round_ctx = engine_payload.get("last_round_context")
    if isinstance(last_round_ctx, dict):
        normalized_ctx = dict(last_round_ctx)
        if "new_I" in normalized_ctx:
            normalized_ctx["new_I"] = [
                int(i) for i in normalized_ctx.get("new_I", [])]
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
    apply_slider_history(engine, slider_payload if isinstance(
        slider_payload, list) else None)

    init_ready_ts = engine_payload.get("init_ready_timestamp")
    try:
        engine.init_ready_timestamp = float(
            init_ready_ts) if init_ready_ts is not None else None
    except (TypeError, ValueError):
        engine.init_ready_timestamp = None

    if "stage_index" in engine_payload:
        stage_cast = _safe_int(engine_payload.get("stage_index", engine.stage_index), engine.stage_index)
        if stage_cast is not None:
            engine.stage_index = max(0, stage_cast)

    engine.state_loaded = True
