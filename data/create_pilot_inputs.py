#!/usr/bin/env python3
import argparse
import os
import json
from copy import deepcopy
from typing import Dict, List, Tuple, Optional

import yaml
import numpy as np
import random


SESSIONS_PER_PARTICIPANT = 2
TASKS_PER_SESSION = 3


def _sum_units_bounds(
    k: int,
    max_units: int,
    min_value: float,
    sum_range: Tuple[Optional[float], Optional[float]],
) -> Tuple[int, int]:
    """
    Helper: compute integer bounds [sum_min_units, sum_max_units] for S_units
    consistent with:
      - each coord >= min_value
      - total sum in (sum_range[0], sum_range[1]] or [*, *] depending on None
    """
    min_units = int(np.ceil(min_value * max_units))
    # baseline lower bound from per-coordinate minimum
    lower_from_min = k * min_units

    # lower bound from sum_range[0]
    if sum_range[0] is None:
        sum_min_units = lower_from_min
    else:
        # open interval on the left: (a, ...] so we go strictly above a
        # e.g. for a = 1.0 and max_units=100 -> 100 -> we want 101 (1.01)
        a_units = int(np.floor(sum_range[0] * max_units))
        sum_min_units = max(lower_from_min, a_units + 1)

    # upper bound from sum_range[1]
    if sum_range[1] is None:
        # no explicit upper bound, but we must at least satisfy lower_from_min
        sum_max_units = max_units  # if you want a hard global cap, keep this
    else:
        sum_max_units = int(np.floor(sum_range[1] * max_units))

    if sum_min_units > sum_max_units:
        raise ValueError(
            f"Infeasible sum range for k={k}, min_value={min_value}, "
            f"sum_range={sum_range}, max_units={max_units}."
        )

    return sum_min_units, sum_max_units


def sample_simplex_point_with_min(
    k: int,
    max_units: int = 100,
    min_value: float = 0.1,
    rng: Optional[random.Random] = None,
    sum_range: Tuple[Optional[float], Optional[float]] = (None, 1.0),
) -> np.ndarray:
    """
    Sample a k-dimensional point with:
      - granularity 1/max_units
      - each w_i >= min_value
      - total sum in (sum_range[0], sum_range[1]] (if provided),
        with the left endpoint treated as open, right as closed.

    Default sum_range (None, 1.0) => sum in [k*min_value, 1.0].
    """
    if rng is None:
        rng = random

    min_units = int(np.ceil(min_value * max_units))

    # Integer bounds on total units S_units
    sum_min_units, sum_max_units = _sum_units_bounds(
        k=k,
        max_units=max_units,
        min_value=min_value,
        sum_range=sum_range,
    )

    # Choose S_units in [sum_min_units, sum_max_units]
    S_units = rng.randint(sum_min_units, sum_max_units)

    # Remaining units after assigning min_units to each coordinate
    R = S_units - k * min_units
    if R < 0:
        raise RuntimeError(
            "Internal error: R < 0, bounds computation went wrong.")

    if R == 0:
        parts_units = [min_units] * k
    else:
        # Random composition of R into k non-negative integers
        cuts = sorted(rng.randint(0, R) for _ in range(k - 1))
        parts_q = []
        prev = 0
        for c in cuts:
            parts_q.append(c - prev)
            prev = c
        parts_q.append(R - prev)
        parts_units = [q + min_units for q in parts_q]

    weights = np.array(parts_units, dtype=float) / float(max_units)

    # Sanity checks
    assert (weights >= min_value - 1e-12).all()
    s = weights.sum()
    # approximate range check at float level
    if sum_range[0] is not None:
        assert s > sum_range[0] - 1e-9
    if sum_range[1] is not None:
        assert s <= sum_range[1] + 1e-9

    return weights


def generate_weight_combinations(
    total_dim: int,
    target_pairs: Dict[int, int],
    seed: int = 0,
    max_units: int = 100,
    min_value: float = 0.1,
    high_sum_samples: Optional[Dict[int, int]] = None,
    high_sum_range: Tuple[float, float] = (1.0, 1.5),
) -> Dict[int, np.ndarray]:
    """
    Generate sparse weight vectors with exact sparsity & min thresholds.

    Args:
        total_dim: total dimensionality n.
        target_pairs: dict mapping k (#non-zero dims) -> total #samples.
        seed: random seed.
        max_units: grid granularity (1/max_units steps).
        min_value: minimum non-zero weight.
        high_sum_samples: optional dict mapping k -> #samples whose sums
            must lie in (high_sum_range[0], high_sum_range[1]].
            The remaining samples for that k will have sums <= 1.0.
        high_sum_range: range (a, b] for "high-sum" samples, default (1.0, 1.5).
    """
    rng = random.Random(seed)
    results: Dict[int, List[np.ndarray]] = {}
    if high_sum_samples is None:
        high_sum_samples = {}

    for k, num_samples in target_pairs.items():
        if k < 1 or k > total_dim:
            raise ValueError(f"Invalid k={k} for total_dim={total_dim}.")

        num_high = high_sum_samples.get(k, 0)
        if num_high < 0 or num_high > num_samples:
            raise ValueError(
                f"high_sum_samples[{k}]={num_high} is invalid for total {num_samples}."
            )
        num_low = num_samples - num_high

        samples_for_k: List[np.ndarray] = []

        # Low-sum samples: sums <= 1.0
        for _ in range(num_low):
            indices = rng.sample(range(total_dim), k)
            local_w = sample_simplex_point_with_min(
                k=k,
                max_units=max_units,
                min_value=min_value,
                rng=rng,
                sum_range=(None, 1.0),  # <= 1.0
            )
            vec = np.zeros(total_dim, dtype=float)
            vec[indices] = local_w
            samples_for_k.append(vec)

        # High-sum samples: sums in (high_sum_range[0], high_sum_range[1]]
        for _ in range(num_high):
            indices = rng.sample(range(total_dim), k)
            local_w = sample_simplex_point_with_min(
                k=k,
                max_units=max_units,
                min_value=min_value,
                rng=rng,
                sum_range=high_sum_range,
            )
            vec = np.zeros(total_dim, dtype=float)
            vec[indices] = local_w
            samples_for_k.append(vec)

        results[k] = np.vstack(samples_for_k)

    # ======== VERIFICATION CHECK ========
    for k, num_samples in target_pairs.items():
        arr = results[k]
        num_high = high_sum_samples.get(k, 0)
        num_low = num_samples - num_high

        # 1. Check #samples matches
        if arr.shape[0] != num_samples:
            raise ValueError(
                f"Sparsity k={k}: expected {num_samples} samples but got {arr.shape[0]}."
            )

        for i, row in enumerate(arr):
            nz = np.count_nonzero(row)
            if nz != k:
                raise ValueError(
                    f"Sparsity k={k}, sample {i}: expected {k} non-zero entries "
                    f"but found {nz}."
                )

            s = row.sum()
            nz_values = row[row > 0]
            if (nz_values < min_value - 1e-12).any():
                raise ValueError(
                    f"Sparsity k={k}, sample {i}: found a weight < min_value={min_value}."
                )

            # Low-sum part (first num_low rows)
            if i < num_low:
                if s > 1.0 + 1e-9:
                    raise ValueError(
                        f"Sparsity k={k}, sample {i}: low-sum sample has sum={s} > 1.0."
                    )
            else:
                # High-sum part
                low, high = high_sum_range
                if not (s > low - 1e-9 and s <= high + 1e-9):
                    raise ValueError(
                        f"Sparsity k={k}, sample {i}: high-sum sample has sum={s}, "
                        f"expected in ({low}, {high}]."
                    )

    print("✔ Verification passed: all constraints satisfied.")
    return results


def flatten_weight_sets(weight_sets: Dict[int, np.ndarray]) -> List[np.ndarray]:
    combos: List[np.ndarray] = []
    for k in sorted(weight_sets.keys()):
        arr = weight_sets[k]
        if arr.ndim != 2:
            raise ValueError(f"weight_sets[{k}] must be 2D, got shape {arr.shape}")
        combos.extend(list(arr))
    return combos

def choose_article(word: str) -> str:
    """
    Very simple heuristic to choose 'a' or 'an'.
    """
    if not word:
        return "a"
    return "an" if word[0].lower() in "aeiou" else "a"


def build_prompts_from_classes(json_path: str) -> List[str]:
    """
    Expected JSON format:
    {
      "people": ["baby", "boy", "girl", "man", "woman"],
      "animals": [...],
      "objects": [...]
    }

    Returns a flat list of prompts in the order:
    people, then animals, then objects.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            "Expected JSON to be a dict with keys 'people', 'animals', 'objects'.")

    prompts: List[str] = []

    # Order is explicit to avoid relying on dict order
    for category in ["people", "animals", "objects"]:
        if category not in data:
            continue
        labels = data[category]
        if not isinstance(labels, list):
            raise ValueError(
                f"JSON[{category}] must be a list of class names.")
        for label in labels:
            if not isinstance(label, str):
                raise ValueError(
                    f"Label in JSON[{category}] must be a string.")
            article = choose_article(label)

            if category == "people":
                # e.g., "A portrait of a woman"
                prompt = f"A portrait of {article} {label}"
            else:
                # e.g., "A drawing of a tiger"
                # If you prefer painting, change "drawing" -> "painting"
                prompt = f"A drawing of {article} {label}"

            prompts.append(prompt)

    return prompts


def load_model_paths(yml_path: str) -> List[str]:
    with open(yml_path, "r") as f:
        data = yaml.safe_load(f)

    if isinstance(data, list):
        if not all(isinstance(x, str) for x in data):
            raise ValueError(
                "YAML list must contain only strings (model paths).")
        return data

    raise ValueError("Unsupported YAML format for model file list.")


def normalize_interface_name(name: str) -> str:
    cleaned = name.strip().replace(" ", "_").replace("/", "_")
    if not cleaned:
        raise ValueError("Interface name cannot be empty.")
    return cleaned


def parse_template_args(template_args: List[str]) -> List[Tuple[str, str]]:
    if not template_args:
        raise ValueError(
            "Provide three --template entries formatted as NAME=PATH.")

    templates: List[Tuple[str, str]] = []
    for raw in template_args:
        if "=" not in raw:
            raise ValueError(
                f"Template '{raw}' must be formatted as NAME=PATH.")
        name, path = raw.split("=", 1)
        name_clean = normalize_interface_name(name)
        path_clean = path.strip()
        if not path_clean:
            raise ValueError("Template path cannot be empty.")
        templates.append((name_clean, path_clean))

    if len(templates) != 3:
        raise ValueError(
            f"Expected exactly three templates, got {len(templates)}.")

    seen = set()
    for name, _ in templates:
        if name in seen:
            raise ValueError("Template names must be unique.")
        seen.add(name)

    return templates


def load_templates(template_specs: List[Tuple[str, str]]) -> Dict[str, dict]:
    loaded: Dict[str, dict] = {}
    for name, path in template_specs:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Template {path} must be a YAML mapping.")
        loaded[name] = data
    return loaded


def infer_participants_count(total_available: int, requested: Optional[int]) -> int:
    per_participant = SESSIONS_PER_PARTICIPANT * TASKS_PER_SESSION

    if total_available <= 0:
        raise ValueError("No prompts/weights available to schedule.")

    if requested is None:
        participants = total_available // per_participant
        if participants < 1:
            raise ValueError(
                f"Need at least {per_participant} samples to fill one participant; only {total_available} available.")
        return participants

    if requested <= 0:
        raise ValueError("--par must be positive if provided.")

    needed = requested * per_participant
    if needed > total_available:
        raise ValueError(
            f"With par={requested}, need {needed} tasks but only {total_available} prompts/weights are available. Reduce --par or supply more inputs.")
    return requested


def build_task_plan(
    participants: int,
    template_specs: List[Tuple[str, str]],
    seed: int,
) -> List[Dict[str, int]]:
    rng = random.Random(seed)
    plan: List[Dict[str, int]] = []

    for pid in range(1, participants + 1):
        for session_id in range(1, SESSIONS_PER_PARTICIPANT + 1):
            session_templates = list(template_specs)
            rng.shuffle(session_templates)
            for task_in_session, (name, _) in enumerate(session_templates, start=1):
                plan.append(
                    {
                        "participant": pid,
                        "session": session_id,
                        "task_in_session": task_in_session,
                        "template_name": name,
                    }
                )

    return plan


def format_weight(w: float) -> str:
    """
    Format a weight as a 'simple' decimal:
    - 2 decimal places, then strip trailing zeros and decimal point if not needed.
    """
    s = f"{w:.2f}".rstrip("0").rstrip(".")
    return s if s else "0"


def build_gt_string(prompt: str, model_paths: List[str], weights: np.ndarray) -> str:
    """Return the GT string "'<prompt>'@path_a:wa,path_b:wb,..."."""
    if len(model_paths) != len(weights):
        raise ValueError(
            f"Model paths length {len(model_paths)} != weight vector length {len(weights)}"
        )

    parts = []
    for path, w in zip(model_paths, weights):
        w_str = format_weight(float(w))
        path_with_ext = path if path.endswith(".safetensors") else f"{path}.safetensors"
        parts.append(f"{path_with_ext}:{w_str}")
    weights_str = ",".join(parts)

    return f"'{prompt}'@{weights_str}"


def main(
    classes_json: str,
    models_yml: str,
    out_dir: str,
    templates: List[str],
    participants: Optional[int] = None,
    tutorial: bool = False,
    shuffle_weights: bool = True,
    seed: int = 0,
):
    os.makedirs(out_dir, exist_ok=True)

    template_specs = parse_template_args(templates)
    template_data = load_templates(template_specs)

    model_paths = load_model_paths(models_yml)
    prompts = build_prompts_from_classes(classes_json)

    dim = len(model_paths)
    if dim == 0:
        raise ValueError("models_yml must contain at least one model path.")

    num_prompts = len(prompts)
    if num_prompts == 0:
        raise ValueError("No prompts generated from classes_json; nothing to do.")

    # Fixed recipe for pilot: predefined sparsity counts and some high-sum samples
    target_pairs = {
        2: 12,
        3: 10,
        4: 8,
    }

    high_sum_samples = {
        2: 2,
        3: 2,
        4: 1,
    }

    weight_sets = generate_weight_combinations(
        total_dim=dim,
        target_pairs=target_pairs,
        seed=seed,
        max_units=100,
        min_value=0.2,
        high_sum_samples=high_sum_samples,
        high_sum_range=(1.0, 1.5),
    )

    combos = flatten_weight_sets(weight_sets)
    if shuffle_weights and len(combos) > 1:
        np.random.default_rng(seed).shuffle(combos)

    available = min(num_prompts, len(combos))
    output_parent = os.path.dirname(os.path.abspath(out_dir))
    rng = random.Random(seed)

    if tutorial:
        if participants is None:
            participants_count = 1
        else:
            if participants <= 0:
                raise ValueError("--par must be positive when provided.")
            participants_count = participants

        # Reserve the first study-sized block of inputs for the actual sessions
        # so tutorial pulls from the remaining pool.
        reserve_for_study = participants_count * SESSIONS_PER_PARTICIPANT * TASKS_PER_SESSION

        if available < reserve_for_study + len(template_specs):
            raise ValueError(
                f"Tutorial mode needs at least {reserve_for_study + len(template_specs)} prompt/weight pairs; only {available} available.")

        base_entries = list(zip(prompts, combos))
        rng.shuffle(base_entries)
        base_entries = base_entries[reserve_for_study:]

        chosen = base_entries[: len(template_specs)]
        interface_gt = {name: pw for (name, _), pw in zip(template_specs, chosen)}

        # Tutorial: single session, fixed interface order for all participants
        total_needed = participants_count * len(template_specs)

        tutorial_order = template_specs
        slider_first = [tpl for tpl in template_specs if tpl[0].lower() == "slider"]
        if slider_first:
            others = [tpl for tpl in template_specs if tpl[0].lower() != "slider"]
            tutorial_order = slider_first + others

        for pid in range(1, participants_count + 1):
            p_tag = f"P{pid:02d}_tut"
            s_tag = "S1"
            for task_idx, (iface_name, _) in enumerate(tutorial_order, start=1):
                prompt, weights = interface_gt[iface_name]
                t_tag = f"T{task_idx:02d}"
                task_label = f"{p_tag}_{s_tag}_{t_tag}_{iface_name}"

                output_dir_value = os.path.join(output_parent, p_tag, f"{s_tag}_{t_tag}")
                init_dir_value = os.path.join(output_parent, p_tag, f"init_{s_tag}_{t_tag}")
                os.makedirs(output_dir_value, exist_ok=True)
                os.makedirs(init_dir_value, exist_ok=True)

                cfg = deepcopy(template_data[iface_name])
                cfg['output_dir'] = output_dir_value
                cfg['init_dir'] = init_dir_value
                cfg['gt_config'] = build_gt_string(prompt, model_paths, weights)

                filename = os.path.join(out_dir, f"{task_label}.yml")
                with open(filename, "w") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False)

                print(f"Wrote {filename} | prompt: {prompt}")

        print(
            f"Done. Tutorial mode wrote {total_needed} config files for {participants_count} participants to: {out_dir}"
        )
        return

    participants_count = infer_participants_count(available, participants)
    total_needed = participants_count * SESSIONS_PER_PARTICIPANT * TASKS_PER_SESSION

    if available < total_needed:
        raise ValueError(
            f"Need {total_needed} prompts/weights for par={participants_count} but only {available} are available.")

    gt_entries = list(zip(prompts, combos))  # zip truncates to the min length

    rng.shuffle(gt_entries)
    gt_entries = gt_entries[:total_needed]

    task_plan = build_task_plan(participants_count, template_specs, seed)

    for (prompt, weights), task in zip(gt_entries, task_plan):
        p_tag = f"P{task['participant']:02d}"
        s_tag = f"S{task['session']}"
        t_tag = f"T{task['task_in_session']:02d}"
        task_label = f"{p_tag}_{s_tag}_{t_tag}_{task['template_name']}"

        output_dir_value = os.path.join(output_parent, p_tag, f"{s_tag}_{t_tag}")
        init_dir_value = os.path.join(output_parent, p_tag, f"init_{s_tag}_{t_tag}")
        os.makedirs(output_dir_value, exist_ok=True)
        os.makedirs(init_dir_value, exist_ok=True)

        cfg = deepcopy(template_data[task['template_name']])
        cfg['output_dir'] = output_dir_value
        cfg['init_dir'] = init_dir_value
        cfg['gt_config'] = build_gt_string(prompt, model_paths, weights)

        filename = os.path.join(out_dir, f"{task_label}.yml")
        with open(filename, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        print(f"Wrote {filename} | prompt: {prompt}")

    print(
        f"Done. Wrote {total_needed} config files for {participants_count} participants to: {out_dir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate test input config files from weight combos, class JSON, and model list."
    )
    parser.add_argument("--classes-json", required=True,
                        help="Path to JSON with people/animals/objects classes.")
    parser.add_argument("--models-yml", required=True,
                        help="Path to YAML with model file paths.")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for config .yml files.")
    parser.add_argument(
        "--template",
        action="append",
        dest="templates",
        required=True,
        metavar="NAME=PATH",
        help="Interface template in the form name=path. Provide exactly three, one per interface.",
    )
    parser.add_argument(
        "--par",
        type=int,
        default=None,
        help="Number of participants. If omitted, inferred by filling as many full participants as available prompts/weights allow (2 sessions × 3 tasks each).",
    )
    parser.add_argument(
        "--tutorial",
        action="store_true",
        help="Generate tutorial configs: pick one prompt/weight per interface, reuse across participants, still writing per-session folders.",
    )
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Do not shuffle generated weight combinations.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for generation and shuffling.")

    args = parser.parse_args()
    main(
        classes_json=args.classes_json,
        models_yml=args.models_yml,
        out_dir=args.out_dir,
        templates=args.templates,
        participants=args.par,
        tutorial=args.tutorial,
        shuffle_weights=not args.no_shuffle,
        seed=args.seed,
    )
