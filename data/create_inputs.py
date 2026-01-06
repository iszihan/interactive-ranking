#!/usr/bin/env python3
import argparse
import os
import json
from collections import defaultdict, Counter
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
    high_sum_bins: Optional[Dict[int,
                                 List[Tuple[int, Tuple[float, float]]]]] = None,
) -> Dict[int, np.ndarray]:
    """
    Generate sparse weight vectors with exact sparsity & min thresholds.

    Args:
        total_dim: total dimensionality n.
        target_pairs: dict mapping k (#non-zero dims) -> total #samples.
        seed: random seed.
        max_units: grid granularity (1/max_units steps).
        min_value: minimum non-zero weight.
        high_sum_bins: optional dict mapping k -> list of (count, (low, high])
            bins. If omitted or empty for a given k, all samples for that k
            are drawn with sums <= 1.0.
    """
    rng = random.Random(seed)
    results: Dict[int, List[np.ndarray]] = {}
    if high_sum_bins is None:
        high_sum_bins = {}

    for k, num_samples in target_pairs.items():
        if k < 1 or k > total_dim:
            raise ValueError(f"Invalid k={k} for total_dim={total_dim}.")

        # Decide bin plan for this k
        bins_for_k: List[Tuple[int, Tuple[float, float]]
                         ] = high_sum_bins.get(k, [])

        total_high = sum(c for c, _ in bins_for_k)
        if total_high > num_samples:
            raise ValueError(
                f"High-sum requests exceed total for k={k}: {total_high} > {num_samples}."
            )
        num_low = num_samples - total_high

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

        # High-sum samples per bin: sums in (low, high]
        for count, sum_range in bins_for_k:
            for _ in range(count):
                indices = rng.sample(range(total_dim), k)
                local_w = sample_simplex_point_with_min(
                    k=k,
                    max_units=max_units,
                    min_value=min_value,
                    rng=rng,
                    sum_range=sum_range,
                )
                vec = np.zeros(total_dim, dtype=float)
                vec[indices] = local_w
                samples_for_k.append(vec)

        results[k] = np.vstack(samples_for_k)

    # ======== VERIFICATION CHECK ========
    for k, num_samples in target_pairs.items():
        arr = results[k]
        bins_for_k = high_sum_bins.get(k, [])
        total_high = sum(c for c, _ in bins_for_k)
        num_low = num_samples - total_high

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
                # Determine which high bin this sample should satisfy.
                offset = i - num_low
                cumulative = 0
                matched = False
                for count, (low, high) in bins_for_k:
                    if offset < cumulative + count:
                        if not (s > low - 1e-9 and s <= high + 1e-9):
                            raise ValueError(
                                f"Sparsity k={k}, sample {i}: high-sum sample has sum={s}, "
                                f"expected in ({low}, {high}]."
                            )
                        matched = True
                        break
                    cumulative += count
                if not matched:
                    raise RuntimeError(
                        "Verification logic failed to match bin.")

    print("✔ Verification passed: all constraints satisfied.")
    return results


def flatten_weight_sets(weight_sets: Dict[int, np.ndarray]) -> List[np.ndarray]:
    combos: List[np.ndarray] = []
    for k in sorted(weight_sets.keys()):
        arr = weight_sets[k]
        if arr.ndim != 2:
            raise ValueError(
                f"weight_sets[{k}] must be 2D, got shape {arr.shape}")
        combos.extend(list(arr))
    return combos


def _parse_gt_string(gt: str) -> Tuple[str, np.ndarray]:
    """
    Parse a GT string of the form "'<prompt>'@path_a:wa,path_b:wb,...".
    Returns (prompt, weights array).
    """
    if "'@" not in gt:
        raise ValueError(f"Invalid GT format: {gt}")
    idx = gt.find("'@")
    prompt = gt[1:idx]
    rest = gt[idx + 2:]
    weights: List[float] = []
    for token in rest.split(","):
        if not token.strip():
            continue
        if ":" not in token:
            raise ValueError(f"Invalid weight token in GT: {token}")
        _, w_str = token.rsplit(":", 1)
        weights.append(float(w_str))
    return prompt, np.array(weights, dtype=float)


def load_prompt_weights_from_dir(input_dir: str) -> List[Tuple[str, np.ndarray, str]]:
    """
    Load prompt-weight combos from a directory of YAML files.
    Each file is expected to contain a YAML list with GT strings.
    Returns a list of (prompt, weights, raw_gt).
    """
    entries: List[Tuple[str, np.ndarray, str]] = []
    for name in sorted(os.listdir(input_dir)):
        if not name.endswith(".yml") and not name.endswith(".yaml"):
            continue
        if '_v0.yml' in str(name) or '_v0.yaml' in str(name):
            continue
        path = os.path.join(input_dir, name)
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if isinstance(data, list):
            candidates = [x for x in data if isinstance(x, str)]
        elif isinstance(data, str):
            candidates = [data]
        else:
            raise ValueError(f"Unsupported YAML content in {path}: expected list or string")

        for gt in candidates:
            prompt, weights = _parse_gt_string(gt)
            entries.append((prompt, weights, gt))

    if not entries:
        raise ValueError(f"No prompt-weight entries found in {input_dir}")
    return entries


def derive_distribution(
    entries: List[Tuple[str, np.ndarray, str]],
) -> Tuple[Dict[int, int], Dict[int, List[Tuple[int, Tuple[float, float]]]]]:
    """
    Derive target_pairs and high_sum_bins from existing prompt-weight entries.
    Bins are 1.0-wide: (1,2], (2,3], ...; sums <=1.0 are treated as low bin.
    """
    target = Counter()
    bin_counts: Dict[int, Counter[Tuple[float, float]]] = defaultdict(Counter)

    for _, weights, _ in entries:
        k = int(np.count_nonzero(weights > 1e-9))
        target[k] += 1
        total = float(weights.sum())
        if total > 1.0 + 1e-9:
            upper = int(np.ceil(total + 1e-9))
            lower = upper - 1.0
            bin_counts[k][(lower, float(upper))] += 1

    target_pairs = {k: v for k, v in sorted(target.items())}
    high_sum_bins: Dict[int, List[Tuple[int, Tuple[float, float]]]] = {}
    for k, ctr in bin_counts.items():
        high_sum_bins[k] = [(count, (low, high)) for (low, high), count in sorted(ctr.items(), key=lambda x: x[0][0])]

    return target_pairs, high_sum_bins


def _bin_label_for_weights(
    weights: np.ndarray,
    high_sum_bins: Dict[int, List[Tuple[int, Tuple[float, float]]]],
) -> Tuple[int, str]:
    """
    Categorize a weight vector by sparsity (k) and sum bin label.

    Returns (k, label) where label is one of:
      - a bin string like "(1.0,2.0]" when matched to a high-sum bin
      - "<=1.0" when no high-sum bin matches and sum <= 1.0
      - "other" as a fallback (should not happen for well-formed bins)
    """
    k = int(np.count_nonzero(weights))
    total = float(weights.sum())
    for _, (low, high) in high_sum_bins.get(k, []):
        if total > low - 1e-9 and total <= high + 1e-9:
            return k, f"({low:.2f},{high:.2f}]"
    if total <= 1.0 + 1e-9:
        return k, "<=1.0"
    return k, "other"


def _desired_counts_from_distribution(
    target_pairs: Dict[int, int],
    high_sum_bins: Dict[int, List[Tuple[int, Tuple[float, float]]]],
    total_needed: int,
) -> Dict[Tuple[int, str], int]:
    """
    Compute how many samples we want per (k, bin) to roughly match the global plan.

    Uses floor allocation then distributes the remainder by largest fractional part.
    """
    categories: List[Tuple[Tuple[int, str], float]] = []
    total_available = sum(target_pairs.values())
    if total_available <= 0:
        raise ValueError("Target pairs must be positive to build desired counts.")

    for k, total_k in target_pairs.items():
        bins = high_sum_bins.get(k, [])
        total_high = sum(count for count, _ in bins)
        low_count = max(total_k - total_high, 0)
        if low_count:
            categories.append(((k, "<=1.0"), low_count / total_available))
        for count, (low, high) in bins:
            label = f"({low:.2f},{high:.2f}]"
            categories.append(((k, label), count / total_available))

    desired: Dict[Tuple[int, str], int] = {}
    remainders: List[Tuple[float, Tuple[int, str]]] = []
    allocated = 0
    for cat, frac in categories:
        raw = total_needed * frac
        base = int(np.floor(raw))
        desired[cat] = base
        allocated += base
        remainders.append((raw - base, cat))

    # Distribute leftover slots by largest remainder
    remaining = total_needed - allocated
    remainders.sort(reverse=True)
    idx = 0
    while remaining > 0 and idx < len(remainders):
        _, cat = remainders[idx]
        desired[cat] += 1
        remaining -= 1
        idx += 1

    return desired


def _summarize_distribution(
    entries: List[Tuple[str, np.ndarray, str]],
    high_sum_bins: Dict[int, List[Tuple[int, Tuple[float, float]]]],
) -> Dict[str, Dict[str, int]]:
    """
    Summarize entries by sparsity k and bin label.
    Returns {"by_k": {k: count}, "by_bin": {"k|label": count}}.
    """
    by_k: Counter[int] = Counter()
    by_bin: Counter[str] = Counter()
    for _, w, _ in entries:
        k, label = _bin_label_for_weights(w, high_sum_bins)
        by_k[k] += 1
        by_bin[f"k{k}:{label}"] += 1
    return {
        "by_k": {str(k): v for k, v in sorted(by_k.items())},
        "by_bin": dict(sorted(by_bin.items())),
    }


def _select_session_entries(
    pool: List[Tuple[str, np.ndarray, str]],
    desired_counts: Dict[Tuple[int, str], int],
    total_needed: int,
    high_sum_bins: Dict[int, List[Tuple[int, Tuple[float, float]]]],
) -> Tuple[List[Tuple[str, np.ndarray, str]], List[Tuple[str, np.ndarray, str]], Dict[Tuple[int, str], int]]:
    """
    Greedily pick entries to match desired_counts per (k, bin),
    then fill any remaining slots with the earliest available.
    Returns (chosen, remaining_pool, actual_counts).
    """
    chosen_indices = set()
    actual: Dict[Tuple[int, str], int] = defaultdict(int)

    for idx, (_, w, _) in enumerate(pool):
        if len(chosen_indices) >= total_needed:
            break
        cat = _bin_label_for_weights(w, high_sum_bins)
        if actual[cat] < desired_counts.get(cat, 0):
            chosen_indices.add(idx)
            actual[cat] += 1

    if len(chosen_indices) < total_needed:
        for idx, _ in enumerate(pool):
            if len(chosen_indices) >= total_needed:
                break
            if idx in chosen_indices:
                continue
            chosen_indices.add(idx)
            cat = _bin_label_for_weights(pool[idx][1], high_sum_bins)
            actual[cat] += 1

    chosen = [pool[i] for i in sorted(chosen_indices)]
    remaining = [pool[i] for i in range(len(pool)) if i not in chosen_indices]
    return chosen, remaining, actual


def _normalize_desired_counts(
    desired_counts_in: Dict[object, int]
) -> Dict[Tuple[int, str], int]:
    """
    Normalize desired count keys that may come as strings like "k2:(1.00,2.00]" to (2, "(1.00,2.00]").
    """
    normalized: Dict[Tuple[int, str], int] = {}
    for key, val in desired_counts_in.items():
        if isinstance(key, tuple):
            normalized[key] = val
        else:
            # Expect format k{int}:{label}
            k_part, label = str(key).split(":", 1)
            if not k_part.startswith("k"):
                raise ValueError(f"Invalid desired key: {key}")
            k_val = int(k_part[1:])
            normalized[(k_val, label)] = val
    return normalized


def _sample_desired_set(
    pool: List[Tuple[str, np.ndarray, str]],
    desired_counts_in: Dict[object, int],
    high_sum_bins: Dict[int, List[Tuple[int, Tuple[float, float]]]],
    tag: str,
) -> Tuple[List[Tuple[str, np.ndarray, str]], List[Tuple[str, np.ndarray, str]], Dict[Tuple[int, str], int]]:
    """
    Sample a set matching desired counts (no replacement) and return chosen and remaining pool.
    """
    desired_counts = _normalize_desired_counts(desired_counts_in)
    total_needed = sum(desired_counts.values())
    chosen, remaining, actual = _select_session_entries(
        pool=pool,
        desired_counts=desired_counts,
        total_needed=total_needed,
        high_sum_bins=high_sum_bins,
    )
    if len(chosen) != total_needed:
        raise RuntimeError(
            f"[{tag}] Unable to sample required {total_needed} items; got {len(chosen)}."
        )
    return chosen, remaining, actual


def _assign_prompts_for_group(
    session_samples: List[Tuple[str, np.ndarray, str]],
    template_names: List[str],
    session_id: int,
    group_start_pid: int,
    group_size: int,
) -> Dict[int, Dict[str, Tuple[str, np.ndarray, str]]]:
    """
    Assign each sample to all three interfaces across distinct participants in a group.

    - Group is even-sized and internally paired.
    - Each sample is used exactly once per interface across the group.
    - No participant sees the same prompt twice within a session.
    """
    if group_size % 2 != 0:
        raise ValueError("Group size must be even to build pairs.")
    pairs_count = group_size // 2
    if len(session_samples) < pairs_count:
        raise ValueError("Need at least as many session samples as pairs.")

    assignments: Dict[int, Dict[str, Tuple[str, np.ndarray, str]]] = {
        pid: {iface: None for iface in template_names}
        for pid in range(group_start_pid, group_start_pid + group_size)
    }
    used_prompts: Dict[int, set] = {pid: set() for pid in assignments.keys()}

    for s_idx, sample in enumerate(session_samples):
        for iface_idx, iface in enumerate(template_names):
            pair_idx = (s_idx + iface_idx) % pairs_count
            p1 = group_start_pid + 2 * pair_idx
            p2 = group_start_pid + 2 * pair_idx + 1
            first = p1 if (session_id + s_idx + iface_idx) % 2 == 0 else p2
            second = p2 if first == p1 else p1

            target_pid = None
            if sample[0] not in used_prompts[first] and assignments[first][iface] is None:
                target_pid = first
            elif sample[0] not in used_prompts[second] and assignments[second][iface] is None:
                target_pid = second
            else:
                raise RuntimeError(
                    f"Cannot place sample '{sample[0]}' for interface {iface} in session {session_id} within group starting at {group_start_pid}."
                )

            assignments[target_pid][iface] = sample
            used_prompts[target_pid].add(sample[0])

    # Verify fill and uniqueness
    for pid, iface_map in assignments.items():
        for iface, value in iface_map.items():
            if value is None:
                raise RuntimeError(
                    f"Missing assignment for participant {pid}, interface {iface} in session {session_id}."
                )
        prompts_seen = [iface_map[iface][0] for iface in template_names]
        if len(set(prompts_seen)) != len(prompts_seen):
            raise RuntimeError(
                f"Participant {pid} has duplicate prompts within session {session_id}."
            )

    return assignments


def _verify_assignments(
    assignments_by_session: Dict[int, Dict[int, Dict[str, Tuple[str, np.ndarray, str]]]],
    template_names: List[str],
    participants_count: int,
):
    """
    Auto verification of assignment constraints:
      1) Each prompt is covered by all three interfaces exactly once.
      2) Each prompt appears in only one (session, group) pair (group A=1..half, B=half+1..end).
      3) Consecutive participant pairs (1&2, 3&4, ...) share no prompts within a session.
      4) No participant sees the same prompt twice in a session.
    """

    half_point = participants_count // 2
    prompt_meta: Dict[str, Dict[str, object]] = {}

    for session_id, pid_map in assignments_by_session.items():
        # Pair-level uniqueness
        pairs = [(2 * i + 1, 2 * i + 2) for i in range(half_point)]

        for pid, iface_map in pid_map.items():
            group_tag = "A" if pid <= half_point else "B"
            seen_prompts_pid: set = set()
            for iface, (prompt, _w, _gt) in iface_map.items():
                meta = prompt_meta.setdefault(prompt, {"ifaces": set(), "count": 0, "groups": set()})
                meta["ifaces"].add(iface)
                meta["count"] += 1
                meta["groups"].add((session_id, group_tag))

                if prompt in seen_prompts_pid:
                    raise RuntimeError(
                        f"Participant {pid} sees prompt '{prompt}' more than once in session {session_id}."
                    )
                seen_prompts_pid.add(prompt)

        for p1, p2 in pairs:
            prompts1 = {v[0] for v in pid_map[p1].values()}
            prompts2 = {v[0] for v in pid_map[p2].values()}
            if prompts1 & prompts2:
                overlap = prompts1 & prompts2
                raise RuntimeError(
                    f"Participants {p1} and {p2} share prompts {overlap} in session {session_id}."
                )

    # Global checks per prompt
    for prompt, meta in prompt_meta.items():
        if meta["count"] != len(template_names) or len(meta["ifaces"]) != len(template_names):
            raise RuntimeError(
                f"Prompt '{prompt}' expected in {len(template_names)} interfaces, saw {meta['count']} occurrences across {meta['ifaces']}"
            )
        if len(meta["groups"]) != 1:
            raise RuntimeError(
                f"Prompt '{prompt}' appears in multiple session/group combos: {meta['groups']}"
            )


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


def parse_ports(ports_arg: Optional[str], participants_count: int, requested_par: Optional[int]) -> List[int]:
    """Parse a comma-separated ports string or generate a default range.

    If ports are omitted, use the requested participant count (par) when provided;
    otherwise fall back to the inferred participant count, assigning sequential
    ports starting at 8000.
    """
    if participants_count <= 0:
        raise ValueError(
            "participants_count must be positive to assign ports.")

    default_count = requested_par if requested_par is not None else participants_count
    if default_count <= 0:
        raise ValueError("Participant count must be positive to assign ports.")

    if ports_arg is None:
        return [8000 + i for i in range(default_count)]

    parts = [p.strip() for p in ports_arg.split(",") if p.strip()]
    try:
        ports = [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError(
            "--ports must be a comma-separated list of integers.") from exc

    if len(ports) != default_count:
        raise ValueError(
            f"--ports must provide exactly {default_count} entries; got {len(ports)}."
        )

    return ports


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


def _balanced_orders_three(names: List[str]) -> List[List[str]]:
    """Balanced Latin square orders for 3 interfaces (size=3, 6 rows)."""
    if len(names) != 3:
        raise ValueError(
            "Balanced ordering is implemented for exactly 3 interfaces.")
    a, b, c = names
    return [
        [a, b, c],
        [b, c, a],
        [c, a, b],
        [a, c, b],
        [c, b, a],
        [b, a, c],
    ]


def build_task_plan(
    participants: int,
    template_specs: List[Tuple[str, str]],
    seed: int,
) -> Tuple[List[Dict[str, int]], Dict[str, List[List[str]]]]:
    names = [name for name, _ in template_specs]
    orders = _balanced_orders_three(names)

    plan: List[Dict[str, int]] = []
    debug_assignment: Dict[str, List[List[str]]] = {}

    for pid in range(1, participants + 1):
        pid_key = f"P{pid:02d}"
        debug_assignment[pid_key] = []
        for session_id in range(1, SESSIONS_PER_PARTICIPANT + 1):
            order = orders[(pid - 1 + (session_id - 1)) % len(orders)]
            debug_assignment[pid_key].append(order)
            for task_in_session, name in enumerate(order, start=1):
                plan.append(
                    {
                        "participant": pid,
                        "session": session_id,
                        "task_in_session": task_in_session,
                        "template_name": name,
                    }
                )

    return plan, debug_assignment


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
        path_with_ext = path if path.endswith(
            ".safetensors") else f"{path}.safetensors"
        parts.append(f"{path_with_ext}:{w_str}")
    weights_str = ",".join(parts)

    return f"'{prompt}'@{weights_str}"


def main(
    classes_json: Optional[str],
    models_yml: Optional[str],
    out_dir: str,
    templates: List[str],
    participants: Optional[int] = None,
    tutorial: bool = False,
    offset: int = 0,
    shuffle_weights: bool = True,
    seed: int = 0,
    ports: Optional[str] = None,
    input_dir: Optional[str] = None,
):
    os.makedirs(out_dir, exist_ok=True)

    template_specs = parse_template_args(templates)
    template_data = load_templates(template_specs)

    rng = random.Random(seed)

    if input_dir:
        base_entries = load_prompt_weights_from_dir(input_dir)
        if shuffle_weights and len(base_entries) > 1:
            rng.shuffle(base_entries)
        entries_with_gt = base_entries
        target_pairs, high_sum_bins = {}, {}
        model_paths = []
    else:
        if not classes_json or not models_yml:
            raise ValueError("--classes-json and --models-yml are required when --input-dir is not provided.")

        model_paths = load_model_paths(models_yml)
        prompts = build_prompts_from_classes(classes_json)

        dim = len(model_paths)
        if dim == 0:
            raise ValueError("models_yml must contain at least one model path.")

        num_prompts = len(prompts)
        if num_prompts == 0:
            raise ValueError(
                "No prompts generated from classes_json; nothing to do.")

        # Fixed recipe for pilot: predefined sparsity counts and some high-sum samples
        target_pairs = {
            2: 11,
            3: 8,
            4: 7,
            5: 4,
        }

        # Example bins per k: list of (count, (low, high])
        high_sum_bins = {
            2: [(1, (0.00, 1.00)), (10, (1.00, 2.00))],
            3: [(3, (1.00, 2.00)), (5, (2.00, 3.00))],
            4: [(2, (1.00, 2.00)), (4, (2.00, 3.00)), (1, (3.00, 4.00))],
            5: [(1, (1.00, 2.00)), (2, (2.00, 3.00)), (1, (3.00, 4.00))],
        }

        weight_sets = generate_weight_combinations(
            total_dim=dim,
            target_pairs=target_pairs,
            seed=0,
            max_units=100,
            min_value=0.2,
            # min_value=0.1,
            high_sum_bins=high_sum_bins,
        )

        combos = flatten_weight_sets(weight_sets)
        if shuffle_weights and len(combos) > 1:
            np.random.default_rng(seed).shuffle(combos)
        entries_with_gt = [(p, w, build_gt_string(p, model_paths, w)) for p, w in zip(prompts, combos)]

    output_parent = os.path.dirname(os.path.abspath(out_dir))

    # Common prompt filter keywords (skip any prompt containing one of these substrings)
    # filter_keywords = ["rose", "woman", "cat", "bottle", "boy"]
    # filter_keywords = ["rose"]
    filter_keywords = []

    def is_filtered(prompt: str) -> bool:
        lower = prompt.lower()
        return any(kw in lower for kw in filter_keywords)

    # Pair prompts and weights once, shuffle, then filter.
    paired_entries = entries_with_gt
    if shuffle_weights and len(paired_entries) > 1:
        rng.shuffle(paired_entries)
    filtered_entries_raw = [(p, w, gt) for p, w, gt in paired_entries if not is_filtered(p)]

    # Deduplicate by raw GT to avoid reusing the same combo across tasks
    filtered_entries: List[Tuple[str, np.ndarray, str]] = []
    seen_gt: set = set()
    for entry in filtered_entries_raw:
        if entry[2] in seen_gt:
            continue
        seen_gt.add(entry[2])
        filtered_entries.append(entry)

    if input_dir:
        target_pairs, high_sum_bins = derive_distribution(filtered_entries)
        dist_summary = _summarize_distribution(filtered_entries, high_sum_bins)
        print("# Derived distribution from input-dir")
        print(f"target_pairs={target_pairs}")
        print(f"high_sum_bins={high_sum_bins}")
        print(f"by_k={dist_summary['by_k']}")
        print(f"by_bin={dist_summary['by_bin']}")

    if tutorial:
        if participants is None:
            participants_count = 1
        else:
            if participants <= 0:
                raise ValueError("--par must be positive when provided.")
            participants_count = participants

        if offset < 0:
            raise ValueError("--offset must be non-negative.")

        # Tutorial should consume prompts after skipping the regular portion.
        tutorial_pool = filtered_entries[offset:]

        if len(tutorial_pool) < len(template_specs):
            raise ValueError(
                f"Not enough prompts after filtering and offset to cover all templates. "
                f"Needed {len(template_specs)}, found {len(tutorial_pool)} after skipping {offset}."
            )

        chosen = tutorial_pool[: len(template_specs)]
        interface_gt = {name: pwg for (
            name, _), pwg in zip(template_specs, chosen)}

        # Tutorial: single session, fixed interface order for all participants
        total_needed = participants_count * len(template_specs)

        tutorial_order = template_specs
        slider_first = [
            tpl for tpl in template_specs if tpl[0].lower() == "slider"]
        if slider_first:
            others = [tpl for tpl in template_specs if tpl[0].lower()
                      != "slider"]
            tutorial_order = slider_first + others

        tasks_info = []

        for pid in range(1, participants_count + 1):
            p_tag = f"P{pid:02d}_tut"
            pid_plain = p_tag.split("_")[0]
            s_tag = "S1"
            for task_idx, (iface_name, _) in enumerate(tutorial_order, start=1):
                prompt, weights, gt_raw = interface_gt[iface_name]
                t_tag = f"T{task_idx:02d}"
                task_label = f"{p_tag}_{s_tag}_{t_tag}_{iface_name}"

                output_dir_value = os.path.join(
                    output_parent, p_tag, f"{s_tag}_{t_tag}")
                init_dir_value = os.path.join(
                    output_parent, p_tag, f"init_{s_tag}_{t_tag}")
                os.makedirs(output_dir_value, exist_ok=True)
                os.makedirs(init_dir_value, exist_ok=True)

                cfg = deepcopy(template_data[iface_name])
                cfg['output_dir'] = output_dir_value
                cfg['init_dir'] = init_dir_value
                cfg['gt_config'] = gt_raw

                filename = os.path.join(out_dir, f"{task_label}.yml")
                with open(filename, "w") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False)

                tasks_info.append({
                    "pid_tag": p_tag,
                    "pid_plain": pid_plain,
                    "s_tag": s_tag,
                    "t_tag": t_tag,
                    "iface": iface_name,
                    "config_path": filename,
                    "output_dir": output_dir_value,
                    "init_dir": init_dir_value,
                })

                print(f"Wrote {filename} | prompt: {prompt}")

        ports_list = parse_ports(ports, participants_count, participants)

        clean_cmds = []
        precompute_cmds = []
        demo_cmds = []
        run_cmds = []

        for info in tasks_info:
            clean_cmds.append(
                f"rm -rf {info['init_dir']}/* {info['output_dir']}/*"
            )
            precompute_cmds.append(
                f"python ./server_{info['iface']}.py --config {info['config_path']} --precompute"
            )

        # Demographic command per participant, using the first tutorial task (S1_T01)
        pid_to_first_task = {}
        for info in tasks_info:
            if info['pid_plain'] not in pid_to_first_task and info['t_tag'] == "T01":
                pid_to_first_task[info['pid_plain']] = info

        for idx, pid_plain in enumerate(sorted(pid_to_first_task.keys())):
            task = pid_to_first_task[pid_plain]
            port = ports_list[idx]
            demo_cmds.append(
                """
PORT={port}; PID={pid}; \
python ./server_{iface}.py --port ${{PORT}} --demographic ${{PID}} --config {cfg} --ssh >server_${{PID}}.log 2>&1
""".strip().format(port=port, pid=pid_plain, iface=task['iface'], cfg=task['config_path'])
            )

        # Actual session run commands for all tutorial tasks
        pid_to_port = {f"P{idx+1:02d}": port for idx,
                       port in enumerate(ports_list)}
        for info in tasks_info:
            pid_key = info['pid_plain']
            port = pid_to_port.get(pid_key, ports_list[0])
            run_cmds.append(
                """
PORT={port}; STATE_DIR={state_dir}; LOG_DIR=.; \
STATE_PATH="${{STATE_DIR}}/{pid_tag}/{pid_tag}_{s}_{iface}_{t}"; LOG_PATH="${{LOG_DIR}}/server_{pid}_{s}_{iface}_{t}.log"; \
python "./server_{iface}.py" --port "${{PORT}}" --save-state-dir "${{STATE_PATH}}" --ssh --config {cfg} >"${{LOG_PATH}}" 2>&1
""".strip().format(
                    pid=pid_key,
                    s=info['s_tag'],
                    iface=info['iface'],
                    t=info['t_tag'],
                    port=port,
                    state_dir=output_parent,
                    pid_tag=info['pid_tag'],
                    cfg=info['config_path'],
                )
            )

        print(
            f"Done. Tutorial mode wrote {total_needed} config files for {participants_count} participants to: {out_dir}"
        )

        print("\n# Clean generated folders")
        for cmd in clean_cmds:
            print(cmd)

        print("\n# Precompute commands")
        for cmd in precompute_cmds:
            print(cmd)

        print("\n# Demographic commands")
        for cmd in demo_cmds:
            print(cmd)

        print("\n# Tutorial session run commands")
        for cmd in run_cmds:
            print(cmd)

        return

    # Study mode: support even participant counts (default 12) and 6 prompts per session
    SAMPLES_PER_SESSION = 6
    if participants is None:
        participants_count = 12
    else:
        if participants <= 0 or participants % 2 != 0:
            raise ValueError("Participant count must be positive and even.")
        participants_count = participants

    if participants_count % 2 != 0:
        raise ValueError("Participant count must be even.")

    total_unique_needed = SESSIONS_PER_PARTICIPANT * SAMPLES_PER_SESSION
    if len(filtered_entries) < total_unique_needed:
        raise ValueError(
            f"Not enough prompts after filtering to build the task plan. "
            f"Needed {total_unique_needed}, found {len(filtered_entries)} after filtering."
        )

    base_desired_counts = _desired_counts_from_distribution(
        target_pairs=target_pairs,
        high_sum_bins=high_sum_bins,
        total_needed=SAMPLES_PER_SESSION,
    )

    # Explicit desired distributions
    desired_main = {
        "k2:(1.00,2.00]": 2,
        "k3:(1.00,2.00]": 1,
        "k3:(2.00,3.00]": 1,
        "k4:(2.00,3.00]": 1,
        "k5:(2.00,3.00]": 1,
    }
    desired_no_k5 = {
        "k2:(1.00,2.00]": 3,
        "k3:(1.00,2.00]": 1,
        "k3:(2.00,3.00]": 1,
        "k4:(2.00,3.00]": 1,
    }

    session_samples_main: List[List[Tuple[str, np.ndarray, str]]] = []
    session_samples_no_k5: List[List[Tuple[str, np.ndarray, str]]] = []
    pool_all = list(filtered_entries)

    session_debug: List[Dict[str, object]] = []

    # Session 1 main
    s1_main, pool_all, act_s1_main = _sample_desired_set(
        pool=pool_all,
        desired_counts_in=desired_main,
        high_sum_bins=high_sum_bins,
        tag="session1_main",
    )
    session_samples_main.append(s1_main)

    # Session 1 no-k5 (from remaining, with k5 removed)
    pool_no_k5 = [e for e in pool_all if int(np.count_nonzero(e[1] > 1e-9)) != 5]
    s1_no, pool_no_k5, act_s1_no = _sample_desired_set(
        pool=pool_no_k5,
        desired_counts_in=desired_no_k5,
        high_sum_bins=high_sum_bins,
        tag="session1_no_k5",
    )
    session_samples_no_k5.append(s1_no)
    # Remove chosen no-k5 from global pool
    used_s1_no = {e[2] for e in s1_no}
    pool_all = [e for e in pool_all if e[2] not in used_s1_no]

    # Session 2 main from remaining pool
    s2_main, pool_all, act_s2_main = _sample_desired_set(
        pool=pool_all,
        desired_counts_in=desired_main,
        high_sum_bins=high_sum_bins,
        tag="session2_main",
    )
    session_samples_main.append(s2_main)

    # Session 2 no-k5 from remaining pool (k5 removed)
    pool_no_k5 = [e for e in pool_all if int(np.count_nonzero(e[1] > 1e-9)) != 5]
    s2_no, pool_no_k5, act_s2_no = _sample_desired_set(
        pool=pool_no_k5,
        desired_counts_in=desired_no_k5,
        high_sum_bins=high_sum_bins,
        tag="session2_no_k5",
    )
    session_samples_no_k5.append(s2_no)
    used_s2_no = {e[2] for e in s2_no}
    pool_all = [e for e in pool_all if e[2] not in used_s2_no]

    session_debug.append(
        {
            "session": 1,
            "desired_counts_main": _normalize_desired_counts(desired_main),
            "actual_counts_main": act_s1_main,
            "distribution_main": _summarize_distribution(s1_main, high_sum_bins),
            "desired_counts_no_k5": _normalize_desired_counts(desired_no_k5),
            "actual_counts_no_k5": act_s1_no,
            "distribution_no_k5": _summarize_distribution(s1_no, high_sum_bins),
        }
    )
    session_debug.append(
        {
            "session": 2,
            "desired_counts_main": _normalize_desired_counts(desired_main),
            "actual_counts_main": act_s2_main,
            "distribution_main": _summarize_distribution(s2_main, high_sum_bins),
            "desired_counts_no_k5": _normalize_desired_counts(desired_no_k5),
            "actual_counts_no_k5": act_s2_no,
            "distribution_no_k5": _summarize_distribution(s2_no, high_sum_bins),
        }
    )

    template_names = [name for name, _ in template_specs]
    task_plan, debug_assignment = build_task_plan(
        participants_count, template_specs, seed)

    assignments_by_session: Dict[int, Dict[int, Dict[str, Tuple[str, np.ndarray, str]]]] = {}
    half_point = participants_count // 2
    for session_id in range(1, SESSIONS_PER_PARTICIPANT + 1):
        assignments_by_session[session_id] = {}
        # Group A: first half participants use main samples
        assign_main = _assign_prompts_for_group(
            session_samples=session_samples_main[session_id - 1],
            template_names=template_names,
            session_id=session_id,
            group_start_pid=1,
            group_size=half_point,
        )
        # Group B: second half participants use no-k5 samples
        assign_nk5 = _assign_prompts_for_group(
            session_samples=session_samples_no_k5[session_id - 1],
            template_names=template_names,
            session_id=session_id,
            group_start_pid=half_point + 1,
            group_size=half_point,
        )

        # Merge with shifted participant ids
        for pid, payload in assign_main.items():
            assignments_by_session[session_id][pid] = payload
        for pid, payload in assign_nk5.items():
            assignments_by_session[session_id][pid] = payload

    _verify_assignments(assignments_by_session, template_names, participants_count)

    tasks_info = []
    for task in task_plan:
        prompt, weights, gt_raw = assignments_by_session[task['session']][task['participant']][task['template_name']]
        p_tag = f"P{task['participant']:02d}"
        s_tag = f"S{task['session']}"
        t_tag = f"T{task['task_in_session']:02d}"
        task_label = f"{p_tag}_{s_tag}_{t_tag}_{task['template_name']}"

        output_dir_value = os.path.join(
            output_parent, p_tag, f"{s_tag}_{t_tag}")
        init_dir_value = os.path.join(
            output_parent, p_tag, f"init_{s_tag}_{t_tag}")
        os.makedirs(output_dir_value, exist_ok=True)
        os.makedirs(init_dir_value, exist_ok=True)

        cfg = deepcopy(template_data[task['template_name']])
        cfg['output_dir'] = output_dir_value
        cfg['init_dir'] = init_dir_value
        cfg['gt_config'] = gt_raw

        filename = os.path.join(out_dir, f"{task_label}.yml")
        with open(filename, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        tasks_info.append({
            "pid_tag": p_tag,
            "pid_plain": p_tag,
            "s_tag": s_tag,
            "t_tag": t_tag,
            "iface": task['template_name'],
            "config_path": filename,
            "output_dir": output_dir_value,
            "init_dir": init_dir_value,
        })

        print(f"Wrote {filename} | prompt: {prompt}")

    ports_list = parse_ports(ports, participants_count, participants)

    clean_cmds = []
    precompute_cmds = []
    run_cmds = []

    for info in tasks_info:
        clean_cmds.append(
            f"rm -rf {info['init_dir']}/* {info['output_dir']}/*"
        )
        precompute_cmds.append(
            f"python ./server_{info['iface']}.py --config {info['config_path']} --precompute"
        )

    pid_to_port = {f"P{idx+1:02d}": port for idx,
                   port in enumerate(ports_list)}
    for info in tasks_info:
        port = pid_to_port.get(info['pid_plain'], ports_list[0])
        run_cmds.append(
            """
PORT={port}; STATE_DIR={state_dir}; LOG_DIR=.; \
STATE_PATH="${{STATE_DIR}}/{pid_tag}/{pid_tag}_{s}_{iface}_{t}"; LOG_PATH="${{LOG_DIR}}/server_{pid}_{s}_{iface}_{t}.log"; \
python "./server_{iface}.py" --port "${{PORT}}" --save-state-dir "${{STATE_PATH}}" --ssh --config {cfg} >"${{LOG_PATH}}" 2>&1
""".strip().format(
                pid=info['pid_plain'],
                s=info['s_tag'],
                iface=info['iface'],
                t=info['t_tag'],
                port=port,
                state_dir=output_parent,
                pid_tag=info['pid_tag'],
                cfg=info['config_path'],
            )
        )

    total_configs = len(tasks_info)
    print(
        f"Done. Wrote {total_configs} config files for {participants_count} participants to: {out_dir}"
    )

    print("\n# Session sample distributions")
    for entry in session_debug:
        print(f"Session {entry['session']} (main): desired={entry['desired_counts_main']}, actual={entry['actual_counts_main']}, by_k={entry['distribution_main']['by_k']}, by_bin={entry['distribution_main']['by_bin']}")
        print(f"Session {entry['session']} (no k5): desired={entry['desired_counts_no_k5']}, actual={entry['actual_counts_no_k5']}, by_k={entry['distribution_no_k5']['by_k']}, by_bin={entry['distribution_no_k5']['by_bin']}")

    print("\n# Final assignments (prompt per participant/session/interface)")
    for session_id in sorted(assignments_by_session.keys()):
        for pid in sorted(assignments_by_session[session_id].keys()):
            iface_summaries = []
            for iface in template_names:
                prompt, weights, _ = assignments_by_session[session_id][pid][iface]
                k, label = _bin_label_for_weights(weights, high_sum_bins)
                iface_summaries.append(
                    f"{iface}: k{k} {label} '{prompt}'"
                )
            group_tag = "A" if pid <= (participants_count // 2) else "B"
            print(f"P{pid:02d} (G{group_tag}) S{session_id}: " + " | ".join(iface_summaries))

    print("\n# Interface order (balanced Latin square)")
    for pid_key, orders in debug_assignment.items():
        for session_idx, order in enumerate(orders, start=1):
            print(f"{pid_key} session {session_idx}: {', '.join(order)}")

    print("\n# Clean generated folders")
    for cmd in clean_cmds:
        print(cmd)

    print("\n# Precompute commands")
    for cmd in precompute_cmds:
        print(cmd)

    print("\n# Session run commands")
    for cmd in run_cmds:
        print(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate test input config files from weight combos, class JSON, and model list."
    )
    parser.add_argument("--classes-json", required=False,
                        help="Path to JSON with people/animals/objects classes (required unless --input-dir is provided).")
    parser.add_argument("--models-yml", required=False,
                        help="Path to YAML with model file paths (required unless --input-dir is provided).")
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
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="When --tutorial is set, skip this many filtered prompt-weight pairs (e.g., the number consumed by the regular run) before selecting tutorial prompts.",
    )
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Do not shuffle generated weight combinations.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for generation and shuffling.")
    parser.add_argument(
        "--ports",
        type=str,
        default=None,
        help="Comma-separated list of ports (one per participant). Defaults to sequential ports starting at 8000 if omitted.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Input directory for pre-generated config .yml files.",
    )

    args = parser.parse_args()
    main(
        classes_json=args.classes_json,
        models_yml=args.models_yml,
        out_dir=args.out_dir,
        templates=args.templates,
        participants=args.par,
        tutorial=args.tutorial,
        offset=args.offset,
        shuffle_weights=not args.no_shuffle,
        seed=args.seed,
        ports=args.ports,
        input_dir=args.input_dir,
    )
