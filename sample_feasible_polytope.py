#!/usr/bin/env python
"""
Sample the feasible alloy polytope on a 5 at% grid with constraints:
- x_i >= 0
- sum x_i = 1
- density <= rho_max
- Pugh_Ratio_PRIOR > pugh_min (via V* linear bound)
- ROM melting temperature >= tm_min
- Exclude element systems pruned by Curtin-Maresca upper-bound analysis

Outputs CSV of feasible compositions.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import re
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd

PROP_SOURCE = Path(__file__).with_name("1_find_stoich_props_parallel_for_Class.py")
PRUNE_SCRIPT = Path(__file__).with_name("prune_systems_ys_1300C_density.py")


def load_vstar_from_prop_source():
    src = PROP_SOURCE.read_text()
    m = re.search(r"elast_data\s*=\s*(\{[\s\S]*?\n\s*\})", src)
    if not m:
        raise RuntimeError("elast_data dict not found in 1_find_stoich_props_parallel_for_Class.py")
    elast_data = ast.literal_eval(m.group(1))
    vstar = {el: props["V*"] for el, props in elast_data.items()}
    return vstar


def load_prop_data_from_prop_source():
    src = PROP_SOURCE.read_text()
    m = re.search(r"prop_data\s*=\s*(\{[\s\S]*?\n\s*\})", src)
    if not m:
        raise RuntimeError("prop_data dict not found in 1_find_stoich_props_parallel_for_Class.py")
    prop_data = ast.literal_eval(m.group(1))
    return prop_data


def v_threshold_from_pugh(pugh_min: float) -> float:
    return ((1.5 * pugh_min) - 1.0) / (1.0 + 3.0 * pugh_min)


def load_prune_module():
    spec = importlib.util.spec_from_file_location("prune_mod", PRUNE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pruned_systems(elements, rho_max, ys_target_mpa=400.0):
    mod = load_prune_module()
    rows = []
    for r in range(2, len(elements) + 1):
        for subset in combinations(elements, r):
            rho = np.array([mod.load_densities_from_prop_source()[el] for el in subset], dtype=float)
            sigma_ub = mod.prune_impossible_element_set(subset, rho, rho_max)
            if sigma_ub < ys_target_mpa:
                rows.append("-".join(subset))
    return set(rows)


def compositions_integer_sum(n, total):
    """Yield integer compositions of length n that sum to total."""
    if n == 1:
        yield (total,)
        return
    for i in range(total + 1):
        for tail in compositions_integer_sum(n - 1, total - i):
            yield (i,) + tail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-atpct", type=float, default=5.0, help="Grid step in at% (default 5)")
    parser.add_argument("--rho-max", type=float, default=9.5, help="Max allowed density (g/cc)")
    parser.add_argument("--pugh-min", type=float, default=2.5, help="Minimum Pugh_Ratio_PRIOR")
    parser.add_argument("--tm-min-c", type=float, default=2086.0, help="Minimum ROM melting temperature (C)")
    parser.add_argument("--out", type=str, default="feasible_samples_5atpct.csv")
    args = parser.parse_args()

    step = args.step_atpct / 100.0
    if abs(round(1.0 / step) - (1.0 / step)) > 1e-12:
        raise RuntimeError("Step must divide 100 exactly (e.g., 5, 2, 1).")
    total = int(round(1.0 / step))

    elements = ["Nb", "Ta", "V", "Cr", "Mo", "Ti", "W"]
    vstar_map = load_vstar_from_prop_source()
    prop_data = load_prop_data_from_prop_source()

    vstar = np.array([vstar_map[e] for e in elements], dtype=float)
    rho = np.array([prop_data[e]["Density [g/cm^3]"] for e in elements], dtype=float)
    tms = np.array([prop_data[e]["Melting Temperature [K]"] for e in elements], dtype=float)

    v_min = v_threshold_from_pugh(args.pugh_min)
    tm_min_k = args.tm_min_c + 273.15

    pruned = pruned_systems(elements, args.rho_max, ys_target_mpa=400.0)

    rows = []
    for comp_int in compositions_integer_sum(len(elements), total):
        if sum(comp_int) != total:
            continue
        x = np.array(comp_int, dtype=float) / total

        # active element set pruning
        active = [elements[i] for i, val in enumerate(x) if val > 0]
        if len(active) >= 2:
            key = "-".join(active)
            if key in pruned:
                continue

        # linear constraints
        if (rho @ x) > args.rho_max + 1e-12:
            continue
        if (vstar @ x) < v_min - 1e-12:
            continue
        if (tms @ x) < tm_min_k - 1e-12:
            continue

        row = {elements[i]: x[i] for i in range(len(elements))}
        row["density"] = float(rho @ x)
        row["V_avgr"] = float(vstar @ x)
        row["Tm_avg_K"] = float(tms @ x)
        row["active_system"] = "-".join(active) if active else ""
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print("Feasible sample grid")
    print(f"Elements: {', '.join(elements)}")
    print(f"Step: {args.step_atpct:.1f} at% (grid total {total})")
    print(f"rho_max: {args.rho_max:.3f} g/cc")
    print(f"Pugh min: {args.pugh_min:.3f} (V_avgr >= {v_min:.6f})")
    print(f"Tm_min: {tm_min_k:.2f} K ({args.tm_min_c:.1f} C)")
    print(f"Pruned systems: {len(pruned)}")
    print(f"Feasible compositions: {len(df)}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
