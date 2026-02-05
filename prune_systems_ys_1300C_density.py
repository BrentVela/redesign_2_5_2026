#!/usr/bin/env python
"""
Prune Nb–Ta–V–Cr–Mo–Ti–W sub-systems using a safe upper bound
on YS at 1300C (1573 K) with an optional density constraint.

This uses the Curtin-Maresca model structure from strength_model.py and
computes a conservative upper bound by:
- maximizing shear modulus and Poisson factor via element-wise extrema
- maximizing misfit variance within the density constraint
- scanning Burgers vector bounds from feasible Vbar range

Outputs CSV with sub-system, sigma_y_UB, and prunable flag.
"""
from __future__ import annotations

import argparse
import ast
import itertools
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

# ---- elemental data from strength_model.py ----
elast_data = {
    'W': {'C11': 517.8, 'C12': 201.7, 'C44': 139.4},
    'Mo': {'C11': 466, 'C12': 165.2, 'C44': 99.5},
    'Ta': {'C11': 260.9, 'C12': 165.2, 'C44': 70.4},
    'Nb': {'C11': 247.2, 'C12': 140, 'C44': 14.2},
    'V': {'C11': 272, 'C12': 144.8, 'C44': 17.6},
    'Ti': {'C11': 95.9, 'C12': 115.9, 'C44': 40.3},
    'Cr': {'C11': 247.6, 'C12': 73.4, 'C44': 48.3},
}

volumes = {
    'W': 16.229,
    'Mo': 15.956,
    'Ta': 18.313,
    'Nb': 18.342,
    'V': 13.453,
    'Ti': 17.123,
    'Cr': 11.575,
}

# ---- model constants (match strength_model.py + prune method) ----
alpha = 1/12
ln_eps_ratio = np.log(1e4/1e-3)
M_max = 3.1
nu_cap = 0.40
k_B_eV = 8.617333262e-5
T_K = 1573.0
conv_to_eV = 1/160.2176621
YS_TARGET_MPA = 400.0

PROP_SOURCE = Path(__file__).with_name("1_find_stoich_props_parallel_for_Class.py")


def load_densities_from_prop_source():
    src = PROP_SOURCE.read_text()
    m = re.search(r"prop_data\s*=\s*(\{[\s\S]*?\n\s*\})", src)
    if not m:
        raise RuntimeError("prop_data dict not found in 1_find_stoich_props_parallel_for_Class.py")
    prop_data = ast.literal_eval(m.group(1))
    densities = {el: props['Density [g/cm^3]'] for el, props in prop_data.items()}
    return densities


def b_from_V(Vbar: float) -> float:
    a = (2 * Vbar) ** (1/3)
    return (math.sqrt(3)/2) * a


def vbar_bounds(elements, rho, rho_max):
    """LP for min/max Vbar under density and simplex constraints."""
    V = np.array([volumes[el] for el in elements], dtype=float)
    n = len(elements)
    bounds = [(0.0, 1.0) for _ in range(n)]
    A_eq = [np.ones(n)]
    b_eq = [1.0]
    A_ub = [rho]
    b_ub = [rho_max]

    res_min = linprog(V, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    res_max = linprog(-V, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if not res_min.success or not res_max.success:
        return None, None
    return float(res_min.fun), float(-res_max.fun)


def gamma_max_density(elements, rho, rho_max):
    """Maximize misfit variance under density + simplex constraints (concave objective)."""
    V = np.array([volumes[el] for el in elements], dtype=float)
    n = len(elements)

    def variance(x):
        Vbar = float(x @ V)
        return float(np.sum(x * (V - Vbar) ** 2))

    def obj(x):
        return -variance(x)

    cons = [
        {"type": "eq", "fun": lambda x: np.sum(x) - 1.0},
        {"type": "ineq", "fun": lambda x: rho_max - float(x @ rho)},
    ]
    bounds = [(0.0, 1.0) for _ in range(n)]

    # Initial points: feasible vertices and uniform over feasible elements
    starts = []
    for i in range(n):
        if rho[i] <= rho_max + 1e-12:
            x = np.zeros(n)
            x[i] = 1.0
            starts.append(x)

    if not starts:
        return None

    # uniform over feasible elements
    feas = [i for i in range(n) if rho[i] <= rho_max + 1e-12]
    if feas:
        x = np.zeros(n)
        x[feas] = 1.0 / len(feas)
        starts.append(x)

    best = None
    for x0 in starts:
        res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons)
        if res.success:
            v = variance(res.x)
            if best is None or v > best:
                best = v

    return best


def prune_impossible_element_set(elements, rho, rho_max):
    C11 = np.array([elast_data[el]['C11'] for el in elements])
    C12 = np.array([elast_data[el]['C12'] for el in elements])
    C44 = np.array([elast_data[el]['C44'] for el in elements])
    V   = np.array([volumes[el] for el in elements])

    C11_max, C12_min, C44_max = C11.max(), C12.min(), C44.max()
    mu_max = math.sqrt(0.5 * C44_max * (C11_max - C12_min))
    nu_max = nu_cap
    f_max = (1 + nu_max) / (1 - nu_max)

    # Density-aware bounds on Vbar (for b) and misfit variance (Gamma)
    vmin, vmax = vbar_bounds(elements, rho, rho_max)
    if vmin is None or vmax is None:
        # fallback to unconstrained bounds
        vmin, vmax = float(V.min()), float(V.max())

    Gamma_max = gamma_max_density(elements, rho, rho_max)
    if Gamma_max is None or not np.isfinite(Gamma_max):
        # fallback to unconstrained UB
        Vmax, Vmin = float(V.max()), float(V.min())
        Gamma_max = 0.25 * (Vmax - Vmin) ** 2

    b_min = b_from_V(vmin)
    b_max = b_from_V(vmax)

    if b_min <= 0 or b_max <= 0:
        return 0.0

    if abs(b_max - b_min) < 1e-12:
        b_grid = np.array([b_min])
    else:
        b_grid = np.unique(np.concatenate([
            [b_min, b_max],
            np.geomspace(b_min, b_max, 40)
        ]))

    def tau_y0_UB(b):
        return (0.040 * (alpha**(-1/3)) * mu_max * (f_max**(4/3)) * ((Gamma_max / b**6)**(2/3)))

    def dEb_UB(b):
        return (2.00 * (alpha**(1/3)) * mu_max * (b**3) * (f_max**(2/3)) * ((Gamma_max / b**6)**(1/3)))

    tau_best = 0.0
    for b in b_grid:
        tau0 = tau_y0_UB(b)
        dEb  = dEb_UB(b) * conv_to_eV
        if dEb <= 0 or not np.isfinite(dEb):
            continue
        expo = -(1/0.55) * ((k_B_eV * T_K * ln_eps_ratio / dEb)**0.91)
        tauT = tau0 * math.exp(expo)
        tau_best = max(tau_best, tauT)

    # tau is in GPa; convert to MPa with *1000 and apply Taylor factor
    sigma_y_UB = M_max * 1000.0 * tau_best
    return sigma_y_UB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho-max", type=float, default=9.5, help="Max allowed density (g/cc)")
    parser.add_argument("--out", type=str, default="prune_ys_1300C_density_ub.csv")
    args = parser.parse_args()

    base = ["Nb", "Ta", "V", "Cr", "Mo", "Ti", "W"]
    densities = load_densities_from_prop_source()

    rows = []
    for r in range(2, len(base) + 1):
        for subset in itertools.combinations(base, r):
            rho = np.array([densities[el] for el in subset], dtype=float)
            sigma_ub = prune_impossible_element_set(subset, rho, args.rho_max)
            rows.append({
                "system": "-".join(subset),
                "n_elements": len(subset),
                "sigma_y_UB_MPa": sigma_ub,
                "prunable": sigma_ub < YS_TARGET_MPA,
            })

    df = pd.DataFrame(rows).sort_values(["prunable", "n_elements", "sigma_y_UB_MPa"]).reset_index(drop=True)
    df.to_csv(args.out, index=False)

    pruned = df[df["prunable"]]
    keep = df[~df["prunable"]]

    print(f"Total systems: {len(df)}")
    print(f"Pruned (cannot reach {YS_TARGET_MPA:.0f} MPa at 1300C): {len(pruned)}")
    print(f"Not prunable (might reach): {len(keep)}")
    print(f"rho_max: {args.rho_max:.3f} g/cc")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
