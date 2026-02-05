#!/usr/bin/env python
"""
Compute polytope vertices for combined constraints:
- x_i >= 0
- sum x_i = 1
- V*_avg >= V_min (Pugh_Ratio_PRIOR > pugh_min)
- rho · x <= rho_max
- Tm_avg >= Tm_min (ROM melting temperature)

Loads V* and melting temps from elast_data/prop_data in
1_find_stoich_props_parallel_for_Class.py.
"""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

PROP_SOURCE = Path(__file__).with_name("1_find_stoich_props_parallel_for_Class.py")


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


def dedupe(points, tol: float = 1e-10):
    unique = []
    for p in points:
        if not any(np.allclose(p, q, atol=tol, rtol=0.0) for q in unique):
            unique.append(p)
    return unique


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pugh-min", type=float, default=2.5, help="Minimum Pugh_Ratio_PRIOR")
    parser.add_argument("--rho-max", type=float, default=9.5, help="Max allowed density (g/cc)")
    parser.add_argument("--tm-min-c", type=float, default=2086.0, help="Minimum ROM melting temperature (C)")
    parser.add_argument("--vertices-out", type=str, default="combined_polytope_vertices.csv")
    args = parser.parse_args()

    elements = ["Nb", "Ta", "V", "Cr", "Mo", "Ti", "W"]
    vstar_map = load_vstar_from_prop_source()
    prop_data = load_prop_data_from_prop_source()

    vstar = np.array([vstar_map[e] for e in elements], dtype=float)
    rho = np.array([prop_data[e]["Density [g/cm^3]"] for e in elements], dtype=float)
    tms = np.array([prop_data[e]["Melting Temperature [K]"] for e in elements], dtype=float)

    v_min = v_threshold_from_pugh(args.pugh_min)
    tm_min_k = args.tm_min_c + 273.15

    n = len(elements)
    vertices = []

    # Active hyperplanes we may include
    planes = [
        ("vstar", vstar, v_min),
        ("rho", rho, args.rho_max),
        ("tm", tms, tm_min_k),
    ]

    # Vertex generation: sum x = 1 is always active.
    # Choose k active planes and z = 6 - k zeroed variables to form 7 independent constraints.
    for k in range(0, 4):
        for plane_idxs in combinations(range(3), k):
            active_planes = [planes[i] for i in plane_idxs]
            z = 6 - k
            for zero_idxs in combinations(range(n), z):
                free = [i for i in range(n) if i not in zero_idxs]
                m = len(free)
                if m != 1 + k:
                    continue

                A = np.zeros((1 + k, m), dtype=float)
                b = np.zeros(1 + k, dtype=float)

                # sum x = 1
                A[0, :] = 1.0
                b[0] = 1.0

                # active planes
                for r, (_, coeffs, bound) in enumerate(active_planes, start=1):
                    A[r, :] = coeffs[free]
                    b[r] = bound

                # solve for free variables
                if abs(np.linalg.det(A)) < 1e-12:
                    continue
                x_free = np.linalg.solve(A, b)

                x = np.zeros(n)
                x[free] = x_free

                # feasibility checks
                if np.min(x) < -1e-10:
                    continue
                if abs(np.sum(x) - 1.0) > 1e-8:
                    continue
                if (vstar @ x) < v_min - 1e-8:
                    continue
                if (rho @ x) > args.rho_max + 1e-8:
                    continue
                if (tms @ x) < tm_min_k - 1e-8:
                    continue

                vertices.append(x)

    vertices = dedupe(vertices)
    if len(vertices) < n:
        raise RuntimeError("Not enough vertices to form a full polytope. Check constraints.")

    V = np.vstack(vertices)

    # Sanity checks
    sum_err = np.max(np.abs(V.sum(axis=1) - 1.0))
    if sum_err > 1e-8:
        raise RuntimeError(f"Vertex sum-to-one violation: max error {sum_err}")
    vavg = V @ vstar
    dens = V @ rho
    tmavg = V @ tms
    if np.min(vavg) < v_min - 1e-8:
        raise RuntimeError("Found vertex violating Pugh constraint.")
    if np.max(dens) > args.rho_max + 1e-8:
        raise RuntimeError("Found vertex violating density constraint.")
    if np.min(tmavg) < tm_min_k - 1e-8:
        raise RuntimeError("Found vertex violating melting constraint.")

    # Volume (if full dimensional)
    volume = None
    try:
        Z = V[:, : n - 1]
        hull = ConvexHull(Z)
        volume = hull.volume * np.sqrt(n)
    except Exception:
        volume = None

    # Write vertices
    df = pd.DataFrame(V, columns=elements)
    df["V_avgr"] = vavg
    df["density"] = dens
    df["Tm_avg_K"] = tmavg
    df.to_csv(args.vertices_out, index=False)

    print("Combined polytope")
    print(f"Elements: {', '.join(elements)}")
    print(f"Pugh min: {args.pugh_min:.3f} (V_avgr >= {v_min:.6f})")
    print(f"rho_max: {args.rho_max:.3f} g/cc")
    print(f"Tm_min: {tm_min_k:.2f} K ({args.tm_min_c:.1f} C)")
    print(f"Vertex count: {len(vertices)}")
    if volume is not None:
        print(f"Intrinsic 6D volume: {volume:.10f}")
    else:
        print("Intrinsic 6D volume: n/a (lower-dimensional polytope)")
    print(f"Wrote {args.vertices_out}")


if __name__ == "__main__":
    main()
