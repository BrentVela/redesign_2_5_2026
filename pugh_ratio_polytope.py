#!/usr/bin/env python
"""
Compute polytope vertices for the constraint Pugh_Ratio_PRIOR > pugh_min
using the algebraic form in 1_find_stoich_props_parallel_for_Class.py.

From that file:
- V_avgr is the composition-weighted average of element V* values
- Pugh_Ratio_PRIOR = (2/3) * (1 + V_avgr) / (1 - 2*V_avgr)

Thus Pugh_Ratio_PRIOR > pugh_min is equivalent to a linear constraint
on V_avgr (since V_avgr is linear in composition).
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


def load_densities_from_prop_source():
    src = PROP_SOURCE.read_text()
    m = re.search(r"prop_data\s*=\s*(\{[\s\S]*?\n\s*\})", src)
    if not m:
        raise RuntimeError("prop_data dict not found in 1_find_stoich_props_parallel_for_Class.py")
    prop_data = ast.literal_eval(m.group(1))
    densities = {el: props["Density [g/cm^3]"] for el, props in prop_data.items()}
    return densities


def v_threshold_from_pugh(pugh_min: float) -> float:
    # (2/3) * (1 + V) / (1 - 2V) > pugh_min
    # => (1 + V) / (1 - 2V) > (3/2) * pugh_min
    # => 1 + V > (3/2) pugh_min * (1 - 2V)
    # => 1 + V > (3/2) pugh_min - 3 pugh_min V
    # => V * (1 + 3 pugh_min) > (3/2) pugh_min - 1
    # => V > [(3/2) pugh_min - 1] / (1 + 3 pugh_min)
    return ((1.5 * pugh_min) - 1.0) / (1.0 + 3.0 * pugh_min)


def vertex_on_edge(v_i: float, v_j: float, v_min: float) -> float:
    """Return t in [0,1] for point t*e_i + (1-t)*e_j where v = v_min."""
    return (v_min - v_j) / (v_i - v_j)


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
    parser.add_argument("--vertices-out", type=str, default="pugh_ratio_polytope_vertices.csv")
    args = parser.parse_args()

    elements = ["Nb", "Ta", "V", "Cr", "Mo", "Ti", "W"]
    vstar_map = load_vstar_from_prop_source()
    vstar = np.array([vstar_map[e] for e in elements], dtype=float)
    densities = load_densities_from_prop_source()
    rho = np.array([densities[e] for e in elements], dtype=float)

    v_min = v_threshold_from_pugh(args.pugh_min)

    n = len(elements)
    vertices = []

    # Simplex vertices e_i
    for i in range(n):
        if vstar[i] >= v_min - 1e-12 and rho[i] <= args.rho_max + 1e-12:
            v = np.zeros(n)
            v[i] = 1.0
            vertices.append(v)

    # Edge intersections with Pugh plane
    for i, j in combinations(range(n), 2):
        vi, vj = vstar[i], vstar[j]
        if (vi - v_min) * (vj - v_min) < 0:  # one feasible, one infeasible
            t = vertex_on_edge(vi, vj, v_min)
            if 0.0 <= t <= 1.0:
                v = np.zeros(n)
                v[i] = t
                v[j] = 1.0 - t
                if (v @ rho) <= args.rho_max + 1e-10:
                    vertices.append(v)
        elif abs(vi - v_min) <= 1e-12 and abs(vj - v_min) <= 1e-12:
            pass

    # Edge intersections with density plane
    for i, j in combinations(range(n), 2):
        ri, rj = rho[i], rho[j]
        if (ri - args.rho_max) * (rj - args.rho_max) < 0:
            t = vertex_on_edge(ri, rj, args.rho_max)
            if 0.0 <= t <= 1.0:
                v = np.zeros(n)
                v[i] = t
                v[j] = 1.0 - t
                if (v @ vstar) >= v_min - 1e-10:
                    vertices.append(v)
        elif abs(ri - args.rho_max) <= 1e-12 and abs(rj - args.rho_max) <= 1e-12:
            pass

    # Triple intersections: sum x = 1, vstar·x = v_min, rho·x = rho_max
    for i, j, k in combinations(range(n), 3):
        A = np.array([
            [1.0, 1.0, 1.0],
            [vstar[i], vstar[j], vstar[k]],
            [rho[i], rho[j], rho[k]],
        ], dtype=float)
        b = np.array([1.0, v_min, args.rho_max], dtype=float)
        if abs(np.linalg.det(A)) < 1e-12:
            continue
        x = np.linalg.solve(A, b)
        if np.all(x >= -1e-10):
            v = np.zeros(n)
            v[i], v[j], v[k] = x
            if np.min(v) >= -1e-10:
                vertices.append(v)

    vertices = dedupe(vertices)
    if len(vertices) < n:
        raise RuntimeError("Not enough vertices to form a full 6D polytope. Check constraints.")

    V = np.vstack(vertices)

    # Sanity checks
    sum_err = np.max(np.abs(V.sum(axis=1) - 1.0))
    if sum_err > 1e-8:
        raise RuntimeError(f"Vertex sum-to-one violation: max error {sum_err}")
    vavg = V @ vstar
    dens = V @ rho
    if np.min(vavg) < v_min - 1e-8:
        raise RuntimeError("Found vertex violating Pugh constraint.")
    if np.max(dens) > args.rho_max + 1e-8:
        raise RuntimeError("Found vertex violating density constraint.")
    if np.min(V) < -1e-8:
        raise RuntimeError("Found negative component in a vertex.")

    # Project to y = (x1..x6), with x7 = 1 - sum(y).
    Z = V[:, : n - 1]
    hull = ConvexHull(Z)
    volume = hull.volume * np.sqrt(n)

    # Write vertices
    df = pd.DataFrame(V, columns=elements)
    df["V_avgr"] = vavg
    df["density"] = dens
    df.to_csv(args.vertices_out, index=False)

    print("Pugh ratio polytope")
    print(f"Elements: {', '.join(elements)}")
    print(f"Pugh min: {args.pugh_min:.3f}")
    print(f"Equivalent V_avgr min: {v_min:.6f}")
    print(f"rho_max: {args.rho_max:.3f} g/cc")
    print(f"Vertex count: {len(vertices)}")
    print(f"Intrinsic 6D volume: {volume:.10f}")
    print(f"Wrote {args.vertices_out}")


if __name__ == "__main__":
    main()
