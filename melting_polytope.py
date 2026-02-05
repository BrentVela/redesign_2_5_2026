#!/usr/bin/env python
"""
Compute polytope vertices for the constraint ROM melting temperature >= Tm_min.

Constraint:
- x_i >= 0
- sum x_i = 1
- Tm · x >= Tm_min (linear rule-of-mixtures in Kelvin)

Loads elemental melting temperatures from prop_data in
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


def load_melting_temps_from_prop_source():
    src = PROP_SOURCE.read_text()
    m = re.search(r"prop_data\s*=\s*(\{[\s\S]*?\n\s*\})", src)
    if not m:
        raise RuntimeError("prop_data dict not found in 1_find_stoich_props_parallel_for_Class.py")
    prop_data = ast.literal_eval(m.group(1))
    tms = {el: props["Melting Temperature [K]"] for el, props in prop_data.items()}
    return tms


def vertex_on_edge(t_i: float, t_j: float, t_min: float) -> float:
    """Return t in [0,1] for point t*e_i + (1-t)*e_j where Tm = t_min."""
    return (t_min - t_j) / (t_i - t_j)


def dedupe(points, tol: float = 1e-10):
    unique = []
    for p in points:
        if not any(np.allclose(p, q, atol=tol, rtol=0.0) for q in unique):
            unique.append(p)
    return unique


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tm-min-c", type=float, default=2086.0, help="Minimum ROM melting temperature (C)")
    parser.add_argument("--vertices-out", type=str, default="melting_polytope_vertices.csv")
    args = parser.parse_args()

    elements = ["Nb", "Ta", "V", "Cr", "Mo", "Ti", "W"]
    tms_map = load_melting_temps_from_prop_source()
    tms = np.array([tms_map[e] for e in elements], dtype=float)

    tm_min_k = args.tm_min_c + 273.15

    n = len(elements)
    vertices = []

    # Simplex vertices e_i
    for i in range(n):
        if tms[i] >= tm_min_k - 1e-12:
            v = np.zeros(n)
            v[i] = 1.0
            vertices.append(v)

    # Edge intersections between feasible and infeasible vertices
    for i, j in combinations(range(n), 2):
        ti, tj = tms[i], tms[j]
        if (ti - tm_min_k) * (tj - tm_min_k) < 0:
            t = vertex_on_edge(ti, tj, tm_min_k)
            if 0.0 <= t <= 1.0:
                v = np.zeros(n)
                v[i] = t
                v[j] = 1.0 - t
                vertices.append(v)
        elif abs(ti - tm_min_k) <= 1e-12 and abs(tj - tm_min_k) <= 1e-12:
            pass

    vertices = dedupe(vertices)
    if len(vertices) < n:
        raise RuntimeError("Not enough vertices to form a full 6D polytope. Check constraints.")

    V = np.vstack(vertices)

    # Sanity checks
    sum_err = np.max(np.abs(V.sum(axis=1) - 1.0))
    if sum_err > 1e-8:
        raise RuntimeError(f"Vertex sum-to-one violation: max error {sum_err}")
    tm_avg = V @ tms
    if np.min(tm_avg) < tm_min_k - 1e-8:
        raise RuntimeError("Found vertex violating melting constraint.")
    if np.min(V) < -1e-8:
        raise RuntimeError("Found negative component in a vertex.")

    # Project to y = (x1..x6), with x7 = 1 - sum(y).
    Z = V[:, : n - 1]
    hull = ConvexHull(Z)
    volume = hull.volume * np.sqrt(n)

    # Write vertices
    df = pd.DataFrame(V, columns=elements)
    df["Tm_avg_K"] = tm_avg
    df.to_csv(args.vertices_out, index=False)

    print("Melting temperature polytope (ROM)")
    print(f"Elements: {', '.join(elements)}")
    print(f"Tm_min: {tm_min_k:.2f} K ({args.tm_min_c:.1f} C)")
    print(f"Vertex count: {len(vertices)}")
    print(f"Intrinsic 6D volume: {volume:.10f}")
    print(f"Wrote {args.vertices_out}")


if __name__ == "__main__":
    main()
