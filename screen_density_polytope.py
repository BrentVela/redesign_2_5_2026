#!/usr/bin/env python
"""
Exact polytope vertices and hypervolume for linear ROM density screen.

Feasible set:
- x_i >= 0
- sum x_i = 1
- rho·x <= rho_max

This is the 7-simplex intersected with one halfspace. Vertices are:
- any simplex vertex e_i with rho_i <= rho_max
- any edge intersection between a feasible vertex and an infeasible vertex
  where rho·x = rho_max

Volume is computed in the 6D affine subspace (sum x_i = 1) by projecting
vertices onto an orthonormal basis for the subspace using SciPy.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

DEFAULT_DENSITIES = {
    "Nb": 8.57,
    "Ta": 16.69,
    "V": 6.11,
    "Cr": 7.15,
    "Mo": 10.28,
    "Ti": 4.51,
    "W": 19.25,
}


def vertex_on_edge(rho_i: float, rho_j: float, rho_max: float) -> float:
    """Return t in [0,1] for point t*e_i + (1-t)*e_j where rho = rho_max."""
    return (rho_max - rho_j) / (rho_i - rho_j)


def dedupe(points: List[np.ndarray], tol: float = 1e-10) -> List[np.ndarray]:
    unique = []
    for p in points:
        if not any(np.allclose(p, q, atol=tol, rtol=0.0) for q in unique):
            unique.append(p)
    return unique


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho-max", type=float, default=9.5, help="Max allowed density (g/cc)")
    parser.add_argument("--vertices-out", type=str, default="density_polytope_vertices.csv")
    args = parser.parse_args()

    elements = ["Nb", "Ta", "V", "Cr", "Mo", "Ti", "W"]
    rho = np.array([DEFAULT_DENSITIES[e] for e in elements], dtype=float)
    n = len(elements)

    vertices = []

    # Simplex vertices e_i
    for i in range(n):
        if rho[i] <= args.rho_max + 1e-12:
            v = np.zeros(n)
            v[i] = 1.0
            vertices.append(v)

    # Edge intersections between feasible and infeasible vertices
    for i, j in combinations(range(n), 2):
        ri, rj = rho[i], rho[j]
        if (ri - args.rho_max) * (rj - args.rho_max) < 0:  # one feasible, one infeasible
            t = vertex_on_edge(ri, rj, args.rho_max)
            if 0.0 <= t <= 1.0:
                v = np.zeros(n)
                v[i] = t
                v[j] = 1.0 - t
                vertices.append(v)
        elif abs(ri - args.rho_max) <= 1e-12 and abs(rj - args.rho_max) <= 1e-12:
            # both on boundary: both simplex vertices already covered
            pass

    vertices = dedupe(vertices)
    if len(vertices) < n:
        raise RuntimeError("Not enough vertices to form a full 6D polytope. Check constraints.")

    V = np.vstack(vertices)

    # Sanity checks
    sum_err = np.max(np.abs(V.sum(axis=1) - 1.0))
    if sum_err > 1e-8:
        raise RuntimeError(f"Vertex sum-to-one violation: max error {sum_err}")
    dens = V @ rho
    if np.max(dens) > args.rho_max + 1e-8:
        raise RuntimeError("Found vertex violating density constraint.")
    if np.min(V) < -1e-8:
        raise RuntimeError("Found negative component in a vertex.")

    # Project to y = (x1..x6), with x7 = 1 - sum(y).
    # This is a linear bijection between the 6D affine subspace and R^6.
    # The 6D volume scale factor for this map is sqrt(det(A^T A)) where
    # A = [I; -1...-1], so det = n and scale = sqrt(n).
    Z = V[:, : n - 1]
    hull = ConvexHull(Z)
    volume = hull.volume * np.sqrt(n)

    # Write vertices
    df = pd.DataFrame(V, columns=elements)
    df["density"] = dens
    df.to_csv(args.vertices_out, index=False)

    print("Density polytope (linear ROM)")
    print(f"Elements: {', '.join(elements)}")
    print(f"rho_max: {args.rho_max:.3f} g/cc")
    print(f"Vertex count: {len(vertices)}")
    print(f"Intrinsic 6D volume: {volume:.10f}")
    print(f"Wrote {args.vertices_out}")


if __name__ == "__main__":
    main()
