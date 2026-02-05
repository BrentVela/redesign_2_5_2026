#!/usr/bin/env python
"""
Compute per-element min/max composition bounds under a density constraint
using linear programming.

Constraints:
- x_i >= 0
- sum x_i = 1
- rho · x <= rho_max

Outputs: CSV with element, x_min, x_max.
"""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog

PROP_SOURCE = Path(__file__).with_name("1_find_stoich_props_parallel_for_Class.py")


def load_densities_from_prop_source():
    """Load elemental densities from prop_data in 1_find_stoich_props_parallel_for_Class.py."""
    src = PROP_SOURCE.read_text()
    m = re.search(r"prop_data\s*=\s*(\{[\s\S]*?\n\s*\})", src)
    if not m:
        raise RuntimeError("prop_data dict not found in 1_find_stoich_props_parallel_for_Class.py")
    prop_data = ast.literal_eval(m.group(1))
    densities = {el: props["Density [g/cm^3]"] for el, props in prop_data.items()}
    return densities


def solve_bounds(elements, rho, rho_max):
    n = len(elements)
    bounds = [(0.0, 1.0) for _ in range(n)]

    # Equality: sum x = 1
    A_eq = [np.ones(n)]
    b_eq = [1.0]

    # Inequality: rho · x <= rho_max
    A_ub = [rho]
    b_ub = [rho_max]

    rows = []
    for i, el in enumerate(elements):
        c = np.zeros(n)
        c[i] = 1.0

        # min
        res_min = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if not res_min.success:
            raise RuntimeError(f"Min LP failed for {el}: {res_min.message}")

        # max = -min(-x_i)
        res_max = linprog(-c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if not res_max.success:
            raise RuntimeError(f"Max LP failed for {el}: {res_max.message}")

        rows.append({
            "element": el,
            "x_min": res_min.x[i],
            "x_max": res_max.x[i],
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho-max", type=float, default=9.5, help="Max allowed density (g/cc)")
    parser.add_argument("--out", type=str, default="density_lp_bounds.csv")
    args = parser.parse_args()

    elements = ["Nb", "Ta", "V", "Cr", "Mo", "Ti", "W"]
    densities = load_densities_from_prop_source()
    rho = np.array([densities[e] for e in elements], dtype=float)

    df = solve_bounds(elements, rho, args.rho_max)
    df.to_csv(args.out, index=False)

    print("LP density bounds")
    print(f"Elements: {', '.join(elements)}")
    print(f"rho_max: {args.rho_max:.3f} g/cc")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
