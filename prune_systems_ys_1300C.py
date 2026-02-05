#!/usr/bin/env python
"""
Prune Nb–Ta–V–Cr–Mo–Ti–W sub-systems using a permissive upper bound
on YS at 1300C (1573 K) derived from the Curtin strength model.

Elemental elastic constants and BCC volumes are taken from
redesign_2_4_26/strength_model.py for consistency.

Outputs CSV with sub-system, sigma_y_UB, and prunable flag.
"""

from __future__ import annotations

import itertools
import math
import numpy as np
import pandas as pd

# ---- elemental data from redesign_2_4_26/strength_model.py ----
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


def prune_impossible_element_set(elements):
    C11 = np.array([elast_data[el]['C11'] for el in elements])
    C12 = np.array([elast_data[el]['C12'] for el in elements])
    C44 = np.array([elast_data[el]['C44'] for el in elements])
    V   = np.array([volumes[el] for el in elements])

    C11_max, C12_min, C44_max = C11.max(), C12.min(), C44.max()
    Vmax, Vmin = V.max(), V.min()

    mu_max = math.sqrt(0.5 * C44_max * (C11_max - C12_min))
    nu_max = nu_cap
    f_max = (1 + nu_max) / (1 - nu_max)
    Gamma_max = 0.25 * (Vmax - Vmin) ** 2

    def b_from_V(Vbar):
        a = (2 * Vbar) ** (1/3)
        return (math.sqrt(3)/2) * a

    b_min = b_from_V(Vmin)
    b_max = b_from_V(Vmax)

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
    base = ["Nb", "Ta", "V", "Cr", "Mo", "Ti", "W"]

    rows = []
    for r in range(2, len(base) + 1):
        for subset in itertools.combinations(base, r):
            sigma_ub = prune_impossible_element_set(subset)
            rows.append({
                "system": "-".join(subset),
                "n_elements": len(subset),
                "sigma_y_UB_MPa": sigma_ub,
                "prunable": sigma_ub < YS_TARGET_MPA,
            })

    df = pd.DataFrame(rows).sort_values(["prunable", "n_elements", "sigma_y_UB_MPa"]).reset_index(drop=True)
    out = "/home/vela/projects/ult/redesign_2_4_26/prune_ys_1300C.csv"
    df.to_csv(out, index=False)

    pruned = df[df["prunable"]]
    keep = df[~df["prunable"]]

    print(f"Total systems: {len(df)}")
    print(f"Pruned (cannot reach {YS_TARGET_MPA:.0f} MPa at 1300C): {len(pruned)}")
    print(f"Not prunable (might reach): {len(keep)}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
