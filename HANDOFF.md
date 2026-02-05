# Handoff Summary

## Context
Project dir: `/home/vela/projects/ult/redesign_2_5_2026`

Alloy space: `Nb–Ta–V–Cr–Mo–Ti–W`

Constraints in use:
- Density `<= 9.5 g/cc`
- Pugh_Ratio_PRIOR > 2.5 ⇒ linear constraint on `V_avgr` (`>= 0.323529`)
- ROM melting temperature `>= 2300°C` (Kelvin in scripts)
- Curtin‑Maresca upper‑bound pruning excludes infeasible element systems

## Key scripts created

- `density_lp_bounds.py`
  - Per‑element min/max bounds under density constraint
  - Loads densities from `1_find_stoich_props_parallel_for_Class.py`

- `prune_systems_ys_1300C_density.py`
  - Density‑aware Curtin‑Maresca UB pruning
  - Outputs `prune_ys_1300C_density_ub.csv`

- `pugh_ratio_polytope.py`
  - Polytope vertices for Pugh constraint, combined with density
  - Output: `pugh_ratio_polytope_vertices.csv`

- `melting_polytope.py`
  - Polytope vertices for ROM Tm constraint
  - Output: `melting_polytope_vertices.csv`

- `combined_polytope.py`
  - Combined polytope: density + Pugh + ROM Tm
  - Output: `combined_polytope_vertices.csv`

- `sample_feasible_polytope.py`
  - Samples feasible polytope on grid (5 at% default)
  - Excludes Curtin‑pruned systems
  - Example output with Tm≥2300C: `feasible_samples_5atpct_Tm2300C.csv`

## Data artifacts

- Feasible samples with YS:
  - `feasible_samples_5atpct_Tm2300C_with_YS.csv`
- YS vs density plot:
  - `ys_vs_density_Tm2300C.png`
- YS vs density with proximity highlighting:
  - `ys_vs_density_Tm2300C_highlighted_0p20.png`
- Density vs Tm plot:
  - `density_vs_tm_Tm2300C.png`
- Density vs Tm with proximity highlighting:
  - `density_vs_tm_Tm2300C_highlighted_0p20.png`
- TC solidus vs density plot:
  - `tc_solidus_vs_density_ysgt400.png`
- ROM Tm vs TC solidus plot:
  - `rom_tm_vs_tc_solidus_ysgt400.png`

## Target composition proximity

Target: `(NbTaV)65_Mo10_W10_Ti10_Cr5`
- Nb/Ta/V = 0.2166667 each
- Mo/W/Ti = 0.10
- Cr = 0.05

Distance: Euclidean (L2) in composition space
Radius used: 0.20

Nearby points:
- `near_target_within_0p20.csv`
- `near_target_within_0p20_YSgt400.csv` (265 alloys)

Average YS for 612 nearby alloys: **379.68 MPa**

## TC solidus/liquidus (YS>400 subset)

Run command:

```bash
python /home/vela/projects/ult/redesign_2_5_2026/TC_Property_Module.py \
  --input /home/vela/projects/ult/redesign_2_5_2026/near_target_within_0p20_YSgt400.csv \
  --out-dir /home/vela/projects/ult/redesign_2_5_2026/tc_solidus_liquidus_ysgt400 \
  --family-chunk-size 10 \
  --max-workers 20
```

Output directory:
- `tc_solidus_liquidus_ysgt400/`

Note: TC output currently has **266 rows** (duplicate/extra row likely). Dedup by composition if needed.

## Notes

- Do not edit `1_find_stoich_props_parallel_for_Class.py` for constraints; use new scripts.
- Pugh constraint derivation:
  - `Pugh_Ratio_PRIOR = (2/3) * (1 + V_avgr) / (1 - 2*V_avgr)`
  - `Pugh > 2.5` ⇒ `V_avgr > 0.323529`
