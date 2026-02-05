#!/usr/bin/env python
import argparse
import os
import re
import numpy as np
import pandas as pd
from tc_python import *
from itertools import compress
from tc_python import server
import time
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
import os.path as path
import time
import os
import traceback
from pathlib import Path


def _ensure_tc_home() -> None:
    if os.environ.get("TC25B_HOME"):
        return
    default_path = "/opt/Thermo-Calc/2025b"
    if os.path.isdir(default_path):
        os.environ["TC25B_HOME"] = default_path
    else:
        raise EnvironmentError("Environment variable 'TC25B_HOME' not found and default path is missing.")


_ensure_tc_home()


# RT THCD helper for Cafer featurization
def compute_rt_thcd(df, element_cols, log_fn=None, db="TCHEA7"):
    out = df.copy()
    out["THCD_25C_W_mK"] = np.nan

    groups = {}
    for idx, row in out.iterrows():
        comp_vals = pd.to_numeric(row[element_cols], errors="coerce").to_numpy()
        comp_vals = np.nan_to_num(comp_vals, nan=0.0)
        comp = pd.Series(comp_vals, index=element_cols)
        total = float(comp.sum())
        if total <= 0:
            continue
        comp = comp / total
        active = tuple([el for el in element_cols if comp[el] > 0])
        groups.setdefault(active, []).append((idx, comp))

    with TCPython(logging_policy=LoggingPolicy.NONE) as session:
        session.disable_caching()
        for active_el, rows in groups.items():
            if len(active_el) == 0:
                continue
            try:
                system = (
                    session
                    .select_database_and_elements(db, list(active_el))
                    .get_system()
                )
                eq = system.with_single_equilibrium_calculation().set_condition(
                    ThermodynamicQuantity.temperature(), 25 + 273.15
                )
            except Exception as exc:
                if log_fn:
                    log_fn(f"TCHEA7 select failed for elements {active_el}: {exc}")
                continue

            for idx, comp in rows:
                try:
                    if len(active_el) == 1:
                        eq.set_dependent_element(active_el[0])
                    else:
                        for el in active_el[:-1]:
                            eq.set_condition(
                                ThermodynamicQuantity.mole_fraction_of_a_component(el),
                                float(comp[el]),
                            )
                    res = eq.calculate()
                    out.at[idx, "THCD_25C_W_mK"] = res.get_value_of("THCD")
                except Exception as exc:
                    if log_fn:
                        log_fn(f"RT THCD failed for {active_el} at row {idx}: {exc}")
                    continue

    return out


#Pre-Processing Data
def EQUIL(param, temps_c):
    indices = param["INDICES"]
    comp_df = param["COMP"]
    elements = param["ACT_EL"]
    active_el = elements

    with TCPython(logging_policy=LoggingPolicy.NONE) as session:
        session.disable_caching()

        system = (
            session.
                select_database_and_elements('TCHEA7', active_el).
                get_system())

        options = SingleEquilibriumOptions().set_required_accuracy(1.0e-2).set_max_no_of_iterations(300).set_smallest_fraction(1.0e-6)
        eq_calculation = system.with_single_equilibrium_calculation(). \
            set_condition(ThermodynamicQuantity.temperature(), 298).with_options(options)
        eq_calculation.disable_global_minimization()

        # Focus on BCC and Laves phases only for faster convergence
        eq_calculation.set_phase_to_suspended(ALL_PHASES)
        for phase in ["BCC_B2", "BCC_B2#2", "C14_LAVES", "C15_LAVES", "C36_LAVES"]:
            eq_calculation.set_phase_to_entered(phase)

        for i in indices:
            # Get the composition list and corresponding active_el list
            solidus = comp_df.loc[i]['PROP ST (K)'] if 'PROP ST (K)' in comp_df.columns else None
            liquidus = comp_df.loc[i]['PROP LT (K)'] if 'PROP LT (K)' in comp_df.columns else None
            comp = np.array(comp_df.loc[i][active_el])
            #Create equilibrium calculation object and set conditions
            try: # with TCAL7, if that fails then we will try another database

                # Check Point 1
                if len(active_el) == 1:
                    # Do not do anything for unaries
                    eq_calculation.set_dependent_element(active_el[0])
                else:
                    for j in range(len(active_el)-1):
                        eq_calculation.set_condition(ThermodynamicQuantity.mole_fraction_of_a_component(active_el[j]),
                                                     comp[j])

                for temp in temps_c:
                    eq_calculation =  eq_calculation.set_condition(ThermodynamicQuantity.temperature(), temp+273)
                    eq_result = eq_calculation.calculate()

                    #Get all possible phases
                    pnames = eq_result.get_phases()

                    for phase in pnames:
                        #Get mol fraction list for each phase
                        if eq_result.get_value_of('NPM(' + phase + ')') > 0: #Only report phases that are present
                            comp_df.at[i,'EQ {}C {} MOL'.format(temp,phase)] = eq_result.get_value_of('NPM(' + phase + ')')


                if solidus is not None:
                    # Properties at Solidus
                    eq_calculation = eq_calculation.set_condition(ThermodynamicQuantity.temperature(), solidus-1)
                    eq_result = eq_calculation.calculate()
                    comp_df.at[i, 'EQ ST H (J/mol)']     = eq_result.get_value_of('HM')
                    comp_df.at[i, 'EQ ST H (J)']         = eq_result.get_value_of('H')
                    comp_df.at[i, 'EQ ST THCD (W/mK)']   = eq_result.get_value_of('THCD')
                    comp_df.at[i, 'EQ ST Density (g/cc)'] =eq_result.get_value_of('BM') / eq_result.get_value_of('VM') / 10 ** 6
                    comp_df.at[i, 'EQ ST MASS (g/mol)'] = eq_result.get_value_of('BM')
                    comp_df.at[i, 'EQ ST VOL (m3/mol)'] = eq_result.get_value_of('VM')

                if liquidus is not None:
                    # Properties at Liquidus
                    eq_calculation = eq_calculation.set_condition(ThermodynamicQuantity.temperature(), liquidus+1)
                    eq_result = eq_calculation.calculate()
                    comp_df.at[i, 'EQ LT H (J/mol)']    = eq_result.get_value_of('HM')  # J/mol
                    comp_df.at[i, 'EQ LT H (J)']        = eq_result.get_value_of('H')
                    comp_df.at[i, 'EQ LT THCD (W/mK)']  = eq_result.get_value_of('THCD')  # W/mK
                    comp_df.at[i, 'EQ LT DVIS (Pa-s)']  = eq_result.get_value_of('DVIS (liquid)')  # Pa-s
                    comp_df.at[i, 'EQ LT KVIS (m2/s)']  = eq_result.get_value_of('KVIS (liquid)')
                    comp_df.at[i, 'EQ LT Density (g/cc)'] = eq_result.get_value_of('BM') / eq_result.get_value_of('VM') / 10 ** 6
                    comp_df.at[i, 'EQ LT VOL (m3/mol)'] = eq_result.get_value_of('VM')
                    comp_df.at[i, 'EQ LT Surface Tension (N/m)'] = eq_result.get_value_of('SURF(LIQUID)')

            except Exception as e2:
                print('Exception occurred on line {}:'.format(traceback.extract_tb(e2.__traceback__)[0][1]))
                print(e2)
                
            finally:
                comp_df.to_csv('CalcFiles/EQUIL_AP_OUT_{}.csv'.format(param["INDICES"][0]))
                continue

    return 'Complete'


def _equil_with_temps(args):
    param, temps_c = args
    return EQUIL(param, temps_c)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV or XLSX with element columns")
    parser.add_argument("--element-cols", default="", help="Comma-separated element columns (auto-detect if empty)")
    parser.add_argument("--family-chunk-size", type=int, default=50, help="Max rows per output file within a family")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional max rows for quick test")
    parser.add_argument("--max-workers", type=int, default=2, help="Parallel workers (one TC session per family)")
    parser.add_argument("--out-dir", default="featurize_database_for_cafer/rt_thcd_chunks", help="Output directory")
    args = parser.parse_args()

    in_path = Path(args.input)
    if in_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(in_path)
    else:
        df = pd.read_csv(in_path)

    if args.element_cols:
        element_cols = [c.strip() for c in args.element_cols.split(",") if c.strip()]
    else:
        element_cols = [c for c in df.columns if re.fullmatch(r"[A-Z][a-z]?", str(c))]

    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group by family across dataset
    families = {}
    for idx, row in df.iterrows():
        comp_vals = pd.to_numeric(row[element_cols], errors="coerce").to_numpy()
        comp_vals = np.nan_to_num(comp_vals, nan=0.0)
        comp = pd.Series(comp_vals, index=element_cols)
        total = float(comp.sum())
        if total <= 0:
            continue
        comp = comp / total
        active = tuple([el for el in element_cols if comp[el] > 0])
        comp_active = [float(comp[el]) for el in active]
        families.setdefault(active, []).append((idx, comp_active))

    total_families = len(families)
    total_rows = len(df)
    print(f"Found {total_families} element families across {total_rows} rows")

    def _compute_family_rt_thcd(args):
        active_el, rows = args
        _ensure_tc_home()
        results = []
        if len(active_el) == 0:
            return results
        with TCPython(logging_policy=LoggingPolicy.NONE) as session:
            session.disable_caching()
            system = (
                session
                .select_database_and_elements("TCHEA7", list(active_el))
                .get_system()
            )
            eq = system.with_single_equilibrium_calculation().set_condition(
                ThermodynamicQuantity.temperature(), 25 + 273.15
            )
            for idx, comp in rows:
                try:
                    if len(active_el) == 1:
                        eq.set_dependent_element(active_el[0])
                    else:
                        for el, val in zip(active_el[:-1], comp[:-1]):
                            eq.set_condition(
                                ThermodynamicQuantity.mole_fraction_of_a_component(el),
                                float(val),
                            )
                    res = eq.calculate()
                    results.append((idx, res.get_value_of("THCD")))
                except Exception:
                    results.append((idx, np.nan))
        return results

    # Parallelize by family, write as each finishes
    with ProcessPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futures = {ex.submit(_compute_family_rt_thcd, (active_el, rows)): active_el for active_el, rows in families.items()}
        done_families = 0
        done_rows = 0
        for fut in as_completed(futures):
            active_el = futures[fut]
            rows = families[active_el]
            done_families += 1
            done_rows += len(rows)
            pct = 100.0 * done_rows / max(1, total_rows)
            fam_tag = "-".join(active_el) if active_el else "EMPTY"
            print(f"[{done_families}/{total_families}] Family {fam_tag} complete ({len(rows)} rows). Overall {pct:.1f}%")

            results = {idx: np.nan for idx, _ in rows}
            try:
                for idx, thcd in fut.result():
                    results[idx] = thcd
            except Exception:
                pass

            family_idx = [idx for idx, _ in rows]
            family_df = df.loc[family_idx].copy()
            family_df["THCD_25C_W_mK"] = [results[i] for i in family_idx]

            max_chunk = max(1, args.family_chunk_size)
            fam_tag = "-".join(active_el)
            for start in range(0, len(family_df), max_chunk):
                end = min(start + max_chunk, len(family_df))
                chunk = family_df.iloc[start:end].copy()
                out_path = out_dir / f"rt_thcd_{fam_tag}_{start:05d}_{end-1:05d}.csv"
                chunk.to_csv(out_path, index=False)
                print(f"Wrote {out_path}")

    # Done
