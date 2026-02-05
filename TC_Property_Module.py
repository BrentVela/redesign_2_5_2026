#!/usr/bin/env python
import numpy as np
import pandas as pd
from tc_python import *
from itertools import compress
from tc_python import server
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context, Queue
import time
import concurrent.futures
import os.path as path
import time
import os
import re
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


def compute_solidus_liquidus(df, element_cols, log_fn=None, db="TCHEA7"):
    _ensure_tc_home()
    out = df.copy()
    out["Solidus_K_TC"] = np.nan
    out["Liquidus_K_TC"] = np.nan
    out["Solidus_C_TC"] = np.nan
    out["Liquidus_C_TC"] = np.nan

    # group by active element set for reuse
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
                calc = (
                    session
                    .select_database_and_elements(db, list(active_el))
                    .get_system()
                    .with_property_model_calculation("Liquidus and Solidus Temperature")
                    .set_argument("upperTemperatureLimit", 4000)
                    .set_composition_unit(CompositionUnit.MOLE_PERCENT)
                    .set_temperature(300)
                )
            except Exception as exc:
                if log_fn:
                    log_fn(f"TCHEA7 select failed for elements {active_el}: {exc}")
                continue

            for idx, comp in rows:
                try:
                    if len(active_el) == 1:
                        calc.set_dependent_element(active_el[0])
                    else:
                        for el in active_el[:-1]:
                            calc.set_composition(el, float(comp[el]) * 100.0)
                    res = calc.calculate()
                    sol_k = res.get_value_of("Solidus temperature")
                    liq_k = res.get_value_of("Liquidus temperature")
                    out.at[idx, "Solidus_K_TC"] = sol_k
                    out.at[idx, "Liquidus_K_TC"] = liq_k
                    out.at[idx, "Solidus_C_TC"] = sol_k - 273.15
                    out.at[idx, "Liquidus_C_TC"] = liq_k - 273.15
                except Exception as exc:
                    if log_fn:
                        log_fn(f"Solidus/liquidus failed for {active_el} at row {idx}: {exc}")
                    continue

    return out


def _calc_single_comp(active_el, comp, out_q):
    try:
        _ensure_tc_home()
        with TCPython(logging_policy=LoggingPolicy.NONE) as session:
            session.disable_caching()
            calc = (
                session
                .select_database_and_elements("TCHEA7", list(active_el))
                .get_system()
                .with_property_model_calculation("Liquidus and Solidus Temperature")
                .set_argument("upperTemperatureLimit", 4000)
                .set_composition_unit(CompositionUnit.MOLE_PERCENT)
                .set_temperature(300)
            )
            if len(active_el) == 1:
                calc.set_dependent_element(active_el[0])
            else:
                for el, val in zip(active_el[:-1], comp[:-1]):
                    calc.set_composition(el, float(val) * 100.0)
            res = calc.calculate()
            sol_k = res.get_value_of("Solidus temperature")
            liq_k = res.get_value_of("Liquidus temperature")
        out_q.put((sol_k, liq_k, False))
    except Exception:
        out_q.put((np.nan, np.nan, False))


def _compute_family_tc(args):
    active_el, rows, per_comp_timeout, per_family_timeout = args
    _ensure_tc_home()
    results = []
    if len(active_el) == 0:
        return results

    start_time = time.time()
    row_pairs = list(rows)

    if per_comp_timeout and per_comp_timeout > 0:
        ctx = get_context("spawn")
        for idx, comp in row_pairs:
            if per_family_timeout and (time.time() - start_time) > per_family_timeout:
                results.append((idx, np.nan, np.nan, True))
                # mark remaining as timed out
                for rem_idx, _ in row_pairs[row_pairs.index((idx, comp)) + 1:]:
                    results.append((rem_idx, np.nan, np.nan, True))
                return results
            q = ctx.Queue()
            p = ctx.Process(target=_calc_single_comp, args=(active_el, comp, q))
            p.start()
            p.join(per_comp_timeout)
            if p.is_alive():
                p.terminate()
                p.join()
                results.append((idx, np.nan, np.nan, True))
            else:
                try:
                    sol_k, liq_k, _ = q.get_nowait()
                except Exception:
                    sol_k, liq_k = np.nan, np.nan
                results.append((idx, sol_k, liq_k, False))
    else:
        with TCPython(logging_policy=LoggingPolicy.NONE) as session:
            session.disable_caching()
            calc = (
                session
                .select_database_and_elements("TCHEA7", list(active_el))
                .get_system()
                .with_property_model_calculation("Liquidus and Solidus Temperature")
                .set_argument("upperTemperatureLimit", 4000)
                .set_composition_unit(CompositionUnit.MOLE_PERCENT)
                .set_temperature(300)
            )

            for idx, comp in row_pairs:
                if per_family_timeout and (time.time() - start_time) > per_family_timeout:
                    results.append((idx, np.nan, np.nan, True))
                    # mark remaining as timed out
                    for rem_idx, _ in row_pairs[row_pairs.index((idx, comp)) + 1:]:
                        results.append((rem_idx, np.nan, np.nan, True))
                    return results
                try:
                    if len(active_el) == 1:
                        calc.set_dependent_element(active_el[0])
                    else:
                        for el, val in zip(active_el[:-1], comp[:-1]):
                            calc.set_composition(el, float(val) * 100.0)
                    res = calc.calculate()
                    sol_k = res.get_value_of("Solidus temperature")
                    liq_k = res.get_value_of("Liquidus temperature")
                    results.append((idx, sol_k, liq_k, False))
                except Exception:
                    results.append((idx, np.nan, np.nan, False))
    return results


#Pre-Processing Data
def Property(param):
    indices = param["INDICES"]
    comp_df = param["COMP"]
    elements = param["ACT_EL"]

    active_el = elements

    prev_active_el = []

    try:
        with TCPython() as session:
            session.disable_caching()

            eq_calculation = (
                session.
                    select_database_and_elements('TCHEA7', active_el).
                    get_system().
                    with_property_model_calculation("Liquidus and Solidus Temperature").
                    set_argument('upperTemperatureLimit', 4000).
                    set_composition_unit(CompositionUnit.MOLE_PERCENT).set_temperature(300))

            prop_calc = (
                session.
                    select_database_and_elements('TCHEA7', active_el).
                    get_system().
                    with_property_model_calculation('Equilibrium with Freeze-in Temperature').
                    set_temperature(25 + 273.15).
                    set_composition_unit(CompositionUnit.MOLE_PERCENT))

            indices = sorted(indices)
            print('Starting alloys with indices {}-{}'.format(indices[0],indices[-1]))
            for i in indices:
                comp = np.array(comp_df.loc[i][active_el])
                
                if len(active_el) != 1:
                    for j in range(len(active_el) - 1):
                        eq_calculation.set_composition(active_el[j], comp[j])
                        prop_calc.set_composition(active_el[j], comp[j])
                else:
                    eq_calculation.set_dependent_element(active_el[0])
                    prop_calc.set_dependent_element(active_el[0])


                result = eq_calculation.calculate()
                comp_df.at[i,'PROP LT (K)']  = result.get_value_of('Liquidus temperature')
                comp_df.at[i,'PROP ST (K)']  = result.get_value_of('Solidus temperature')

                # #Solidus Props
                # prop_calc = prop_calc.set_argument('Freeze-in-temperature', comp_df.at[i,'PROP ST (K)'])
                # prop_calc = prop_calc.set_argument('Minimization strategy', 'Global minimization only')
                # prop_calc = prop_calc.set_temperature(comp_df.at[i,'PROP ST (K)'])
                # prop_result = prop_calc.calculate()
                # comp_df.at[i,'PROP ST Density (g/cm3)'] = prop_result.get_value_of('Density (g/cm3)')
                # comp_df.at[i,'PROP ST C (J/(mol K))']     = prop_result.get_value_of('Heat capacity (J/(mol K))')
                # comp_df.at[i,'PROP ST THCD (W/(mK))'] = prop_result.get_value_of('Thermal conductivity (W/(mK))')
                # comp_df.at[i,'PROP ST THRS (mK/W)'] = prop_result.get_value_of('Thermal resistivity (mK/W)')
                # comp_df.at[i,'PROP ST TDIV (m2/s)']    = prop_result.get_value_of('Thermal diffusivity (m2/s)')
                # comp_df.at[i,'PROP ST ELRS (Ohm m)'] = prop_result.get_value_of('Electric resistivity (ohm m)')
                # comp_df.at[i,'PROP ST ELCD (S/m)'] = prop_result.get_value_of('Electric conductivity (S/m)')
                #
                # #Liquidus Props
                # prop_calc = prop_calc.set_argument('Freeze-in-temperature',comp_df.at[i,'PROP LT (K)'])
                # prop_calc = prop_calc.set_argument('Minimization strategy', 'Global minimization only')
                # prop_calc = prop_calc.set_temperature(comp_df.at[i,'PROP LT (K)'])
                # prop_result = prop_calc.calculate()
                # comp_df.at[i,'PROP LT Density (g/cm3)'] = prop_result.get_value_of('Density (g/cm3)')
                # comp_df.at[i,'PROP LT C (J/(mol K))']     = prop_result.get_value_of('Heat capacity (J/(mol K))')
                # comp_df.at[i,'PROP LT THCD (W/(mK))'] = prop_result.get_value_of('Thermal conductivity (W/(mK))')
                # comp_df.at[i,'PROP LT THRS (mK/W)'] = prop_result.get_value_of('Thermal resistivity (mK/W)')
                # comp_df.at[i,'PROP LT TDIV (m2/s)']    = prop_result.get_value_of('Thermal diffusivity (m2/s)')
                # comp_df.at[i,'PROP LT ELRS (Ohm m)'] = prop_result.get_value_of('Electric resistivity (ohm m)')
                # comp_df.at[i,'PROP LT ELCD (S/m)'] = prop_result.get_value_of('Electric conductivity (S/m)')
                #
                # #T Props, Specify T in celcius
                # temperatures = [25, 600, 1300, 2000]
                # for temp in temperatures:
                #     prop_calc = prop_calc.set_argument('Freeze-in-temperature',2000+273)
                #     prop_calc = prop_calc.set_argument('Minimization strategy', 'Global minimization only')
                #     prop_calc = prop_calc.set_argument('Reference temperature for technical CTE',25+273)
                #     prop_calc = prop_calc.set_temperature(temp+273)
                #     prop_result = prop_calc.calculate()
                #     comp_df.at[i,'PROP {}C Density (g/cm3)'.format(temp)] = prop_result.get_value_of('Density (g/cm3)')
                #     comp_df.at[i,'PROP {}C C (J/(mol K))'.format(temp)]     = prop_result.get_value_of('Heat capacity (J/(mol K))')
                #     comp_df.at[i,'PROP {}C THCD (W/(mK))'.format(temp)] = prop_result.get_value_of('Thermal conductivity (W/(mK))')
                #     comp_df.at[i,'PROP {}C THRS (mK/W)'.format(temp)] = prop_result.get_value_of('Thermal resistivity (mK/W)')
                #     comp_df.at[i,'PROP {}C TDIV (m2/s)'.format(temp)]    = prop_result.get_value_of('Thermal diffusivity (m2/s)')
                #     comp_df.at[i,'PROP {}C ELRS (Ohm m)'.format(temp)] = prop_result.get_value_of('Electric resistivity (ohm m)')
                #     comp_df.at[i,'PROP {}C ELCD (S/m)'.format(temp)] = prop_result.get_value_of('Electric conductivity (S/m)')

                
                    

    except Exception as e2:
        print('Exception occurred on line {}:'.format(traceback.extract_tb(e2.__traceback__)[0][1]))
        print(e2)

    finally:
        comp_df.to_csv('CalcFiles/PROP_OUT_{}.csv'.format(param["INDICES"][0]),index=False)
        print('Saving alloys with indices {}-{}'.format(indices[0],indices[-1]))

    return 'Complete'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV or XLSX with element columns")
    parser.add_argument("--element-cols", default="", help="Comma-separated element columns (auto-detect if empty)")
    parser.add_argument("--family-chunk-size", type=int, default=10, help="Max rows per output file within a family")
    parser.add_argument("--max-workers", type=int, default=20, help="Parallel workers (one TC session per family)")
    parser.add_argument("--per-comp-timeout", type=int, default=0, help="Timeout per composition in seconds (0 disables)")
    parser.add_argument("--per-family-timeout", type=int, default=0, help="Timeout per family in seconds (0 disables)")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional max rows for quick test")
    parser.add_argument("--out-dir", default="featurize_database_for_cafer/prop_chunks", help="Output directory")
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

    # Group by active element family across the whole dataset
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

    # Parallelize by family (one TC session per family), write as each finishes
    with ProcessPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futures = {
            ex.submit(_compute_family_tc, (active_el, rows, args.per_comp_timeout, args.per_family_timeout)): active_el
            for active_el, rows in families.items()
        }
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
            results = {idx: (np.nan, np.nan, False) for idx, _ in rows}
            try:
                for idx, sol_k, liq_k, timed_out in fut.result():
                    results[idx] = (sol_k, liq_k, timed_out)
            except Exception:
                pass

            family_idx = [idx for idx, _ in rows]
            family_df = df.loc[family_idx].copy()
            family_df["Solidus_K_TC"] = [results[i][0] for i in family_idx]
            family_df["Liquidus_K_TC"] = [results[i][1] for i in family_idx]
            family_df["Solidus_C_TC"] = family_df["Solidus_K_TC"] - 273.15
            family_df["Liquidus_C_TC"] = family_df["Liquidus_K_TC"] - 273.15
            family_df["ST_LT_TIMEOUT"] = [results[i][2] for i in family_idx]

            max_chunk = max(1, args.family_chunk_size)
            fam_tag = "-".join(active_el)
            for start in range(0, len(family_df), max_chunk):
                end = min(start + max_chunk, len(family_df))
                chunk = family_df.iloc[start:end].copy()
                out_path = out_dir / f"solidus_liquidus_{fam_tag}_{start:05d}_{end-1:05d}.csv"
                chunk.to_csv(out_path, index=False)
                print(f"Wrote {out_path}")
