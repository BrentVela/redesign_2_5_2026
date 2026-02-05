import numpy as np
import pandas as pd
import time
tic = time.time()
import strength_model
import os.path as path
import os
from itertools import compress
import concurrent.futures

import numpy as np
import pandas as pd
from itertools import compress
import time
import concurrent.futures
from pickle import dump
import os.path as path
import numpy as np
import pandas as pd
import time
import logging
import datetime
from scipy.constants import R
import os
import sys
import numpy as np
import pandas as pd
from scipy.integrate import *
import scipy.version
import ctypes
import traceback


from itertools import combinations
from itertools import permutations




def writeToTracker(calcName, text):
    no_write = True
    while no_write:
        try:
            with open('{}-tracker.txt'.format(calcName), 'a') as f:
                f.write(text)
            no_write = False
        except:
            pass
    return

#Pre-Processing Data
def Property(param):
    indices = param["INDICES"]
    comp_df = param["COMP"]
    elements = param["ACT_EL"]

    # Hard-coded elemental properties
    prop_data = {
        'W': {'Atomic Weight [g/mol]': 183.84, 'Density [g/cm^3]': 19.25, 'Pauling Electronegativity': 2.36, 'Allen Electronegativity': 1.47, 'Melting Temperature [K]': 3695, 'Valence Electrons': 6},
        'Mo': {'Atomic Weight [g/mol]': 95.95, 'Density [g/cm^3]': 10.28, 'Pauling Electronegativity': 2.16, 'Allen Electronegativity': 1.47, 'Melting Temperature [K]': 2896, 'Valence Electrons': 6},
        'Ta': {'Atomic Weight [g/mol]': 180.95, 'Density [g/cm^3]': 16.69, 'Pauling Electronegativity': 1.5, 'Allen Electronegativity': 1.34, 'Melting Temperature [K]': 3290, 'Valence Electrons': 5},
        'Nb': {'Atomic Weight [g/mol]': 92.906, 'Density [g/cm^3]': 8.57, 'Pauling Electronegativity': 1.6, 'Allen Electronegativity': 1.41, 'Melting Temperature [K]': 2750, 'Valence Electrons': 5},
        'V': {'Atomic Weight [g/mol]': 50.942, 'Density [g/cm^3]': 6.11, 'Pauling Electronegativity': 1.63, 'Allen Electronegativity': 1.53, 'Melting Temperature [K]': 2183, 'Valence Electrons': 5},
        'Al': {'Atomic Weight [g/mol]': 26.982, 'Density [g/cm^3]': 2.7, 'Pauling Electronegativity': 1.61, 'Allen Electronegativity': 1.613, 'Melting Temperature [K]': 933, 'Valence Electrons': 3},
        'Ti': {'Atomic Weight [g/mol]': 47.867, 'Density [g/cm^3]': 4.506, 'Pauling Electronegativity': 1.54, 'Allen Electronegativity': 1.38, 'Melting Temperature [K]': 1941, 'Valence Electrons': 4},
        'Zr': {'Atomic Weight [g/mol]': 91.224, 'Density [g/cm^3]': 6.52, 'Pauling Electronegativity': 1.33, 'Allen Electronegativity': 1.32, 'Melting Temperature [K]': 2128, 'Valence Electrons': 4},
        'Hf': {'Atomic Weight [g/mol]': 178.49, 'Density [g/cm^3]': 13.31, 'Pauling Electronegativity': 1.3, 'Allen Electronegativity': 1.16, 'Melting Temperature [K]': 2506, 'Valence Electrons': 4},
        'Cr': {'Atomic Weight [g/mol]': 51.996, 'Density [g/cm^3]': 7.15, 'Pauling Electronegativity': 1.66, 'Allen Electronegativity': 1.65, 'Melting Temperature [K]': 2180, 'Valence Electrons': 6},
        'Re': {'Atomic Weight [g/mol]': 186.207, 'Density [g/cm^3]': 21.02, 'Pauling Electronegativity': 1.9, 'Allen Electronegativity': 1.6, 'Melting Temperature [K]': 3459, 'Valence Electrons': 7},
        'Ru': {'Atomic Weight [g/mol]': 101.07, 'Density [g/cm^3]': 12.37, 'Pauling Electronegativity': 2.2, 'Allen Electronegativity': 1.54, 'Melting Temperature [K]': 2607.15, 'Valence Electrons': 8},
        'Fe': {'Atomic Weight [g/mol]': 55.845, 'Density [g/cm^3]': 7.874, 'Pauling Electronegativity': 1.83, 'Allen Electronegativity': 1.8, 'Melting Temperature [K]': 1811, 'Valence Electrons': 8},
        'Ni': {'Atomic Weight [g/mol]': 58.6934, 'Density [g/cm^3]': 8.908, 'Pauling Electronegativity': 1.91, 'Allen Electronegativity': 1.88, 'Melting Temperature [K]': 1728, 'Valence Electrons': 10},
        'Co': {'Atomic Weight [g/mol]': 58.933194, 'Density [g/cm^3]': 8.9, 'Pauling Electronegativity': 1.88, 'Allen Electronegativity': 1.84, 'Melting Temperature [K]': 1768, 'Valence Electrons': 9},
        'Mn': {'Atomic Weight [g/mol]': 54.938044, 'Density [g/cm^3]': 7.47, 'Pauling Electronegativity': 1.55, 'Allen Electronegativity': 1.75, 'Melting Temperature [K]': 2334, 'Valence Electrons': 7},
        'Cu': {'Atomic Weight [g/mol]': 63.546, 'Density [g/cm^3]': 8.96, 'Pauling Electronegativity': 1.9, 'Allen Electronegativity': 1.85, 'Melting Temperature [K]': 1358, 'Valence Electrons': 11}
    }

    # Hard-coded elastic constants
    elast_data = {
        'W': {'C11': 517.8, 'C12': 201.7, 'C44': 139.4, 'B': 306.4, 'G': 139.4, 'B*': 310, 'G*': 161, 'V*': 0.28, 'Rm': 140.8},
        'Mo': {'C11': 466, 'C12': 165.2, 'C44': 99.5, 'B': 265.8, 'G': 99.5, 'B*': 230, 'G*': 20, 'V*': 0.31, 'Rm': 140},
        'Ta': {'C11': 260.9, 'C12': 165.2, 'C44': 70.4, 'B': 197.1, 'G': 70.4, 'B*': 200, 'G*': 67, 'V*': 0.34, 'Rm': 146.7},
        'Nb': {'C11': 247.2, 'C12': 140, 'C44': 14.2, 'B': 175.7, 'G': 14.2, 'B*': 170, 'G*': 38, 'V*': 0.4, 'Rm': 146.8},
        'V': {'C11': 272, 'C12': 144.8, 'C44': 17.6, 'B': 187.2, 'G': 17.6, 'B*': 160, 'G*': 47, 'V*': 0.37, 'Rm': 134.6},
        'Al': {'C11': 38.7, 'C12': 79.2, 'C44': 33, 'B': 65.7, 'G': 33, 'B*': 76, 'G*': 26, 'V*': 0.35, 'Rm': 143.2},
        'Ti': {'C11': 95.9, 'C12': 115.9, 'C44': 40.3, 'B': 109.2, 'G': 40.3, 'B*': 110, 'G*': 44, 'V*': 0.32, 'Rm': 146.2},
        'Zr': {'C11': 81.8, 'C12': 94.3, 'C44': 30.2, 'B': 90.1, 'G': 30.2, 'B*': 91.1, 'G*': 33, 'V*': 0.34, 'Rm': 160.2},
        'Hf': {'C11': 73.7, 'C12': 117, 'C44': 51.7, 'B': 102.8, 'G': 51.7, 'B*': 110, 'G*': 30, 'V*': 0.37, 'Rm': 158},
        'Cr': {'C11': 247.6, 'C12': 73.4, 'C44': 48.3, 'B': 131.5, 'G': 48.3, 'B*': 160, 'G*': 115, 'V*': 0.21, 'Rm': 136},
        'Re': {'C11': 325, 'C12': 380.2, 'C44': 158.6, 'B': 361.5, 'G': 158.6, 'B*': 324, 'G*': 185, 'V*': 0.3, 'Rm': 137},
        'Ru': {'C11': 46.6, 'C12': 401.1, 'C44': 173.4, 'B': 283.1, 'G': 173.4, 'B*': None, 'G*': None, 'V*': None, 'Rm': 134},
        'Fe': {'C11': 279.2, 'C12': 148.8, 'C44': 93, 'B': 192.3, 'G': 93, 'B*': 170, 'G*': 82, 'V*': 0.291, 'Rm': 127.4},
        'Ni': {'C11': 148.6, 'C12': 214.3, 'C44': 151.7, 'B': 192.4, 'G': 151.7, 'B*': 181, 'G*': 79, 'V*': 0.31, 'Rm': 124.6},
        'Co': {'C11': 129.3, 'C12': 140.9, 'C44': 93.5, 'B': 136.9, 'G': 93.5, 'B*': 193, 'G*': 74, 'V*': 0.32, 'Rm': 125.2},
        'Mn': {'C11': 256.9, 'C12': 272.2, 'C44': 105.4, 'B': 267.1, 'G': 105.4, 'B*': 92.6, 'G*': 76.4, 'V*': 0.35, 'Rm': 135},
        'Cu': {'C11': 129.3, 'C12': 140.9, 'C44': 93.5, 'B': 136.9, 'G': 93.5, 'B*': 137.5, 'G*': 46.5, 'V*': 0.34, 'Rm': 127.8}
    }

    # Get Elemental lists
    el_C11 = [elast_data[el]['C11'] for el in elements]
    el_C12 = [elast_data[el]['C12'] for el in elements]
    el_C44 = [elast_data[el]['C44'] for el in elements]
    el_B = [elast_data[el]['B'] for el in elements]
    el_G = [elast_data[el]['G'] for el in elements]
    el_ve = [prop_data[el]['Valence Electrons'] for el in elements]
    el_pen = [prop_data[el]['Pauling Electronegativity'] for el in elements]

    el_EN = [prop_data[el]['Allen Electronegativity'] for el in elements]
    el_density = [prop_data[el]['Density [g/cm^3]'] for el in elements]
    el_Tm = [prop_data[el]['Melting Temperature [K]'] for el in elements]
    el_mw = [prop_data[el]['Atomic Weight [g/mol]'] for el in elements]

    density_avg = np.dot(comp_df[elements], el_density)
    comp_df['Density Avg'] = density_avg

    mw_avg = np.dot(comp_df[elements], el_mw)
    comp_df['MW Avg'] = mw_avg

    # Calculate Melting Temperature
    Tm_avg = np.dot(comp_df[elements], el_Tm)
    comp_df['Tm Avg'] = Tm_avg

    # Calculate Valence Electron Concentration
    ve_avg = np.dot(comp_df[elements], el_ve)
    comp_df['VEC Avg'] = ve_avg

    # Calculate Average Elastic Constants C11 C12 C44
    C11_avg = np.dot(comp_df[elements], el_C11)
    C12_avg = np.dot(comp_df[elements], el_C12)
    C44_avg = np.dot(comp_df[elements], el_C44)

    el_Br = [elast_data[el]['B*'] for el in elements]
    el_Gr = [elast_data[el]['G*'] for el in elements]
    el_Vr = [elast_data[el]['V*'] for el in elements]
    el_R = [elast_data[el]['Rm'] for el in elements]

    B_avgr = np.dot(comp_df[elements], el_Br)
    G_avgr = np.dot(comp_df[elements], el_Gr)
    V_avgr = np.dot(comp_df[elements], el_Vr)

    # Prashant's method
    E_avgr3 = 3 * B_avgr * (1 - 2 * V_avgr)
    G_avgr3 = (1 / 2) * E_avgr3 / (1 + V_avgr)
    Pugh_Ration_avg3 = B_avgr / G_avgr3

    comp_df['Pugh_Ratio_PRIOR'] = Pugh_Ration_avg3
    comp_df['V_avgr'] = V_avgr
    comp_df['B_avgr'] = B_avgr
    comp_df['G_avgr PS'] = G_avgr3
    comp_df['E_avgr PS'] = E_avgr3
    comp_df['G_avgr'] = G_avgr

    # Calculate Average Cauchy Pressure
    Cauchy_Pres_avg = C12_avg - C44_avg
    comp_df['C11'] = C11_avg
    comp_df['C12'] = C12_avg
    comp_df['C44'] = C44_avg
    comp_df['Cauchy Pres Avg'] = Cauchy_Pres_avg

    # Calculate Average Zener Ratio
    ar = 2 * C44_avg / (C11_avg - C12_avg)
    comp_df['Zener Ratio'] = ar

    # Calculate Universal Anisotropy
    au = (6 / 5) * (comp_df['Zener Ratio'].apply(np.sqrt) - (1 / comp_df['Zener Ratio'].apply(np.sqrt))) ** 2
    comp_df['Universal Anisotropy'] = au

    # Calculate Average G and B
    B_avg = np.dot(comp_df[elements], el_B)
    G_avg = np.dot(comp_df[elements], el_G)
    comp_df['Bulk Modulus Avg'] = B_avg
    comp_df['Shear Modulus Avg'] = G_avg

    toc = time.time()

    # Pauling Electroneg
    pen_avg = np.dot(comp_df[elements], el_pen)
    comp_df['Pauling Electronegativity Avg'] = pen_avg

    # Metallic Radius
    comp_df['R'] = np.dot(comp_df[elements], el_R)
    comp_df = comp_df.reset_index()
    for i in range(len(comp_df)):
        print(comp_df)
        comp_i = comp_df.iloc[i][elements]
        test_temp = comp_df.iloc[i]['Test temperature']
        
        # Calculate Configurational Entropy
        Sconfi = 0
        for j in range(len(comp_i)):
            if comp_i[j] > 0:
                Sconfi = Sconfi + comp_i[j] * np.log(comp_i[j])
        Sconf = -Sconfi
        comp_df.at[i, 'Sconf'] = Sconf
       
        # Calculate Strength from Curtin Model
        [tau_y_0, delta_Eb, tau_1573] = strength_model.model_Control(elements, comp_i, T=25)

        comp_df.at[i, 'Tau_y 0'] = tau_y_0
        comp_df.at[i, 'delta Eb'] = delta_Eb
        comp_df.at[i, 'Tau_y 25C'] = tau_1573
        comp_df.at[i, 'YS 25-273C PRIOR'] = 3000 * tau_1573
        comp_df.at[i, 'HV 25-273C PRIOR'] = 3000 * tau_1573 * (3 / 9.807) + 150

        # Calculate Strength from Curtin Model
        [tau_y_0, delta_Eb, tau_1573] = strength_model.model_Control(elements, comp_i, T=25 + 273)

        comp_df.at[i, 'Tau_y 0'] = tau_y_0
        comp_df.at[i, 'delta Eb'] = delta_Eb
        comp_df.at[i, 'Tau_y 25C'] = tau_1573
        comp_df.at[i, 'YS 25C PRIOR'] = 3000 * tau_1573
        comp_df.at[i, 'HV 25C PRIOR'] = 3000 * tau_1573 * (3 / 9.807) + 150

        # Calculate Strength from Curtin Model
        [tau_y_0, delta_Eb, tau_1573] = strength_model.model_Control(elements, comp_i, T=test_temp + 273)

        comp_df.at[i, 'Tau_y 0'] = tau_y_0
        comp_df.at[i, 'delta Eb'] = delta_Eb
        comp_df.at[i, 'Tau_y 25C'] = tau_1573
        comp_df.at[i, 'YS T C PRIOR'] = 3000 * tau_1573
        comp_df.at[i, 'HV T C PRIOR'] = 3000 * tau_1573 * (3 / 9.807) + 150

        [tau_y_0, delta_Eb, tau_ht] = strength_model.model_Control(elements, comp_i, T=test_temp)
        comp_df.at[i, 'Tau_y 25-273C'] = tau_ht
        comp_df.at[i, 'YS T-273C PRIOR'] = 3000 * tau_ht
        comp_df.at[i, 'HV T-273C PRIOR'] = 3000 * tau_ht * (3 / 9.807) + 150

    comp_df.to_csv('CalcFiles/STOIC_OUT_{}.csv'.format(param["INDICES"][0]),index=False)
    print('Done.')
    print('Done in ' + str(round(toc - tic, 3)) + ' sec')
    return None


if __name__ == '__main__':


    ##########################################################################
    results_df = pd.read_parquet('SPACE_v0.par')
    
    results_df['Test temperature'] = 1300 #C
    elements =  sorted(['Cr','Mo','Nb','Ta','Ti','V','W'])

    elements = sorted(elements)
    savename = 'prop_out'
    ##########################################################################

    if not path.exists("CalcFiles"):
        os.mkdir("CalcFiles")

    writeToTracker('PROP',"*****Start Generating Calculation Sets*****\n")
    results_df.reset_index()
    indices = results_df.index

    prev_active_el = []
    parameters = []
    count = 0
    print(results_df)
    for i in indices:
        comp = results_df.loc[i][elements]
        active_el = list(compress(elements, list(comp > 0)))
        # print(i, prev_active_el, active_el)

        if (active_el != prev_active_el) or (count == 100):
            try:
                new_calc_dict["COMP"] = results_df.loc[new_calc_dict["INDICES"]]
                new_calc_dict["ACT_EL"] = prev_active_el
                if not os.path.exists(
                        "CalcFiles/{}-Results-Set-{}".format('EQUIL', new_calc_dict["INDICES"][0])):
                    parameters.append(new_calc_dict)
                    writeToTracker('PROP', "******Calculation Added to list: Start Index {} \n".format(
                        new_calc_dict["INDICES"][0]))
                    print(i)
                else:
                    writeToTracker('PROP', "******Calculation Already Completed: Start Index {} \n".format(
                        new_calc_dict["INDICES"][0]))
                new_calc_dict = {"INDICES": [],
                                 "COMP": [],
                                 "ACT_EL": []}
            except Exception as e:
                new_calc_dict = {"INDICES": [],
                                 "COMP": [],
                                 "ACT_EL": []}
            count = 0

        new_calc_dict["INDICES"].append(i)
        prev_active_el = active_el
        count += 1

    # add the last calculation set
    new_calc_dict["COMP"] = results_df.loc[new_calc_dict["INDICES"]]
    new_calc_dict["ACT_EL"] = prev_active_el
    if not os.path.exists("CalcFiles/{}-Results-Set-{}".format('PROP', new_calc_dict["INDICES"][0])):
        parameters.append(new_calc_dict)
        writeToTracker('PROP', "**Calculation Added to list: Start Index {} \n".format(new_calc_dict["INDICES"][0]))
    else:
        writeToTracker('PROP',
                       "**Calculation Already Completed: Start Index {} \n".format(new_calc_dict["INDICES"][0]))

    writeToTracker('PROP', "*****Calculation Sets Generated*****\n")

    completed_calculations = []
    print('Parameters:')
    print(parameters)
    with concurrent.futures.ProcessPoolExecutor(2) as executor:
        for result_from_process in zip(parameters, executor.map(Property, parameters)):
            # params can be used to identify the process and its parameters
            params, results = result_from_process
            if results == "Calculation Completed":
                completed_calculations.append('Completed')


