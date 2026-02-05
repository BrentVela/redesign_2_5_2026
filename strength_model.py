import numpy as np
import pandas as pd

def model(elements, alpha=1/12, prop_dfs=[]):
    # Hard-coded elastic constants
    elast_const_data = {
        'W': [517.8, 201.7, 139.4],
        'Mo': [466, 165.2, 99.5],
        'Ta': [260.9, 165.2, 70.4],
        'Nb': [247.2, 140, 14.2],
        'V': [272, 144.8, 17.6],
        'Al': [38.7, 79.2, 33],
        'Ti': [95.9, 115.9, 40.3],
        'Zr': [81.8, 94.3, 30.2],
        'Hf': [73.7, 117, 51.7],
        'Cr': [247.6, 73.4, 48.3],
        'Fe': [279.2, 148.8, 93],
        'Ni': [148.6, 214.3, 151.7],
        'Co': [129.3, 140.9, 93.5],
        'Re': [325, 380.2, 158.6],
        'Ru': [46.6, 401.1, 173.4],
        'Cu': [129.3, 140.9, 93.5],
        'Mn': [256.9, 272.2, 105.4]
    }

    # Hard-coded volumes #cubic anstroms
    volumes_data = {
        'W': 16.229,
        'Mo': 15.956,
        'Ta': 18.313,
        'Nb': 18.342,
        'V': 13.453,
        'Al': 17.08,
        'Ti': 17.123,
        'Zr': 22.885,
        'Hf': 22.128,
        'Cr': 11.575,
        'Fe': 11.358,
        'Ni': 11.012,
        'Co': 11.07,
        'Re': 15.135,
        'Cu': 12.077,
        'Mn': 10.985,
        'Ru': 14.348
    }

    # Use hard-coded data if no external data is provided
    if not prop_dfs:
        elast_const = elast_const_data
        volumes = volumes_data
    else:
        elast_const = prop_dfs[0]
        volumes = prop_dfs[1]

    # Initialize averages
    bar_C11 = 0
    bar_C12 = 0
    bar_C44 = 0
    bar_V = 0

    phases_record = []
    el_for_total = []
    fr_for_total = []

    for ele in elements:
        bar_C11 += elast_const[ele][0] * elements[ele]['fraction']
        bar_C12 += elast_const[ele][1] * elements[ele]['fraction']
        bar_C44 += elast_const[ele][2] * elements[ele]['fraction']

        el_for_total.append(ele)
        fr_for_total.append(elements[ele]['fraction'])

        elements[ele]['BCCVol'] = volumes[ele]
        bar_V += elements[ele]['BCCVol'] * elements[ele]['fraction']

    misfit_Vol_Factor = 0
    misfit = {}

    for ele in elements:
        misfit[ele] = (elements[ele]['BCCVol'] - bar_V)
        misfit_Vol_Factor += elements[ele]['fraction'] * ((elements[ele]['BCCVol'] - bar_V) ** 2)

    mu_bar = np.sqrt(0.5 * (bar_C44) * (bar_C11 - bar_C12))
    B_bar = (bar_C11 + 2 * bar_C12) / 3
    nu_bar = (3 * B_bar - 2 * mu_bar) / (2 * (3 * B_bar + mu_bar))

    unitCell_a = (2 * bar_V) ** (1 / 3)
    b_bar = (unitCell_a * np.sqrt(3) / 2)

    tau_y_zero = 0.040 * (alpha ** (-1 / 3)) * mu_bar * (((1 + nu_bar) / (1 - nu_bar)) ** (4 / 3)) * (
                (misfit_Vol_Factor / (b_bar ** 6)) ** (2 / 3))

    delta_E_b = 2.00 * (alpha ** (1 / 3)) * mu_bar * (b_bar ** 3) * (((1 + nu_bar) / (1 - nu_bar)) ** (2 / 3)) * (
                (misfit_Vol_Factor / (b_bar ** 6)) ** (1 / 3))

    output = {
        'tau_y_0': tau_y_zero,
        'delta_Eb': delta_E_b / 160.2176621,
        'Average C': [bar_C11, bar_C12, bar_C44],
        'misfit': misfit,
        'a': unitCell_a,
        'bar_V': bar_V,
        'phases_record': phases_record,
        'b_bar': b_bar,
        'mu_bar': mu_bar,
        'nu_bar': nu_bar
    }
    return output

def temp_model(results, eps_dot, approx_model=False, T=1573):
    eps_dot_0 = 1e4
    k = 8.617e-5
    if approx_model:
        tau = results['tau_y_0'] * np.exp(-(1 / 0.55) * ((((k * T) / (results['delta_Eb'])) * np.log(eps_dot_0 / eps_dot)) ** 0.91))
    else:
        tau_low = results['tau_y_0'] * (1 - (((k * T) / (results['delta_Eb'])) * np.log(eps_dot_0 / eps_dot)) ** (2 / 3))
        tau_high = results['tau_y_0'] * np.exp(-(1 / 0.55) * ((k * T) / (results['delta_Eb'])) * np.log(eps_dot_0 / eps_dot))
        tau = []
        for i in range(len(T)):
            if tau_low[i] / results['tau_y_0'] > 0.5:
                tau.append(tau_low[i])
            else:
                tau.append(tau_high[i])
        tau = np.array(tau)
    return tau

def model_Control(element_list, comp_list, T=1573, prop_dfs=[]):
    model_input = {}
    for i in range(len(element_list)):
        model_input[element_list[i]] = {'fraction': comp_list[i]}

    result = model(model_input, prop_dfs=prop_dfs)
    out = temp_model(result, 0.001, True, T)
    return (result['tau_y_0'], result['delta_Eb'], out)

if __name__ == "__main__":
    # Example of using the strength model function below Cr5Mo45Nb30V10Ti10
    elements = ['Cr', 'Mo', 'Nb', 'V', 'Ti']
    comp = [0.05, 0.45, 0.3, 0.1, 0.1]

    out = model_Control(elements, comp)
    print(out[2] * 3000)