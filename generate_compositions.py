import numpy as np
import pandas as pd
import time
import itertools
from itertools import combinations
from itertools import compress
import time
tic = time.time()
##### USER INPUTS ###########################
elements = ['Co','Fe','Ni']

only_extra = False # only produce compositions with extra elements
elements = elements

n_comps = 20 # 100/20 = 5 number of compositions per 100 at%
sys_d = 3 # dimension of largest subsystem
#############################################

# Build dataframe of compositions

# Identify order of largest systems (max = number of elements) 
sys_list = []
for sys in combinations(elements,sys_d):
    sys_list.append(list(sys))

if only_extra:
    sys_list = [sys for sys in sys_list if set(sys).intersection(extra_elements)]

# Create grided samplng for arbitrary subsystem in sys_list (simplex in sys_d-dimensional space) (Number of samples --> (1/sys_d!)*(n_comps)**sys_d)
#         Fast but memory limitations
#    xi_list = [grid_inds for i in range(5)]
#    comp_grid = np.meshgrid(*xi_list, sparse = True)
#    comp_grid = np.meshgrid(grid_inds, grid_inds, grid_inds, grid_inds, grid_inds)
#    comps = np.transpose(np.vstack(map(np.ravel,comp_grid)))
#    comps = comps[(np.sum(comps,1)<(1+1e-9))&(np.sum(comps,1)>(1-1e-9)),:] # only keep points that sum to unity

#         Slow but no memory limitations (yet)...    
comps = []
indices = np.ndindex(*[n_comps for i in range(sys_d)])
j = 0
for index in indices:
    j += 1
    comp = list(index)
    # only keep compositions that sum to unity
    if (sum(comp)==(n_comps)):
        comps.append(comp)
        toc = time.time()
        print(str(round(j/(n_comps**sys_d)*100,2))+' % Done: Total Compositions = '+str(len(comps)) +' in '+str(round(toc-tic,3))+' secs')
comps = np.array(comps)/(n_comps)

# Create dataframe with each sampled composition
for si in range(len(sys_list)):
    s_els = sys_list[si]
    if si == 0:
        results_df = pd.DataFrame(comps, columns = s_els)
    else:
        new_df = pd.DataFrame(comps, columns = s_els)
        results_df = pd.concat([results_df, new_df], ignore_index=True)
    toc = time.time()
    print(str(round(si/len(sys_list)*100,3))+' % Done Adding Systems to Dataframe in '+str(round(toc-tic,3))+' secs')
    
results_df = results_df.fillna(0)
results_df = results_df.drop_duplicates() # Drop duplicate compositions
results_df = results_df.reset_index()
results_df = results_df.drop(columns = 'index')  

# Reorganize dataframe to have ordered sequence of element groups (to minimize Thermo-Calc inits)
Els = list()
prev_active_el = []
for row in range(results_df.shape[0]):
    comp = results_df.iloc[row][elements]
    active_el = list(compress(elements,list(comp>0)))
    if active_el not in Els:
        Els.append(active_el)
    prev_active_el = active_el
    toc = time.time()
    print(str(round(row/results_df.shape[0]*100,3))+' % Done Gathering Systems in '+str(round(toc-tic,3))+' secs')

for El_i in Els:
    cond = (np.all(results_df[El_i]>0, axis = 1))&(np.sum(results_df[El_i],1)>1-1e-9)&(np.sum(results_df[El_i],1)<1+1e-9)
    if El_i == Els[0]:
        results_df2 = results_df[cond]
    else:
        results_df2 = pd.concat([results_df2, results_df[cond]], ignore_index=True)

    toc = time.time()
    print(str(round(Els.index(El_i)/len(Els)*100,3))+' % Done Rearranging in '+str(round(toc-tic,3))+' secs')
results_df = results_df2
results_df = results_df.reset_index()
results_df = results_df.drop(columns = 'index')

#results_df = results_df.loc[results_df['W']>minW]
results_df = results_df.reset_index(drop=True)
results_df.to_csv('FeNiCo.csv')
print(len(results_df))


