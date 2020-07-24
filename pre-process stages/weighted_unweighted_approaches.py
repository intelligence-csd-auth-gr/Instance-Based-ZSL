# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:50:46 2020

@author: stam
"""

import os, pickle
os.chdir(r'D:\BioASQ\evaluate_py')

z = 'label_dependence_results_top100labels_pureZSL_mode_44kranking.pickle'
z = 'label_dependence_results_top100labels_pureZSL_mode_44kranking_shuffled_70percent.pickle'
z = 'label_dependence_results_top100labels_pureZSL_mode_44kranking_shuffled_70percent_plus_noise.pickle'

with open(z, "rb") as f:
    			decisions, isolated_predictions, positions, rank_info = pickle.load(f)
f.close() 

#%% unweighted version

label_dependence = []
for i in range(0, len(isolated_predictions)):
    if type(isolated_predictions[i]) == list:#['None']:
        label_dependence.append(list(q.sort_values(ascending=True).index)) # i put the previous insead of random choice
        print(i)
        continue
    #q = isolated_predictions[i].max().rank(ascending = False) # sum or max  #biggest has rank 1
    #q = isolated_predictions[i].var().rank(ascending = False) # var des
    q = isolated_predictions[i].var().rank(ascending = True) # var asc

    label_dependence.append(list(q.sort_values(ascending=True).index))



print(os.getcwd())
name = 'label_dependence_var_NN_bioBERT_44k_decisions_scores_official.pickle'
name = 'label_dependence_var_NN_bioBERT_44k_decisions_scores_shuffled_70percent_official.pickle'
name = 'label_dependence_var_NN_bioBERT_44k_decisions_scores_shuffled_70percent_plus_noise_official.pickle'
with open(name, 'wb') as handle:
     pickle.dump(label_dependence, handle)                
handle.close()
#%%  load values with max or other selected function per instance and label    


z = 'max_values_per_top100labels_44k'
with open(z + ".pickle", "rb") as f:
    			l	 = pickle.load(f)
f.close()    

import copy
max_l_values = copy.deepcopy(l) 


file = open("top_100_labels.txt")
labels=list()
for line in file:
    labels.append(line[:-1])
#%% weighted version

label_dependence = []
for i in range(0, len(isolated_predictions)):
    if type(isolated_predictions[i]) == list:#['None']:
        label_dependence.append(list(q.sort_values(ascending=True).index)) # i put the previous insead of random choice
        print(i)
        continue
    qq = isolated_predictions[i].sum()
    for label in labels:
        qq.loc[label] = qq.loc[label] * max_l_values[label][i]
    q = qq.rank(ascending = False) # sum or max  #biggest has rank 1
    #q = isolated_predictions[i].var().rank(ascending = False) # var des
    #q = isolated_predictions[i].var().rank(ascending = True) # var asc

    label_dependence.append(list(q.sort_values(ascending=True).index))



print(os.getcwd())
name = 'label_dependence_sum_weighted_NN_bioBERT_44k_decisions_scores_official.pickle'
name = 'label_dependence_sum_weighted_NN_bioBERT_44k_decisions_scores_shuffled_70percent_official.pickle'
name = 'label_dependence_sum_weighted_NN_bioBERT_44k_decisions_scores_shuffled_70percent_plus_noise_official.pickle'
with open(name, 'wb') as handle:
     pickle.dump(label_dependence, handle)                
handle.close()