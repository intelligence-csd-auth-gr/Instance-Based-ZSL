# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:50:46 2020

@author: stam
"""

import os, pickle
path = ... #define the path for pre-computed files  
os.chdir(path)


# load values with max or other selected function per instance and label    

z = 'max_values_per_top100labels'
with open(z + ".pickle", "rb") as f:
    			max_l_values = pickle.load(f)
f.close()    

#import copy
#max_l_values = copy.deepcopy(l) 

file = open("top_100_labels.txt")
labels=list()
for line in file:
    labels.append(line[:-1])



# unweighted version
    
files = ['label_dependence_results_top100labels_pureZSL_mode_44kranking.pickle', 'label_dependence_results_top100labels_pureZSL_mode_44kranking_shuffled_70percent.pickle', 'label_dependence_results_top100labels_pureZSL_mode_44kranking_shuffled_70percent_plus_noise.pickle']
approaches = ['label_dependence_max_NN_bioBERT_44k_decisions_scores_official.pickle', 'label_dependence_max_NN_bioBERT_44k_decisions_scores_shuffled_70percent_official.pickle' , 'label_dependence_max_NN_bioBERT_44k_decisions_scores_shuffled_70percent_plus_noise_official.pickle']

for pos,f in enumerate(files):
    
    with open(f, "rb") as f:
        			decisions, isolated_predictions, positions, rank_info = pickle.load(f)
    f.close() 


    label_dependence = []
    for i in range(0, len(isolated_predictions)):
        
        if type(isolated_predictions[i]) == list:
            label_dependence.append(['Empty']) # i put the previous insead of random choice
            print(i)
            continue
        else:
            q = isolated_predictions[i].max().rank(ascending = False) 
            label_dependence.append(list(q.sort_values(ascending=True).index))



    print(os.getcwd())
    with open(approaches[pos], 'wb') as handle:
         pickle.dump(label_dependence, handle)                
    handle.close()



# weighted version

files = ['label_dependence_results_top100labels_pureZSL_mode_44kranking.pickle', 'label_dependence_results_top100labels_pureZSL_mode_44kranking_shuffled_70percent.pickle', 'label_dependence_results_top100labels_pureZSL_mode_44kranking_shuffled_70percent_plus_noise.pickle']
approaches = ['label_dependence_max_weighted_NN_bioBERT_44k_decisions_scores_official.pickle', 'label_dependence_max_weighted_NN_bioBERT_44k_decisions_scores_shuffled_70percent_official.pickle', 'label_dependence_max_weighted_NN_bioBERT_44k_decisions_scores_shuffled_70percent_plus_noise_official.pickle']


for pos,f in enumerate(files):
    
    with open(f, "rb") as f:
        			decisions, isolated_predictions, positions, rank_info = pickle.load(f)
    f.close() 
    
    
    label_dependence = []
    for i in range(0, len(isolated_predictions)):
        if type(isolated_predictions[i]) == list:
            label_dependence.append(['Empty']) 
            print(i)
            continue
        else:
            qq = isolated_predictions[i].max() # could be replaced with .sum()
            for label in labels:
                qq.loc[label] = qq.loc[label] * max_l_values[label][i]
            
            q = qq.rank(ascending = False)
            label_dependence.append(list(q.sort_values(ascending=True).index))


    print(os.getcwd())
    
    with open(approaches[pos], 'wb') as handle:
         pickle.dump(label_dependence, handle)                
    handle.close()