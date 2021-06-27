# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:50:46 2020

@authors:
Nikos Mylonas   myloniko@csd.auth.gr
Stamatis Karlos stkarlos@csd.auth.gr
Grigorios Tsoumakas greg@csd.auth.gr
"""

import os, pickle
#path = ... #define the path for pre-computed files  
#os.chdir(path)

path = r'C:\Users\stam\Documents\git\Instance-Based-ZSL\pre-computed files'#... #define the path for pre-computed files  
os.chdir(path)
#%% load values with max or other selected function per instance and label    

z = 'max_values_per_top100labels'
with open(z + ".pickle", "rb") as f:
    			max_l_values = pickle.load(f)
f.close()    


file = open("top_100_labels.txt")
labels=list()
for line in file:
    labels.append(line[:-1])

#%% we load the file that is created from the record_label_similaritiy_scores.py
    
###### these are the files that are evaluated into the original work for mode1 (ideal oracle) #########

# unweighted version (LSSc with max label similarity) -> you can add the rest file names into the two next lists

files = ['label_dependence_results_top100labels_pureZSL_mode_ranking_total.pickle']
approaches = ['label_dependence_SW_unweighted_bioBERT_44k_decisions_scores_mode_ranking.pickle']


for pos,f in enumerate(files):
    
    with open(f, "rb") as f:
        decisions, isolated_predictions, positions, rank_info = pickle.load(f)
    f.close() 


    label_dependence = []
    for i in range(0, len(isolated_predictions)):
        
        if type(isolated_predictions[i]) == list:
            label_dependence.append(['Empty']) # i put the previous instead of random choice -> this is removed later
            print(i)
            continue
        else:
            q = isolated_predictions[i].max().rank(ascending = False) 
            label_dependence.append(list(q.sort_values(ascending=True).index))


    print(os.getcwd())
    with open(approaches[pos], 'wb') as handle:
         pickle.dump(label_dependence, handle)                
    handle.close()



# weighted version (RankSc with max label similarity and the selected similarity between labels and sentences) -> you can add the rest file names into the two next lists
kind = input('Give your input for the kind of the weighting stage between the labels and the sentences:\n1. max \n2. sum\n\n .. ')
if kind == '1':
    kind_str = 'max'
else:
    kind_str = 'sum'

files = ['label_dependence_results_top100labels_pureZSL_mode_ranking_total.pickle']
approaches = ['label_dependence_SW_weighted_' + kind_str + '_bioBERT_44k_decisions_scores_mode_ranking.pickle']

#os.chdir(r'D:\datasets\mode2')
#files = ['label_dependence_results_top100labels_pureZSL_mode_ranking_shuffled_70percent.pickle']
#approaches = ['label_dependence_weighted_' + kind_str + '_bioBERT_44k_decisions_scores_mode_ranking_shuffled_70percent.pickle']


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
            
            if kind == '1':
                qq = isolated_predictions[i].max()
            else:
                qq = isolated_predictions[i].sum()
                
            for label in labels:
                qq.loc[label] = qq.loc[label] * max_l_values[label][i]
            
            q = qq.rank(ascending = False)
            label_dependence.append(list(q.sort_values(ascending=True).index))


    print(os.getcwd())
    
    with open(approaches[pos], 'wb') as handle:
         pickle.dump(label_dependence, handle)                
    handle.close()