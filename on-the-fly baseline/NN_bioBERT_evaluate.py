# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 10:44:38 2020

@author: stam
"""

import os, pickle
import numpy as np
import pandas as pd

os.chdir(r'D:\NN_bioBERT')


def read_pickles(file):
    with open(file, "rb") as f:
            label_embeddings = pickle.load(f)
    f.close()
    
    return label_embeddings, list(label_embeddings['instance0'].keys()) # second argument is a list of labels 


def sort_similarities(d, labels):
    
    decisions = []
    best_3_scores = []
    for instance in label_embeddings.keys():
        k = []
        flag = False
        for label in labels:
            if d[instance][label] == []:
                flag = True
                print(instance)
                break
            else:
                k.append(np.max(d[instance][label]))  # min for clustering ----- max for cosine similarity
                
        if flag:
            decisions.append(['None'])
            best_3_scores.append(['None'])
        else:
            s = sorted(range(len(k)), key=lambda pos: k[pos])
            s.reverse()
            decisions.append(np.array(labels)[s])
            best_3_scores.append(np.array(k)[s][0:3])   # [0:3] for clustering  ------ [-3:] for cosine similarity 

    return decisions, best_3_scores

#%% create NN_bioBERT_44k_decisions_scores.pickle and torch_NN_bioBERT_44k_decisions_scores.pickle
    
argument = '44k'
    
files = os.listdir(os.getcwd())

full_decisions = []
full_best_3_scores = []

for f in files:
    
    decisions = []
    best_3_scores = []
    
    if argument not in f:
        continue
    elif 'torch' not in f:
        continue
    else:
        print(f)
        label_embeddings, labels = read_pickles(f)
        decisions, best_3_scores = sort_similarities(label_embeddings, labels)
    
        full_decisions += decisions
        full_best_3_scores += best_3_scores
        
        
print(os.getcwd())
with open('torch_NN_bioBERT_' + argument + '_decisions_scores.pickle', 'wb') as handle:
     pickle.dump([full_decisions, full_best_3_scores], handle)                
handle.close()
#%%

def bring_new_y():
    
    os.chdir(r'D:\BioASQ\evaluate_py')
    test_file = "pure_zero_shot_test_set_top100.txt"
    
    y = []
    file = open(test_file)
    for line in file:
    	y.append(line[2:-2].split("labels: #")[1])
    print(len(y))
    
    file = open("top_100_labels.txt")
    labels=list()
    for line in file:
    	labels.append(line[:-1])
        
    new_y = []
    known_y = []
    for label_y in y:
    	string = ""
    	flag = "false"
    	string_known=""
    	for label in label_y.split("#"):
    		if label in labels:
    			flag = "true"
    			string = string + label + "#"
    		else:
    			string_known=string_known+label+"#"
    	if (flag == "false"):
    		string = "None#"
    	new_y.append(string[:-1].split('#'))
    	known_y.append(string_known[:-1].split('#'))
     
    return new_y


def coverage_custom(y_test, y_pred):
    ''' Input:
            y_test: vector of ground truth
            y_pred: vector of predictions
        
        Return:
            Coverage score per instance'''
            
    k = []
    for y in range(0, len(y_test)):
        
        kk = []
        for i in range(0, len(y_test[y])):
            
            kk.append( list(y_pred[y]).index(y_test[y][i]) )
        
        k.append(max(kk))
        
    return k

#%%  if you want to load the summarization files

os.chdir(r'D:\NN_bioBERT_summary')
files = os.listdir(os.getcwd())
print(files)

which = files[2] #[0] or [2]
with open(which, "rb") as f:
    full_decisions, full_best_3_scores = pickle.load(f)
f.close()


which = 'torch_NN_bioBERT_test_set_decisions_scores_official' 
NN_bioBERT_44k_decisions = full_decisions[:] 


#%%

y_test = bring_new_y()[:]

k = coverage_custom(y_test, NN_bioBERT_44k_decisions)
print(np.round(np.mean(k),3) , len(k))
pd.DataFrame(k).hist(bins = 50)

bin_range = np.array([0,1,3,10,30,50,100])
out, bins  = pd.cut(k, bins=bin_range, include_lowest=True, right=False, retbins=True)
out.value_counts()
pd.DataFrame(out.value_counts()).to_csv(which + '_bins.csv')


#%%
    
os.chdir(r'D:\BioASQ\evaluate_py')
print(os.getcwd())
zz = 'label_dependence_max_weighted_NN_bioBERT_44k_decisions_scores'
#zz = 'label_dependence_max3_weighted_NN_bioBERT_44k_decisions_scores'
#zz = 'label_dependence_sum_distribution_NN_bioBERT_44k_decisions_scores'
zz = 'label_dependence_max_with_occurebne_weighted_NN_bioBERT_44k_decisions_scores'
zz = 'label_dependence_NN_bioBERT_44k_decisions_scores'
zz = 'label_dependence_max_NN_bioBERT_44k_decisions_scores_shuffled_70percent'
zz = 'label_dependence_max_weighted_NN_bioBERT_44k_decisions_scores_shuffled_70percent'
zz = 'label_dependence_max_weighted_with_occurence_NN_bioBERT_44k_decisions_scores_shuffled_70percent'
zz = 'label_dependence_max_NN_bioBERT_44k_decisions_scores_shuffled_70percent_plus_noise'
zz = 'label_dependence_max_weighted_NN_bioBERT_44k_decisions_scores_shuffled_70percent_plus_noise'
zz = 'label_dependence_max_weighted_with_occurence_NN_bioBERT_44k_decisions_scores_shuffled_70percent_plus_noise'

with open(zz + ".pickle", "rb") as f:
    				label_dependence = pickle.load(f)
f.close()

which = zz
NN_bioBERT_44k_decisions = label_dependence
#%%
which = 'label_dependence_max_weighted'
#which = 'label_dependence_max3_weighted'
#which = 'label_dependence_sum_distribution'
which = 'label_dependence_max_with_occurence_weighted'
which = 'label_dependence_NN_bioBERT'
which = 'label_dependence_max_NN_bioBERT_44k_decisions_scores_shuffled_70percent'
which = 'label_dependence_max_weighted_NN_bioBERT_44k_decisions_scores_shuffled_70percent'
which = 'label_dependence_max_weighted_with_occurence_NN_bioBERT_44k_decisions_scores_shuffled_70percent'
which = 'label_dependence_max_NN_bioBERT_44k_decisions_scores_shuffled_70percent_plus_noise'
which = 'label_dependence_max_weighted_NN_bioBERT_44k_decisions_scores_shuffled_70percent_plus_noise'
which = 'label_dependence_max_weighted_with_occurence_NN_bioBERT_44k_decisions_scores_shuffled_70percent_plus_noise'


