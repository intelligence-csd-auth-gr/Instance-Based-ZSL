# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 10:44:38 2020

@author: stam
"""

import os, pickle
import numpy as np
import pandas as pd


def bring_new_y(current_path):
    
    
    path = r'C:\Users\stam\Documents\git\AMULET_SCIS\pre-computed files'  #define the path for pre-computed files  
    os.chdir(path)
    
    with open('novel_labels_actual.pickle', 'rb') as handle:
        new_y = pickle.load(handle)                
    handle.close()
    
    os.chdir(current_path)
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
        
    return np.round(np.mean(k),3), k

def one_rank_custom(y_test, y_pred):
    ''' Input:
            y_test: vector of ground truth
            y_pred: vector of predictions
        
        Return:
            Coverage score per instance'''
            
    one_rank = []
    for y in range(0, len(y_test)):
        
        if y_pred[y][0] not in y_test[y]:
            
            one_rank.append(1)
        
    return np.round(sum(one_rank) / len(y_test), 3)


def remove_instances_with_empty_known_labels(y_test, y_preds, current_path):
    
    path = r'C:\Users\stam\Documents\git\AMULET_SCIS\pre-computed files'  #define the path for pre-computed files  
    os.chdir(path)
    
    with open('known_labels.pickle', 'rb') as handle:
        known_y = pickle.load(handle)                
    handle.close()
    
    y_test_new, y_preds_new = [], []
    for i in range(0, len(known_y)):
        if known_y[i] == ['']:
            continue
        else:
            y_test_new.append(y_test[i])
            y_preds_new.append(y_preds[i])
    
    os.chdir(current_path)
    return y_test_new, y_preds_new
    
#%%  you have to load the summarization files
path = r'D:\NN_bioBERT_summary_files'
os.chdir(path)
files = os.listdir(os.getcwd())
print(files)

approach = ['NN-bioBERT(Manhattan)', 'NN-bioBERT(Cosine)']
makeplot = True

for pos,f in enumerate(files):
    
    files = os.listdir(os.getcwd())

    with open(f, "rb") as f:
        full_decisions, full_best_3_scores = pickle.load(f)
    f.close()
    
    NN_bioBERT_44k_decisions = full_decisions[:] 
    y_test = bring_new_y(path)
    
    y_test, NN_bioBERT_44k_decisions = remove_instances_with_empty_known_labels(y_test, NN_bioBERT_44k_decisions, path)
    
    
    k, k_list = coverage_custom(y_test, NN_bioBERT_44k_decisions)
    one_rank = one_rank_custom(y_test, NN_bioBERT_44k_decisions)

    print('Approach: %s\n Coverage error:\t%4.3f \n One-error:\t\t%4.3f ' %(approach[pos], k, one_rank))
    
    if makeplot:
        pd.DataFrame(k_list).hist(bins = 100)
        bin_range = np.array([0,1,3,10,30,50,100])
        out, bins  = pd.cut(k_list, bins=bin_range, include_lowest=True, right=False, retbins=True)
        out.value_counts()


#%%
    
os.chdir(r'D:\BioASQ\evaluate_py')
print(os.getcwd())
#zz = 'label_dependence_max_NN_bioBERT_44k_decisions_scores_official.pickle'
zz = 'label_dependence_sum_weighted_NN_bioBERT_44k_decisions_scores_shuffled_70percent_plus_noise_official.pickle'

with open(zz, "rb") as f:
    				label_dependence = pickle.load(f)
f.close()

which = zz
NN_bioBERT_44k_decisions = label_dependence

#%%

os.chdir(r'D:\BioASQ\evaluate_py')

print(os.getcwd())
zz = 'occurence_label_dependence_sum_weighted_NN_bioBERT_44k_decisions_scores_official.pickle'
#zz = 'occurence_label_dependence_max_weighted_NN_bioBERT_44k_decisions_scores_shuffled_70percent_official.pickle'
#zz = 'occurence_label_dependence_max_weighted_NN_bioBERT_44k_decisions_scores_shuffled_70percent_plus_noise_official.pickle'

zz = 'occurence_modified_label_dependence_max_weighted_NN_bioBERT_44k_decisions_scores_official.pickle'
with open(zz, "rb") as f:
    				label_dependence = pickle.load(f)
f.close()

which = zz
NN_bioBERT_44k_decisions = label_dependence

y_test = bring_new_y()[:]

k = coverage_custom(y_test, NN_bioBERT_44k_decisions)
print(np.round(np.mean(k),3) , len(k))
pd.DataFrame(k).hist(bins = 50)

bin_range = np.array([0,1,3,10,30,50,100])
out, bins  = pd.cut(k, bins=bin_range, include_lowest=True, right=False, retbins=True)
out.value_counts()
pd.DataFrame(out.value_counts()).to_csv(which + '_bins.csv')