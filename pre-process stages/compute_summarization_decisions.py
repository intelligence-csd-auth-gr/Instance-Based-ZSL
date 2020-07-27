# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 10:44:38 2020

@author: 
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
                k.append(np.max(d[instance][label]))  
                
        if flag:
            decisions.append(['None'])
            best_3_scores.append(['None'])
        else:
            s = sorted(range(len(k)), key=lambda pos: k[pos])
            s.reverse()
            decisions.append(np.array(labels)[s])
            best_3_scores.append(np.array(k)[s][0:3])   

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
        
        print(f)
        label_embeddings, labels = read_pickles(f)
        decisions, best_3_scores = sort_similarities(label_embeddings, labels)
    
        full_decisions += decisions
        full_best_3_scores += best_3_scores
        name = 'NN_bioBERT_' + argument + '_decisions_scores.pickle'
        
    else:
        
        print(f)
        label_embeddings, labels = read_pickles(f)
        decisions, best_3_scores = sort_similarities(label_embeddings, labels)
    
        full_decisions += decisions
        full_best_3_scores += best_3_scores
        name = 'torch_NN_bioBERT_' + argument + '_decisions_scores.pickle'
        
        
print(os.getcwd())
with open(name, 'wb') as handle:
     pickle.dump([full_decisions, full_best_3_scores], handle)                
handle.close()

print('**Summarization has been completed***')