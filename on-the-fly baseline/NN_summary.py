# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 10:44:38 2020

@authors:
Nikos Mylonas   myloniko@csd.auth.gr
Stamatis Karlos stkarlos@csd.auth.gr
Grigorios Tsoumakas greg@csd.auth.gr
"""

import os, pickle
import numpy as np

path = ... #define the path with the outputs of 'NN_baseline proprocess.py 
os.chdir(path)

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
            best_3_scores.append(np.array(k)[s][0:3])  # hold the 3 best values for profiling reasons, it is not used in SCIS pipeline. 

    return decisions, best_3_scores


#%% create summary files
    
argument = '44k'
    
files = os.listdir(os.getcwd())

full_decisions, full_decisions_torch = [], []
full_best_3_scores, full_best_3_scores_torch = [], []

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
    else:
        print(f)
        label_embeddings_torch, labels_torch = read_pickles(f)
        decisions_torch, best_3_scores_torch = sort_similarities(label_embeddings_torch, labels_torch)
    
    
        full_decisions_torch += decisions_torch
        full_best_3_scores_torch += best_3_scores_torch
        
        
print(os.getcwd())

with open('NN_bioBERT_' + argument + '_decisions_scores.pickle', 'wb') as handle:
     pickle.dump([full_decisions, full_best_3_scores], handle)                
handle.close()

with open('torch_NN_bioBERT_' + argument + '_decisions_scores.pickle', 'wb') as handle:
     pickle.dump([full_decisions_torch, full_best_3_scores_torch], handle)                
handle.close()