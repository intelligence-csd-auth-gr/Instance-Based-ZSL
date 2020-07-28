# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 13:36:11 2020

@author: 
"""

import os, pickle
import numpy as np

def exploit_sent_emb(choice, x):
        if choice == 1:
            return np.round( max(x) , 4) 
        elif choice == 2:
            return np.round( np.mean(np.sort(x)[-3:]) , 4)
        elif choice == 3:
            return np.round(np.mean(x) , 4)
        elif choice == 4:
            return np.round( np.mean(np.sort(x)[-2:]) , 4)
        else:
            if len(x) >= 3:
                return np.round( np.mean(np.sort(x)[1:-1]) , 4)
            else:
                return np.round( np.mean(x) , 4)
            
        return
    
    
path = ... #define the path for pre-computed files  
os.chdir(path)
    
file = open("top_100_labels.txt")
labels = list()
for line in file:
   labels.append(line[:-1])

path = ... #define the path for .pickles obtained by the 'calculate_similarities.py'
os.chdir(path)

choice = input('what kind of sentence embeddings exploitation do you prefer? 1: max 2: max3 3: average 4: max2  5: all but min and max') # we use the first choice


l = {}
for _ in labels:
    l[_] = []

files = os.listdir(path)
for j in files:
    if '.pickle' not in j:
        continue
    
    print(j)
    with open(j, "rb") as f:
        				x = pickle.load(f)
    f.close()

    for _ in labels:
        for i in x.keys():    
            l[_].append( exploit_sent_emb(int(choice),x[i][_]) )
                       
            
#%% save them into pickle
print(os.getcwd())
with open('max_values_per_top100labels.pickle', 'wb') as handle:
    pickle.dump(l, handle)                
handle.close()