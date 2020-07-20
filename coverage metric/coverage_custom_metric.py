# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 15:37:56 2020

@author: 

Implementation of coverage metric for categorical labels
"""

import numpy as np

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
            
            kk.append( list(y_pred[y]).index(y_test[y][i]) + 1)
        
        k.append(max(kk))
        
    return k

#Example 1
print(coverage_custom([ ['ball'] , ['music', 'food'] ] , [ ['ball'] , ['music', 'food'] ]) )
# [1,2]
k = coverage_custom([ ['ball'] , ['music', 'food'] ] , [ ['ball'] , ['music', 'food'] ])
print(np.round(np.mean(k),3))
# 1.5


#Example 2
print(coverage_custom([ ['music'] , ['food', 'music'] ] , [ ['ball', 'music'] , ['music', 'walk', 'run', 'food'] ]) )
# [2,4]
k = coverage_custom([ ['music'] , ['food', 'music'] ] , [ ['ball', 'music'] , ['music', 'walk', 'run', 'food'] ])
print(np.round(np.mean(k),3))
# 3.0


# Perfect match
print(coverage_custom([ ['music'] , ['food', 'music'] ] , [ ['music', '1'] , ['music', 'food'] ]) )
# [1,2]
k = coverage_custom([ ['music'] , ['food', 'music'] ] , [ ['music', '1'] , ['music', 'food'] ])
print(np.round(np.mean(k),3))
# 1.5

# Worst case scenario
print(coverage_custom([ ['music'] , ['food', 'music'] ] , [ ['1','2','3','music'] , ['music', '1','2', 'food'] ]) )
# [4,4]
k = coverage_custom([ ['music'] , ['food', 'music'] ] , [ ['1','2','3','music'] , ['music', '1','2', 'food'] ])
print(np.round(np.mean(k),3))
# 4