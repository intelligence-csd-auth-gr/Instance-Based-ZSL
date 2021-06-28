# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 11:04:22 2021

@authors:
Nikos Mylonas   myloniko@csd.auth.gr
Stamatis Karlos stkarlos@csd.auth.gr
Grigorios Tsoumakas greg@csd.auth.gr
"""
# we extract the appropriate information from the MTI predictions as they are exported by the official tool found in: https://ii.nlm.nih.gov/MTI/
    
import os
import numpy as np

def mti_to_list(line):
	_ = line.split('|')
	key, value = _[0], _[1]
	d = {}
	d[key] = value
	return  d

# provide the path that contains the mti_output.out
#os.chdir('..Instance-Based-ZSL\MTI')
os.chdir(r'C:\Users\stam\Documents\git\Instance-Based-ZSL\MTI')
filepath = 'mti_output.out'

####################################################################################
y_mti = {}

with open(filepath) as fp:
	line = fp.readline()
	while line:

		#print("Line {}: {}".format(cnt, line.strip()))
        
		d = mti_to_list(line)
		key = list(d.keys())[0]
        
		if key not in y_mti.keys():
			y_mti[key] = ''
		else:
			y_mti[key] += '#' + d[key]
		
		line = fp.readline()


# store the predictions into a dictionary structure	
y_mti_proper = {}
for i in y_mti.keys():
    y_mti_proper[int(i)] = y_mti[i].split('#')[1:]

values = []
counter = 0

# store the predictions into a list, which is compatible with our implementation
mti_proper_list = []
for i in y_mti_proper.keys():
    mti_proper_list.append(y_mti_proper[i])
    counter += len(y_mti_proper[i])
    values.append(len(y_mti_proper[i]))
    
    
print('Number of predictions for our examined dataset: \n\nInstances: %d\tAmount of predictions: %d\nAverage number of predictions per instance (MTI): %6.3f \nStd of the amount of predictions per instance (MTI): %6.3f' %( len(mti_proper_list), counter, np.round(counter / len(mti_proper_list), 3), np.round(np.std(values), 3) ))
    
#%% remove the novel labels from the predictions whever they appear

#path = '/home/stkarlos/Instance-Based-ZSL/pre-computed files'#... #define the path for pre-computed files  
os.chdir(r'C:\Users\stam\Documents\git\Instance-Based-ZSL\pre-computed files')
file = open("top_100_labels.txt")
labels=list()
for line in file:
    labels.append(line[:-1])
    
mti_clear_list = []
for i in mti_proper_list:
    for novel in labels:
        if novel in i:
            i.remove(novel)
    mti_clear_list.append(i)
    
values = []
counter = 0

for i in mti_clear_list:
    counter += len(i)
    values.append(len(i))
    
    
print('Number of predictions for our examined dataset: \n\nInstances: %d\tAmount of predictions: %d\nAverage number of predictions per instance (MTI): %6.3f \nStd of the amount of predictions per instance (MTI): %6.3f' %( len(mti_clear_list), counter, np.round(counter / len(mti_clear_list), 3), np.round(np.std(values), 3) ))
    
#%%
import pickle
os.chdir(r'C:\Users\stam\Documents\git\Instance-Based-ZSL\MTI')

# for storing the predictions into a .pickle file
with open('mti_predictions.pickle', 'wb') as handle:
    pickle.dump(mti_proper_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

# for loading a .pickle file
#with open('mti_predictions.pickle', 'rb') as handle:
#    mti_proper_list = pickle.load(handle)