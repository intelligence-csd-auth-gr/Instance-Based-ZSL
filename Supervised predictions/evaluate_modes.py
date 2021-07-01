# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 18:05:01 2021

@authors:
Nikos Mylonas   myloniko@csd.auth.gr
Stamatis Karlos stkarlos@csd.auth.gr
Grigorios Tsoumakas greg@csd.auth.gr
"""

import os, pickle
import numpy as np


# provide the path that contains the supervised predictions from the git repo
path = '..\Instance-Based-ZSL\Supervised predictions'
os.chdir(path)
#%%

choice = input('Give your choice regarding the set of predicted existing labels that you want to evaluate: \nmode1. Ground truth \nmode2. 70% of the ground truth without noisy labels (does not appear in the original work) \nmode3. 70% of the ground truth with 30% random noisy labels \nmode4. MTI-tool predictions (state-of-the-art approach) \nmode5. User defined predictions \n\n ...')


if choice == '1':

	z = 'known_y_mode1.pickle' 

	with open(z, "rb") as f:
				known_y_mode1 = pickle.load(f)
	f.close() 
	y_preds = known_y_mode1

elif choice == '2':
   

	z = 'known_y_mode2.pickle' 

	with open(z, "rb") as f:
				known_y_mode2 = pickle.load(f)
	f.close()
	y_preds = known_y_mode2


 
elif choice == '3':
    
    # the evaluation metrics of the different seed versions are the same, but the noisy labels are different per different seed-based scenario
	choice_seed = input('Provide the seed that was used for creating the noisy predictions. Current choices: 23, 24, or 2021:\n\n... ')
	
	z = 'known_y_mode3_random_seed_' + choice_seed + '.pickle' 

	with open(z, "rb") as f:
				known_y_mode3 = pickle.load(f)
	f.close() 
	y_preds = known_y_mode3


elif choice == '4':
   
	z = 'known_y_mode4.pickle' 

	with open(z, "rb") as f:
				known_y_mode4 = pickle.load(f)
	f.close()
	y_preds = known_y_mode4


elif choice == '5':
   
   # user can provide any other .pickle file that contains predictions of the existing labels per test instance and obtain the corresponding metrics, as in the case of MTI
	#os.chdir(path + '\Instance-Based-ZSL\pre-computed files')
	
    #z = 'user_predictions.pickle' 

	with open(z, "rb") as f:
	    		y_preds = pickle.load(f)
	f.close()

else:
	raise SystemExit('Wrong input')

# depict some stats for the selected predictions
counter_mode = 0
values_mode = []
for i in y_preds:

    counter_mode += len(i)
    values_mode.append(len(i))
    
print('#############')
print('Some useful stats regarding the selected predictions for our examined dataset: \n\nInstances: %d\tAmount of predictions: %d\nAverage number of predictions per instance : %6.3f \nStd of the amount of predictions per instance : %6.3f' %( len(y_preds), counter_mode, np.round(counter_mode / len(y_preds), 3), np.round(np.std(values_mode), 3) ))
print('#############')
        
# load the actual labels
os.chdir('..\Instance-Based-ZSL\pre-computed files')
z = 'known_labels.pickle'

with open(z, "rb") as f:
    		y_actual = pickle.load(f)
f.close() 

#depict some stats for the ground truth
counter_actual = 0
values_actual = []
for i in y_actual:
    
    counter_actual += len(i)
    values_actual.append(len(i))
    
print('Number of ground truth labels for our examined dataset: \n\nInstances: %d\tAmount of labels: %d\nAverage number of labels per instance : %6.3f \nStd of the amount of labels per instance : %6.3f' %( len(y_actual), counter_actual, np.round(counter_actual / len(y_actual), 3), np.round(np.std(values_actual), 3) ))
print('#############')

#%% evaluation

from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()

y_true_binarized = mlb.fit_transform(y_actual)
y_pred_binarized = mlb.transform(y_preds)


# Examine the accuracy of the existing labels that are used for creating the proposed multi-label ranking approach
print(classification_report(y_true_binarized, y_pred_binarized))

# these results regard the MTI scenario (mode4)

#                precision    recall  f1-score  support
#   micro avg       0.63      0.58      0.60    569994
#   macro avg       0.59      0.61      0.56    569994
#weighted avg       0.67      0.58      0.58    569994
# samples avg       0.63      0.59      0.59    569994


#-------> useful for individual stats per label <-------- 
# for specific labels or for all the existing ones

#print(classification_report(y_true_binarized, y_pred_binarized, 
#                           target_names=[str(cls) for cls in mlb.classes_]))