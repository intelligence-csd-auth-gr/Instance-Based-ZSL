# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 18:05:01 2021

@authors:
Nikos Mylonas   myloniko@csd.auth.gr
Stamatis Karlos stkarlos@csd.auth.gr
Grigorios Tsoumakas greg@csd.auth.gr
"""

import os, pickle
path = r'C:\Users\stam\Documents\git\Instance-Based-ZSL\MTI'
os.chdir(path)

choice = input('Give your choice regarding the set of predicted existing labels that you want to evaluate: \nmode1. Ground truth \nmode2. 70% of the ground truth without noisy labels \nmode3. 70% of the ground truth with 30% random noisy labels \nmode4. MTI-tool predictions (state-of-the-art approach) \nmode5. User defined predictions \n\n ...')


if choice == '1':

	z = 'known_y_mode1.pickle' 

	with open(z, "rb") as f:
				y_preds_mode1 = pickle.load(f)
	f.close() 
	y_preds = y_preds_mode1

elif choice == '2':
   

	z = 'known_y_mode2.pickle' 

	with open(z, "rb") as f:
				y_preds_mode2 = pickle.load(f)
	f.close()
	y_preds = y_preds_mode2

elif choice == '3':

	z = 'known_y_mode3.pickle' 

	with open(z, "rb") as f:
				y_preds_mode3 = pickle.load(f)
	f.close() 
	y_preds = y_preds_mode3


elif choice == '4':
   
	z = 'known_y_mode4.pickle' 

	with open(z, "rb") as f:
				y_preds_mode4 = pickle.load(f)
	f.close()
	y_preds = y_preds_mode4


elif choice == '5':
   
   # user can provide any other .pickle file that contains predictions of the existing labels per test instance and obtain the corresponding metrics, as in the case of MTI
	#os.chdir(path + '\Instance-Based-ZSL\pre-computed files')
	
    #z = 'user_predictions.pickle' 

	with open(z, "rb") as f:
	    		y_preds = pickle.load(f)
	f.close()

else:
	raise SystemExit('Wrong input')


cmode = 0

for i in y_preds_mode4:
    for j in i:
        cmode += len(j)
print('The current mode contains %d predictions for all the test set' %cmode)
        
# load the actual labels
os.chdir(r'C:\Users\stam\Documents\git\Instance-Based-ZSL\pre-computed files')
z = 'known_labels.pickle'

with open(z, "rb") as f:
    		y_actual = pickle.load(f)
f.close() 

#%% evaluation

from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()

y_true_binarized = mlb.fit_transform(y_actual)
y_pred_binarized = mlb.transform(y_preds)


# Examine the accuracy of the existing labels that are used for creating the proposed multi-label ranking approach
print(classification_report(y_true_binarized, y_pred_binarized))

#   micro avg       0.63      0.58      0.60    569994
#   macro avg       0.59      0.61      0.56    569994
#weighted avg       0.67      0.58      0.58    569994
# samples avg       0.63      0.59      0.59    569994


# for specific labels or for all the existing ones
#print(classification_report(y_true_binarized, y_pred_binarized, 
#                           target_names=[str(cls) for cls in mlb.classes_]))