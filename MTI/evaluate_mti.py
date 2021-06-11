# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 18:05:01 2021

@author: stam
"""

import os, pickle
os.chdir(r'C:\Users\stam\Documents\git\Instance-Based-ZSL\Results\imperfect oracle')
os.chdir(r'C:\Users\stam\Documents\git\Instance-Based-ZSL\pre-computed files')


z = 'noisy_labels_70percent.pickle'
z = 'known_labels.pickle'

with open(z, "rb") as f:
    			y = pickle.load(f)
f.close() 

os.chdir(r'C:\Users\stam\Documents\git\Instance-Based-ZSL\MTI')

z = 'mti_predictions.pickle'

with open(z, "rb") as f:
    			y_mti = pickle.load(f)
f.close() 


from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer

y_true = [(12, 2, 4, 5),
          (5, 2),
          (12,)
         ]

y_pred = [(4, 5),
          (5, 2),
          (5, 4)
         ]

mlb = MultiLabelBinarizer()

y_true_binarized = mlb.fit_transform(y_true)
y_pred_binarized = mlb.transform(y_pred)

print(classification_report(y_true_binarized, y_pred_binarized, 
                            target_names=[str(cls) for cls in mlb.classes_]))



y_true_binarized = mlb.fit_transform(y)
y_pred_binarized = mlb.transform(y_mti)


print(classification_report(y_true_binarized, y_pred_binarized))


#   micro avg       0.63      0.58      0.60    569994
#   macro avg       0.59      0.61      0.56    569994
#weighted avg       0.67      0.58      0.58    569994
# samples avg       0.63      0.59      0.59    569994