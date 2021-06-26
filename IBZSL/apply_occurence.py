# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 10:22:48 2020

@authors:
Nikos Mylonas   myloniko@csd.auth.gr
Stamatis Karlos stkarlos@csd.auth.gr
Grigorios Tsoumakas greg@csd.auth.gr
"""

import pandas as pd
import os, pickle


def positions_with_empty_known_labels(current_path):
    
    with open('known_labels.pickle', 'rb') as handle:
        known_y = pickle.load(handle)                
    handle.close()
    
    empty_known_labls = []
    for i in range(0, len(known_y)):
        if known_y[i] == ['']:
            empty_known_labls.append(i)

    
    os.chdir(current_path)
    return empty_known_labls

##############################################################################
path = r'C:\Users\stam\Documents\git\Instance-Based-ZSL\pre-computed files'
os.chdir(path)


z = 'predictions_label_occurence'
with open(z + ".pickle", "rb") as f:
    			y	 = pickle.load(f)
f.close()

y_occ = []

for i in y:
	y_occ.append(i.split('#'))

# we detect the positions where the label occurence returns 'None' as a decision
where = []
c = -1

for i in y_occ:
    
    c +=1
    if i == ['None']:
        where.append(c)
        
# we compute the positions where the known labels are empty
empty_pos = positions_with_empty_known_labels(path)

##############################################################################
# here we load the vector with predictions
#os.chdir(r'D:\BioASQ\evaluate_py')

###### these are the files that are evaluated into the original work #########

# LSSc(max)
which = 'label_dependence_max_NN_bioBERT_44k_decisions_scores_official.pickle'
which = 'label_dependence_max_NN_bioBERT_44k_decisions_scores_shuffled_70percent_plus_noise_official.pickle'

# w-LSSc(max) --> combined with the label_occurence it leads to ONZSL(max) ** proposed one
which = 'label_dependence_max_weighted_NN_bioBERT_44k_decisions_scores_official.pickle'
which = 'label_dependence_max_weighted_NN_bioBERT_44k_decisions_scores_shuffled_70percent_plus_noise_official.pickle'

# w-LSSc(sum) --> combined with the label_occurence it leads to ONZSL(sum)
which = 'label_dependence_sum_weighted_NN_bioBERT_44k_decisions_scores_official.pickle'
which = 'label_dependence_sum_weighted_NN_bioBERT_44k_decisions_scores_shuffled_70percent_plus_noise_official.pickle'

os.chdir(r'D:\datasets\mode4')
which = 'label_dependence_max_NN_bioBERT_44k_decisions_scores_MTI_official.pickle'
which = 'label_dependence_max_weighted_NN_bioBERT_44k_decisions_scores_MTI_official.pickle'

os.chdir(r'D:\datasets\mode4')
which = 'label_dependence_max_sum_NN_bioBERT_44k_decisions_scores_MTI_official.pickle'
which = 'label_dependence_max_sum_weighted_NN_bioBERT_44k_decisions_scores_MTI_official.pickle'

os.chdir(r'D:\datasets\mode2')
which = 'label_dependence_max_NN_bioBERT_44k_decisions_scores_mode_ranking_shuffled_70percent.pickle'
which = 'label_dependence_max_weighted_NN_bioBERT_44k_decisions_scores_mode_ranking_shuffled_70percent.pickle'

os.chdir(r'D:\datasets\mode1')
which = 'label_dependence_max_NN_bioBERT_44k_decisions_scores_mode_rankingt.pickle'
which = 'label_dependence_max_weighted_NN_bioBERT_44k_decisions_scores_mode_ranking.pickle'

os.chdir(r'D:\datasets\mode3')
which = 'label_dependence_max_NN_bioBERT_44k_decisions_scores_mode_ranking_shuffled_70percent_plus_noise.pickle_total.pickle'
which = 'label_dependence_max_weighted_NN_bioBERT_44k_decisions_scores_mode_ranking_shuffled_70percent_plus_noise.pickle_total.pickle'
###############################################################################
with open(which, "rb") as f:
    	label_dependence = pickle.load(f)
f.close()


yy = label_dependence
all_pos = []
for i in range(0, len(y_occ)):
    
    if i in where or i in empty_pos:
        continue    
    else:
        pos, labels = [], []
        for j in y_occ[i]:
            yy[i] = list(yy[i])
            pos.append(yy[i].index(j))
            labels.append(j)
        
        how = sorted(range(len(pos)), key=lambda k: pos[k], reverse = True)
    
        for _ in how:     
            yy[i].remove(labels[_])
            yy[i].insert(0, labels[_])
    
          
    for i in pos:
        all_pos.append(i)

##############################################################################
makeplot = True
#provides the ranking of the labels that were found by occurence
if makeplot:
    pd.DataFrame(all_pos).hist(bins = 50) 


# save the vector with the ranked decisions having also applied the label occurence stage
with open('occurence_modified_' + which,  'wb') as handle:
     pickle.dump(yy, handle)                
handle.close()