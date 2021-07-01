# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:50:46 2020

@authors:
Nikos Mylonas   myloniko@csd.auth.gr
Stamatis Karlos stkarlos@csd.auth.gr
Grigorios Tsoumakas greg@csd.auth.gr
"""

import os, pickle

path = '..' # define the path with the files that are produced by the record_label_similarity_scores.py
os.chdir(path)

# we select one of the 3 seed-based variants that are exported by mode2, so as to inject noisy labels to them and create the input for the mode3

#z = 'label_dependence_results_top100labels_pureZSL_random_seed_23_mode_ranking_shuffled_70percent.pickle'
#z = 'label_dependence_results_top100labels_pureZSL_random_seed_24_mode_ranking_shuffled_70percent.pickle'
#z = 'label_dependence_results_top100labels_pureZSL_random_seed_2021_mode_ranking_shuffled_70percent.pickle'

choice_seed = input('Provide the seed that was used for creating the noisy predictions. Current choices: 23, 24, or 2021:\n\n... ')
z = 'label_dependence_results_top100labels_pureZSL_random_seed_' + choice_seed + '_mode_ranking_shuffled_70percent.pickle'

with open(z, "rb") as f:
    	decisions, isolated_predictions, positions, rank_info = pickle.load(f)
f.close() 

#%%

path = r'..\Instance-Based-ZSL\pre-computed files'#... #define the path for pre-computed files  
os.chdir(path)

file = open("top_100_labels.txt")
labels=list()

for line in file:
   labels.append(line[:-1])


test_file = "pure_zero_shot_test_set_top100.txt"
y = []

file = open(test_file)
for line in file:
    y.append(line[2:-2].split("labels: #")[1])

print('Number of test instances: ', len(y))

new_y = []
known_y = []

for label_y in y:
    string = ""
    flag = "false"
    string_known=""
    for label in label_y.split("#"):
        if label in labels:
            flag = "true"
            string = string + label + "#"
        else:
            string_known=string_known+label+"#"
    if (flag == "false"):
        string = "None#"
    new_y.append(string[:-1].split('#'))
    known_y.append(string_known[:-1].split('#'))
    
                   
k = []    

for i in range(0, len(known_y)):
    if len(known_y[i]) > 3:
        k.append(list(set(known_y[i]) - set(decisions[i][0])))
    else:
        k.append(known_y[i])

all_k = []
c = 0    

for _ in k:
    c += 1
    for j in _:
        all_k.append(j)
    if c % 100 == 0:
        all_k = list(set(all_k))
        
all_k.remove('') 

del c,f,file,flag, isolated_predictions,j,k,label, positions, rank_info, string, string_known, test_file, y, z

#%% manipulate properly the known labels and the randomly selected subset of it 

import pandas as pd
pd.DataFrame(all_k).to_csv('70_percent_of_known_labels_random_seed_' + choice_seed + '.csv')    

x = pd.read_csv('known_y_labels.csv')
x = x.iloc[1:,1] #remove index and ''
x = x.to_list()

diff = list(set(x) - set(all_k))
#%%

import bioBERT #local file
biobert = bioBERT.BiobertEmbedding()
#%% replace the missing ground truth with some randomly selected labels from a larger pool 
#   thus we sample here 20 labels randomly per label to be replaced
#   and compute the Cosine similarity of the latter with the former ones, 
#   This dictionary is latter exploited from the record_label_similarity_scores for mode3

import random

counter = 0
d = {}

for label in all_k:
    
    d[label] = {}
    random.seed(counter)
    sampled_list = random.sample(diff, 20)
    counter += 1
    

    if len(label.split(" ")) == 1 and len(label.split("-")) == 1:
        label_array = torch.stack(biobert.word_vector(label))[0]
    elif len(label.split(" ")) > 1 or len(label.split("-")) > 1:
        label_array = biobert.sentence_vector(label)
    
    #ranks = []
    for i in sampled_list:

        if len(i.split(" ")) == 1 and len(i.split("-")) == 1:
            actual_emb = torch.stack(biobert.word_vector(i))[0]
        elif len(i.split(" ")) > 1 or len(i.split("-")) > 1:
            actual_emb = biobert.sentence_vector(i)

        dist = torch.cosine_similarity(actual_emb, label_array, dim=0)
        d[label][i] = float(dist.cpu().numpy())
        
    if counter % 100 == 0:
        print(counter)

print(os.getcwd())
with open('noisy_labels_70percent_new_random_seed_' + choice_seed + '.pickle', 'wb') as handle:
     pickle.dump(d, handle)                
handle.close()
    
print('**Adding noisy labels instead of the rejected ones***')