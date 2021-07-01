# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 10:44:38 2020

@authors:
Nikos Mylonas   myloniko@csd.auth.gr
Stamatis Karlos stkarlos@csd.auth.gr
Grigorios Tsoumakas greg@csd.auth.gr
"""

import os
import pickle
import numpy as np
import pandas as pd

import contextlib, time

import bioBERT #local file

@contextlib.contextmanager
def timer():
    """Time the execution of a context block.

    Yields:
        None
    """
    start = time.time()
    # Send control back to the context block
    yield
    end = time.time()
    print('Elapsed: {:.2f}s'.format(end - start))


#%% pre-process
    
path = r'..\Instance-Based-ZSL\pre-computed files'  #... #define the path for pre-computed files  
os.chdir(path)

choice = int(input('How many labels you want? \n1: 100 labels \n2: user defined labels (add .txt file into source path) \n\n Your choice ...  '))

if choice == 1:

    file = open("top_100_labels.txt")
    labels=list()

    for line in file:
        labels.append(line[:-1])

else:
    exit('Needs user input')


test_file = 'pure_zero_shot_test_set_top100.txt'
file = open(test_file)
y = []

for line in file:
    y.append(line[2:-2].split("labels: #")[1])

print('\n#####\nThere are %d instances regarding MeSH 2020 and the selected top-100 novel labels regarding their frequency to the test set. \n#####\n' %len(y))

new_y = []
known_y = []

for label_y in y:
    string = ""
    flag = "false"
    string_known = ""
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

del choice, file, test_file, label_y, string, flag, label, path, string_known, line

# check if any novel label appears into known_y (vectors of seen labels per instance)
#for i in labels: 
#    for j in known_y:
#        if i in j:
#            print(j)
#%% use the bioBERT library, assigning embeddings computations to GPU 

# we load this from the file bioBERT.py which is placed on the same folder
biobert = bioBERT.BiobertEmbedding()

#%%
import random
rand = input('Give random seed: .. ')
random.seed(int(rand))


#%% Decide for: 
#1. saving intermediate pickles per 5k instances 
#2. save the necessary dataframes for all the test set or for specific instances (correction mode - not applicable here)
 
save_choice = input('\n#####\nDo you want intermediate save of pickles?  \n\t Press y / n  ... ')
ZSL = input('\n#####\nDo you want to \n1: Examine the total test set (proper choice for the IBZSL approach) \n2: Correct only existing predictions (still not implemented)  \n\t Press 1 or 2 ... ')
print('\n#####\n')

if ZSL == '2':
    pass
    #end = len(where)
    #scenario = 'combined'
else:
    end = len(new_y)
    where = []
    scenario = 'pureZSL' + '_random_' + rand 


def set_pos(ZSL,n,where):
    if ZSL == '2':
        return where[n]
    else:
        return n

#%% main evaluations

c = 0
counter = 0
k = []
decisions = {}
isolated_predictions = {}
positions = {}
rank_info = {}

batch = -1
start = int(input('Provide the index of the dataset that you want to start. Press 0 for examining all the dataset  .. '))


mode = int(input('\n#####\nWhich mode do you want to apply:  \n1. All known labels are provided \n2. 70% of the known labels are provided \n3. 70% of the known labels are provided and noisy labels are added in the place of the missing ones  \n4. MTI tool''s predictions (existing state-of-the-art approach) \n\n Your choice ...  '))
print('\n#####\n')
      
      
if mode == 3:
    
    arg = 'label_dependence_results_top100labels_' + scenario + '_mode_ranking_shuffled_70percent_plus_noise.pickle'
    
    with open("noisy_labels_70percent_random_seed_" + rand + ".pickle", "rb") as f:
                noisy_dict = pickle.load(f)
    f.close()


    for pos in range(0, len(known_y)):
    
        random.shuffle(known_y[pos])
        
        if len(known_y[pos]) > 3:
            
            rejected = known_y[pos][int(np.ceil(0.7 * len(known_y[pos]))) + 1 : ]
            known_y[pos] = known_y[pos][0: int(np.ceil(0.7 * len(known_y[pos])))] 

            for noise in rejected:
                term = noisy_dict[noise]
                sorted_dict = {k: v for k, v in sorted(term.items(), reverse=True, key=lambda item: item[1])}
                noisy_labels = []

                for _ in sorted_dict:
                    noisy_labels.append(_)
                    break
                known_y[pos] = known_y[pos] + noisy_labels
                
elif mode == 2:
            
    arg = 'label_dependence_results_top100labels_' + scenario + '_mode_ranking_shuffled_70percent.pickle'

    for pos in range(0, len(known_y)):
    
        random.shuffle(known_y[pos])
        if len(known_y[pos]) > 3:
            
            known_y[pos] = known_y[pos][0: int(np.ceil(0.7 * len(known_y[pos])))] 

elif mode == 1:
    
    arg = 'label_dependence_results_top100labels_' + scenario + '_mode_ranking.pickle'

               
elif mode == 4:
    
        arg = 'label_dependence_results_top100labels_' + scenario + '_mode_MTI_ranking.pickle'
        
        z = #(give the path of the pickle like in the example) #'../Instance-Based-ZSL/MTI/mti_predictions.pickle' #... #define the path for pre-computed files  


        with open(z, "rb") as f:
            			y_mti = pickle.load(f)
        f.close()
        
        known_y = y_mti
    
else:
    raise SystemExit('Wrong input')

cmode = 0
for i in known_y:
    cmode += len(i)
print('Number of actual predictions for the total test set: ', cmode)

if mode == 3:
    _ = 'known_y_mode' + str(mode) + '_random_seed_' + rand + '.pickle'
else:
    _ = 'known_y_mode' + str(mode) + '.pickle'

with open(_, 'wb') as handle:
    pickle.dump(known_y, handle)                
handle.close()
#raise SystemExit('Save and close') # for early stop

#%% we have saved into pickle the similarity scores of the top-100 novel labels and all the existing labels
#   per case for accelerting the reproduction of experiments

if mode == 4:
    _ = 'dict_similarities_novel_MTI_labels'
else:
    _ = 'dict_similarities_novel_known_labels'
    
    
with open(_ + ".pickle", "rb") as f:
                dict_top100 = pickle.load(f)
f.close()

print('\n#####\n')
for i in dict_top100.keys():
    print("There are %d different labels into the total predictions."  %len(dict_top100[i]))
    break
print('\n#####\n')
 
#%% the main loop

      
with timer():
    for n in range(start, 10):
        
        if n % 100 == 0:
            print('******', n)
        
        #in case of small tests break early
        #if n == 10:
        #    break
        
        pos = set_pos(ZSL,n,where)
        
        df = pd.DataFrame()
        
        decisions[n] = []
        isolated_predictions[n] = []
        rank_info[n] = []
        positions[n] = pos

        
        if known_y[pos] != ['']:
            
            if ZSL == '2':
               pass
            else:
                search_region = labels

            final_preds = []
            df_copy = []
            
            for label in search_region:

                    counter += 1
                    try:                                   
                        dd = dict_top100[label]
                        #raise KeyError
                    
                    except KeyError:
                        
                        if len(label.split(" ")) == 1 and len(label.split("-")) == 1:
                            label_array = torch.stack(biobert.word_vector(label))[0]
                        elif len(label.split(" ")) > 1 or len(label.split("-")) > 1:
                            label_array = biobert.sentence_vector(label)
                    
                    
                    ranks = []
                    
                    for i in known_y[pos]:
                        if i == '':
                            print('----->', n)
                            continue
                        
                        try:
                            ranks.append(dd[i])
                            #raise KeyError

                        except KeyError:    
                            if len(i.split(" ")) == 1 and len(i.split("-")) == 1:
                                actual_emb = torch.stack(biobert.word_vector(i))[0]
                            elif len(i.split(" ")) > 1 or len(i.split("-")) > 1:
                                actual_emb = biobert.sentence_vector(i)
            
                            if len(label.split(" ")) == 1 and len(label.split("-")) == 1:
                                label_array = torch.stack(biobert.word_vector(label))[0]
                            elif len(label.split(" ")) > 1 or len(label.split("-")) > 1:
                                label_array = biobert.sentence_vector(label)
            
                            dist = torch.cosine_similarity(actual_emb, label_array, dim=0)
                        
                            ranks.append( float(dist.cpu().numpy()) )
                    
                    df[label] = ranks
            
            isolated_predictions[n] = df
            
            # this command is used only for holding the randomly shuffled known_y vector
            decisions[n].append(known_y[pos])

                
                
        else: #this is the case that known_y is empty
            
            print('Empty known_y vector in place: ' , n)
            isolated_predictions[n] = ['None']
            continue
        
            
        ########################################################################################################################
                                        
        rank_info[n].append((df.shape[0], df.shape[1]))  # it contains [number of known labels, number of predicted labels]     
                         
        if save_choice == 'y':
            
            if (n % 5000 == 0) or (n == len(where)-1):
                print('Saving...')
                batch += 1
                with open('checkpoint_batch_' + str(batch) + 'ranking.pickle', 'wb') as handle:
                    pickle.dump([decisions, isolated_predictions, positions, rank_info], handle)                
                handle.close()
#%% recording our output
            
print(os.getcwd())
SystemExit('Stop')
with open(arg, 'wb') as handle:
     pickle.dump([decisions, isolated_predictions, positions, rank_info], handle)                
handle.close()

print('End')