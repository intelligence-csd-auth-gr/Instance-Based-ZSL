import os
import pickle
import time
import numpy as np
from scipy.spatial import distance

##############################################################################
def find_similarities(label_embeddings, test_embeddings, this_count, search_region, choice, output_name):
  
    similarities = {}
    count = this_count
    
    for sentences in test_embeddings:
        
        similarities['instance:' + str(count)] = {}
        
        if choice == 1:
            where = label_embeddings
        else:
            where = search_region[count]
            
        for label in where:
          
            if label == '':
              continue
          
            similarities['instance:' + str(count)][label] = []
          
            for sentence in sentences:
            
                if (len(sentence)!=0) :
                    dist1 = 1 - distance.cosine(label_embeddings[label], sentence.cpu())
                    similarities['instance:' + str(count)][label].append(dist1)
                else:
                    print(count)
        
        count+=1
        if count % 1000 == 0:
            print(count)


    i = int(this_count/10000)
    with open('similarities_' + output_name + '_' + str(i) + '.pickle', 'wb') as f:
        pickle.dump(similarities, f)
    f.close()
    
    return similarities

##############################################################################
# load from the pre-computed folder

path = r'D:\BioASQ\evaluate_py'
os.chdir(path)
choice = int(input('Press your choice: Compute similarities between each test abstract and \n1. all the top100 labels? \n2. each known label? \n... '))

if choice == 1:
    name = 'label_embeddings_top100'
    output_name = 'top_100'
    
elif choice == 2:
    name = 'known_labels_embeddings'
    output_name = 'known_labels'
    
    with open("known_labels.pickle", "rb") as f:
        known_y = pickle.load(f)
    f.close()
    

with open(name + ".pickle", "rb") as f:
            label_embeddings = pickle.load(f)
f.close()
##############################################################################

# define the corresponnding path which contains the pickles with the embeddings
# for the whole test set into batches

path = # define the output folder path   e.g. r'D:\BioASQ\evaluate_py'
os.chdir(path)

for i in range(0,5):
  
    name = #define the full path of the pickle files (look the example) that are computed from the obtain_text_embeddings.py script    e.g. r'D:\44k pickles\pure_Zeroshot_test_set_'
    with open(name + str(i) + ".pickle", "rb") as f:
        sentence_embeddings = pickle.load(f)
    f.close()


    number = i*10000
    start=time.time()
    
    if choice == 1:
        search_region = []
    else:
        search_region = known_y
        
    sim = find_similarities(label_embeddings, sentence_embeddings, number, search_region, choice, output_name)
    
    end=time.time()

    print('Batch ', i, ' was completed after ', np.round(end-start), ' seconds')

print('**Similarities have been computed***')