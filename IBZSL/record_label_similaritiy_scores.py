# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 10:44:38 2020

@authors:
Nikos Mylonas   myloniko@csd.auth.gr
Stamatis Karlos stkarlos@csd.auth.gr
Grigorios Tsoumakas greg@csd.auth.gr
"""

import pickle
import numpy as np
import pandas as pd
import os



path = r'C:\Users\stam\Documents\git\Instance-Based-ZSL\pre-computed files'#... #define the path for pre-computed files  
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


del choice, file, test_file, label_y, string, flag, label, path, string_known, line

#%% use the bioBERT library, assigning embeddings computations to GPU 

import tensorflow as tf
from biobert_embedding import downloader
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM    
import torch
import logging

logging.basicConfig(filename='app.log', filemode='w',format='%(asctime)s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


if torch.cuda.is_available():    

	# Tell PyTorch to use the GPU.    
	device = torch.device("cuda")

	print('There are %d GPU(s) available.' % torch.cuda.device_count())

	print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    
    device = 'cpu'

class BiobertEmbedding(object):
	"""
	Encoding from BioBERT model (BERT finetuned on PubMed articles).
	Parameters
	----------
	model : str, default Biobert.
			pre-trained BERT model
	"""

	def __init__(self, model_path=None):

		if model_path is not None:
			self.model_path = model_path
		else:
			self.model_path = downloader.get_BioBert("google drive")


		self.tokens = ""
		self.sentence_tokens = ""
		self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
		# Load pre-trained model (weights)
		self.model = BertModel.from_pretrained(self.model_path)
		self.model.to(device)
		logger.info("Initialization Done !!")

	def process_text(self, text):

		marked_text = "[CLS] " + text + " [SEP]"
		# Tokenize our sentence with the BERT tokenizer.
		tokenized_text = self.tokenizer.tokenize(marked_text)
		return tokenized_text


	def handle_oov(self, tokenized_text, word_embeddings):
		embeddings = []
		tokens = []
		oov_len = 1
		for token,word_embedding in zip(tokenized_text, word_embeddings):
			if token.startswith('##'):
				token = token[2:]
				tokens[-1] += token
				oov_len += 1
				embeddings[-1] += word_embedding
			else:
				if oov_len > 1:
					embeddings[-1] /= oov_len
				tokens.append(token)
				embeddings.append(word_embedding)
		return tokens,embeddings


	def eval_fwdprop_biobert(self, tokenized_text):

		# Mark each of the tokens as belonging to sentence "1".
		segments_ids = [1] * len(tokenized_text)
		# Map the token strings to their vocabulary indeces.
		indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

		# Convert inputs to PyTorch tensors
		tokens_tensor = torch.tensor([indexed_tokens]).to(device)
		segments_tensors = torch.tensor([segments_ids]).to(device)

		# Put the model in "evaluation" mode, meaning feed-forward operation.
		self.model.eval()
		# Predict hidden states features for each layer
		with torch.no_grad():
			encoded_layers, _ = self.model(tokens_tensor, segments_tensors)

		return encoded_layers


	def word_vector(self, text, handle_oov=True, filter_extra_tokens=True):

		tokenized_text = self.process_text(text)

		encoded_layers = self.eval_fwdprop_biobert(tokenized_text)

		# Concatenate the tensors for all layers. We use `stack` here to
		# create a new dimension in the tensor.
		token_embeddings = torch.stack(encoded_layers, dim=0)
		token_embeddings = torch.squeeze(token_embeddings, dim=1)
		# Swap dimensions 0 and 1.
		token_embeddings = token_embeddings.permute(1,0,2)

		# Stores the token vectors, with shape [22 x 768]
		word_embeddings = []
		logger.info("Summing last 4 layers for each token")
		# For each token in the sentence...
		for token in token_embeddings:

			# `token` is a [12 x 768] tensor
			# Sum the vectors from the last four layers.
			sum_vec = torch.sum(token[-4:], dim=0)

			# Use `sum_vec` to represent `token`.
			word_embeddings.append(sum_vec)

		self.tokens = tokenized_text
		if filter_extra_tokens:
			# filter_spec_tokens: filter [CLS], [SEP] tokens.
			word_embeddings = word_embeddings[1:-1]
			self.tokens = tokenized_text[1:-1]

		if handle_oov:
			self.tokens, word_embeddings = self.handle_oov(self.tokens,word_embeddings)
		logger.info(self.tokens)
		logger.info("Shape of Word Embeddings = %s",str(len(word_embeddings)))
		return word_embeddings



	def sentence_vector(self,text):

		logger.info("Taking last layer embedding of each word.")
		logger.info("Mean of all words for sentence embedding.")
		tokenized_text = self.process_text(text)
		self.sentence_tokens = tokenized_text
		encoded_layers = self.eval_fwdprop_biobert(tokenized_text)

		# `encoded_layers` has shape [12 x 1 x 22 x 768]
		# `token_vecs` is a tensor with shape [22 x 768]
		token_vecs = encoded_layers[11][0]

		# Calculate the average of all 22 token vectors.
		sentence_embedding = torch.mean(token_vecs, dim=0)
		logger.info("Shape of Sentence Embeddings = %s",str(len(sentence_embedding)))
		return sentence_embedding

## call the above class
biobert = BiobertEmbedding()

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
    scenario = 'pureZSL'


def set_pos(ZSL,n,where):
    if ZSL == '2':
        return where[n]
    else:
        return n

#%% main evaluations

import random
random.seed(24)

c = 0
counter = 0
k = []
decisions = {}
isolated_predictions = {}
positions = {}
rank_info = {}

batch = -1
start = 0


mode = int(input('\n#####\nWhich mode do you want to apply:  \n1. All known labels are provided \n2. 70% of the known labels are provided \n3. 70% of the known labels are provided and noisy labels are added in the place of the missing ones  \n4. MTI tool''s predictions (existing state-of-the-art approach) \n\n Your choice ...  '))
print('\n#####\n')
      
      
if mode == 3:
    
    arg = 'label_dependence_results_top100labels_' + scenario + '_mode_' + 'ranking_shuffled_70percent_plus_noise.pickle'
    
    with open("noisy_labels_70percent.pickle", "rb") as f:
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
            
    arg = 'label_dependence_results_top100labels_' + scenario + '_mode_' + 'ranking_shuffled_70percent.pickle'

    for pos in range(0, len(known_y)):
    
        random.shuffle(known_y[pos])
        if len(known_y[pos]) > 3:
            
            known_y[pos] = known_y[pos][0: int(np.ceil(0.7 * len(known_y[pos])))] 

elif mode == 1:
    
    arg = 'label_dependence_results_top100labels_' + scenario + '_mode_' + 'ranking.pickle'

               
elif mode == 4:
    
        arg = 'label_dependence_results_top100labels_' + scenario + '_mode_' + 'MTI_ranking.pickle'
        
        z = r'C:\Users\stam\Documents\git\Instance-Based-ZSL\MTI\mti_predictions.pickle'

        with open(z, "rb") as f:
            			y_mti = pickle.load(f)
        f.close()
        
        known_y = y_mti
    
else:
    raise SystemExit('Wrong input')


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

for n in range(start, end):
    
    if n % 100 == 0:
        print('******', n, 'with ', c, 'yes')
    
    #for small tests break early
    #if n == 100:
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
        
        # this command is ised only for holding the random shuffled and kept known_y vector
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
#%%
            
print(os.getcwd())

with open(arg, 'wb') as handle:
     pickle.dump([decisions, isolated_predictions, positions, rank_info], handle)                
handle.close()

print('End')