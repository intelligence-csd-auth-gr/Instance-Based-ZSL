# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 11:29:52 2021

@author: stam
"""

import pickle
import numpy as np
import pandas as pd
import os
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

#%%


def label_embeddings(asked_labels):
	
	d = {}
	c1, c2 = 0, 0

	for label in asked_labels:
		if label == '':
			continue
		else:
			if len(label.split(" ")) == 1 and len(label.split("-")) == 1:
				label_array = torch.stack(biobert.word_vector(label))[0]
				c1 += 1
			elif len(label.split(" ")) > 1 or len(label.split("-")) > 1:
				label_array = biobert.sentence_vector(label)
				c2 += 1
			d[label] = label_array
	print(c1,c2)
            

def label_pair(novel_labels_embeddings, existing_labels_embeddings):
    
    d = {}
    for i in list(novel_labels_embeddings.keys()):
        
        d[i] = {}
        print(i)
        for j in list(existing_labels_embeddings.keys()):
            
            d[i][j] = float( torch.cosine_similarity(novel_labels_embeddings[i], existing_labels_embeddings[j], dim=0) )
            
    
    return d

                  





path = r'C:\Users\stam\Documents\git\Instance-Based-ZSL\pre-computed files'#... #define the path for pre-computed files  
os.chdir(path)

choice = int(input('How many labels you want? \n1: 100 labels \n2: user defined labels (add .txt file into source path) \n\n Your choice ...  '))
if choice == 1:

    file = open("top_100_labels.txt")
    novel_labels=list()
    for line in file:
        novel_labels.append(line[:-1])

else:
    exit('Needs user input')
    

choice = int(input('Compute similarities of each novel label with: \n1: actual labels \n2: MTI predictions \n\n Your choice ...  '))

if choice == 1:
    
        _ = r'C:\Users\stam\Documents\git\Instance-Based-ZSL\pre-computed files\known_labels.pickle'
        with open(_, "rb") as f:
            			supervised_predictions = pickle.load(f)
        f.close()
        to_save = 'known_labels'

elif choice == 2:
        
        _ = r'C:\Users\stam\Documents\git\Instance-Based-ZSL\MTI\mti_predictions.pickle'
        with open(_, "rb") as f:
            			supervised_predictions = pickle.load(f)
        f.close()
        to_save = 'MTI_labels'

else:
    exit('Wrong input')
    

# find the separate labels into the already given predictions
    
counter = 0
l = []

for i in supervised_predictions:
    counter += len(i)
    for _ in i:
        l.append(_)
        
existing_labels_set = list(set(l))
print('There are %d separate labels which appear %d times in total into the available %d test instances' %(len(existing_labels_set), counter, len(supervised_predictions)) )
    
novel_labels_embeddings = label_embeddings(novel_labels)
existing_labels_embeddings = label_embeddings(existing_labels_set)
d = label_pair(novel_labels_embeddings, existing_labels_embeddings)

with open('novel_labels_embeddings.pickle', 'wb') as handle:
    pickle.dump(novel_labels_embeddings, handle)                
handle.close()


with open(to_save + '_embeddings.pickle', 'wb') as handle:
    pickle.dump(existing_labels_embeddings, handle)                
handle.close()

with open('dict_similarities_novel_' + to_save + '.pickle', 'wb') as handle:
    pickle.dump(d, handle)                
handle.close()
#actual_emb = novel_labels_embeddings['Non-Smokers']
#label_array = novel_labels_embeddings['Nutrients']
#dist = torch.cosine_similarity(actual_emb, label_array, dim=0)
