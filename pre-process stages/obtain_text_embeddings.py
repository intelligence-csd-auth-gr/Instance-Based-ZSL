import os
import pandas as pd
import numpy as np
import torch
import spacy
import pickle
import time
import logging
from biobert_embedding.embedding import BiobertEmbedding
from nltk.corpus import stopwords
import nltk

import warnings
warnings.filterwarnings("ignore")

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
spacy.load('en_core_web_sm')
nlp = spacy.load("en_core_web_sm")

##############################################################################
logging.basicConfig(filename='app.log', filemode='w',format='%(asctime)s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# If there's a GPU available...
if torch.cuda.is_available():    

	# Tell PyTorch to use the GPU.    
	device = torch.device("cuda")

	print('There are %d GPU(s) available.' % torch.cuda.device_count())

	print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
	print('No GPU available, using the CPU instead.')
	device = torch.device("cpu")

print('====', device.type)
##############################################################################

# define the aporopriate defs for working on dataframes
def break_se(tokens):
	return [i.string.strip() for i in tokens if len(i.string.strip()) >= 15]


def catch(func, handle=lambda e : e):
	try:
		return biobert.sentence_vector(func).to(device)
	except Exception as e:
		return torch.empty(0)


def get_bert_embeddings(wr):    
	return [catch(i) for i in wr]


def get_bert_embedding(wr):    
	return [biobert.sentence_vector(i).to(device) for i in wr]


def saving(data, i):
    
    # we just define a more proper name for out exported files
	with open('pure_Zeroshot_test_set_' + str(i) + '.pickle', 'wb') as f:
		pickle.dump(data.embeddings, f)
	f.close()
	return


def create_embs(filename, iid):
	
	data = pd.read_csv(filename, delimiter="\n",header=None)
	print(data.shape)

	print('Batch starts: ', str(iid))
	start = time.time()
		
	data.rename(columns={0: 'text'}, inplace=True)
	print(data.columns)

	data.text = data.text.apply(lambda x: x[2:-2])
	data['X'] = data.text.apply(lambda x: x.split(" labels: #")[0])
	data['Y'] = data.text.apply(lambda x: x.split(" labels: #")[1])
	data['tokens'] = data.X.apply(lambda x: nlp(x).sents)
	data['large_tokens'] = data.tokens.apply(break_se)
	data['embeddings'] = data.large_tokens.apply(lambda x: get_bert_embeddings(x)) 

	saving(data, iid)
	end = time.time()

	print('Batch :', str(iid), ' lasts ', np.round(end - start), 'seconds')

	return
##############################################################################
start_total = time.time()

#define the corresponnding path


os.chdir(r'D:\BioASQ\allMeSH_2020\for embeddings known of 44k')
os.getcwd()

biobert = BiobertEmbedding()
	
for f, file in enumerate(os.listdir(os.getcwd())):
		print(f, file)
		create_embs(file,f)


end_total = time.time()
print('Total Batches : lasts ', np.round(end_total - start_total), 'seconds')

print('**Embeddings of test set have been computed into batches***')