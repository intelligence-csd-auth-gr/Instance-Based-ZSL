import os, pickle

path = r'C:\Users\stam\Documents\git\Instance-Based-ZSL\Results\imperfect oracle'#... #define the path for pre-computed files  
path = r'D:\datasets\mode2'
os.chdir(path)


z = 'label_dependence_results_top100labels_pureZSL_mode_ranking_shuffled_70percent.pickle'
with open(z, "rb") as f:
    			decisions, isolated_predictions, positions, rank_info = pickle.load(f)
f.close() 

#%%
path = r'C:\Users\stam\Documents\git\Instance-Based-ZSL\pre-computed files'#... #define the path for pre-computed files  
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
print(len(y))

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
pd.DataFrame(all_k).to_csv('70_percent_of_known_labels.csv')    

x = pd.read_csv('known_y_labels.csv')
x = x.iloc[1:,1] #remove index and ''
x = x.to_list()

diff = list(set(x) - set(all_k))
#%%

from biobert_embedding import downloader
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM    
import torch
import logging
import numpy as np

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

biobert = BiobertEmbedding()
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
with open('noisy_labels_70percent_new.pickle', 'wb') as handle:
     pickle.dump(d, handle)                
handle.close()
    
print('**Adding noisy labels instead of the rejected ones***')