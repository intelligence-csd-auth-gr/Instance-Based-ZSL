# AMULET_SCIS

## Semi-automatic indexing method for BioASQ task through on-the-fly ZSL method.

This repository constitutes an implementation of the submitted papaer wuth the above title by our research team.
It has been created for facilitating the reproducubility of the proposed online Zero-shot Learning algorithm, applied on data mined from the MeSH 2020 database, as it is defined by the BioASQ challenge.

Some brief documentation is provided here for running all the necessary steps, since uploading all the source file demands several GB (a permanent link might be provided). 

link: https://mc03.manuscriptcentral.com/scis


### Source data

- **abstracts from MeSH 2020 with top100 labels appeared.7z** 🠊 contains 5 .txt files (split by 10,000 instances) which include the abstracts from MeSH 2020 that at least one of the top100 most frequent novel labels into this version appears.
- **frequency_novel_labels_test_set_MeSH2020.csv**  🠊 the names of all the novel labels in this version, along with their frequencies of appearence into the test set.

### pre-computed files

Here are added some files for accelerating the execution of several needed computations:

- **label_embeddings.pcikle** 🠊 a dictionary structure whose: *keys* are the names of the top100 most frequent novel labels, *values* their bioBERT embedding vector stored as a Numpy array (768,)

- **predictions_label_occurence.pickle** 🠊 the output of the label occurence stage fot the whole test size into a list structure, where each item is into the next format: 'label_1#label_2#...#label_r'

- **noisy_labels_70percent.pickle** 🠊 a dictionary structure whose: *keys* the known labels that are replaced during the imperfect oracle scenrio, **values** a dictionary with 20 randomly selected MeSH terms in the role of keys, and their corresponding cosine similarity score with the original key of upper level


### Pre-process stages

- **obtain_text_embeddings.py** 🠊 this file computes the embeddings of the input text files and saves them into corresponding pickles with the bioBERT embeddings.


## Process:

 - run NN_baseline_preprocess.py 
 
 Input: path, 'top_100_labels_embeddings.pickle', 44k pickles folder with 5 pickles (~2GB)
 
 - run record_label_similaritiy_scores.py -> produce pickle (1.5 GB) with dataframes for eq1
 
 Input: path, top_100_labels.txt, pure_zero_shot_test_set_top100.txt, noisy_labels_70percent.pickle
