# AMULET_SCIS

## Semi-automatic indexing method for BioASQ task through on-the-fly ZSL method.

This repository constitutes an implementation of the submitted paper to the Science China Information Science ([SCIS](https://www.springer.com/journal/11432)) journal with the above title by our research team.
It has been created for facilitating the reproducubility of the proposed online Zero-shot Learning algorithm, applied on data mined from the MeSH 2020 database, as it is defined by the coresponding [BioASQ challenge](http://bioasq.org) regarding the Biomedical indexing.

Some brief documentation is provided here for running all the necessary steps, since uploading all the source file demands several of GBs (a permanent link might be provided). 

link: https://mc03.manuscriptcentral.com/scis


### source data 

- **abstracts from MeSH 2020 with top100 labels appeared.7z** 🠊 contains 5 .txt files (split per 10,000 instances) which include the abstracts from MeSH 2020 that at least one of the top100 most frequent novel labels appears into its label space. (*input text files*)
- **frequency_novel_labels_test_set_MeSH2020.csv**  🠊 the names of all the novel labels in this version, along with the frequency of appearence per each one into the examined test set.

### pre-computed files

Here are added some files for accelerating the execution of several needed computations:


- **label_embeddings_top100.pcikle** 🠊 a dictionary structure whose: *keys* are the names of the top100 most frequent novel labels, *values* their bioBERT embedding vector stored as a Numpy array (768,)

- **known_labels_embeddings.pickle'** 🠊 a dictionary structure whose: *keys* are the names of all the known labels into the examined test set, *values* their bioBERT embedding vector stored as a Numpy array (768,)

- **known_labels.pickle** 🠊 the known labels of the whole test set, being stored as a list with list items per instance, which contain the separate labels inside them 

- **predictions_label_occurence.pickle** 🠊 the output of the label occurence stage fot the whole test size into a list structure, where each item is into the next format: 'label_1#label_2#...#label_r'

- **noisy_labels_70percent.pickle** 🠊 a dictionary structure whose: *keys* the known labels that are replaced during the imperfect oracle scenario, *values* a dictionary with 20 randomly selected MeSH terms in the role of keys, and their corresponding cosine similarity score with the original key of upper level


### pre-process stages

We describe the necessary files that need to be executed for producing the official results of the proposed ONZSL algorithm. These steps concern the full examined test set, while the proposed algorithm can be applied to each one arrived test instance.


- **obtain_text_embeddings.py** 🠊 this script computes the embeddings of the *input text files* and saves them into corresponding pickles with the bioBERT embeddings at a sentence-level. The context of each pickle is a Series object, whose each item is a list with **p** Numpy arrays of dimension (768,)

- **calculate_similarities.py** 🠊 based on the pickles that are created from the above script, we compute the Cosine similarity scores of each sentence per different abstract with either the top-100 most frequent novel labels (choice == 1) or the existing known labels per instance (choice == 2). The produced 5 .pickles per case are equal in total to 1.58 GB and 213 MB, respectively.


### on-the-fly baseline

- **NN_baseline proprocess.py** 🠊 Create the appropriate .pickle files per examined input .pickle (batch) where each instance corresponds to one dictionary structure: 
                                   
                                   - keys -> instanceX, value: dictionary structure (dictX)
                                   - dictX: keys -> investigated labels, value: i) the Manhattan distance (~1.5GB) or ii) the corresponding cosine similarity (~2.3 GB)
                                   from the bioBERT embedding of each instance's sentence
                            
                            
## Process:

 - run NN_baseline_preprocess.py 
 
 Input: path, 'top_100_labels_embeddings.pickle', 44k pickles folder with 5 pickles (~2GB)
 
 - run record_label_similaritiy_scores.py -> produce pickle (1.5 GB) with dataframes for eq1
 
 Input: path, top_100_labels.txt, pure_zero_shot_test_set_top100.txt, noisy_labels_70percent.pickle
