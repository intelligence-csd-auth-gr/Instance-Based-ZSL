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

- **predictions_label_occurence.pickle** 🠊 the output of the label occurence stage for the whole test size into a list structure, where each item is into the next format: 'label_1#label_2#...#label_r'

- **abstract_occurence_predictions** 🠊 the output of the label occurence stage for the whole test size into a list structure, where each item contains a list with the predictions

- **noisy_labels_70percent.pickle** 🠊 a dictionary structure whose: *keys* the known labels that are replaced during the imperfect oracle scenario, *values* a dictionary with 20 randomly selected MeSH terms in the role of keys, and their corresponding cosine similarity score with the original key of upper level

- **novel_labels_actual.pickle** 🠊 the actual novel labels of the whole test set, being stored as a list with list items per instance, which contain the separate labels inside them

- **top_100_labels.txt** 🠊 the top100 most frequent novel labels into .txt format

- **known_y_labels.csv** 🠊 the known labels of the whole examined set into a .csv format


### pre-process stages

We describe the necessary files that need to be executed for producing the official results of the proposed ONZSL algorithm. These steps concern the full examined test set, while the proposed algorithm can be applied to each one arrived test instance.


- **obtain_text_embeddings.py** 🠊 this script computes the embeddings of the *input text files* and saves them into corresponding pickles with the bioBERT embeddings at a sentence-level. The context of each pickle is a Series object, whose each item is a list with **p** Numpy arrays of dimension (768,) (total size: 1.58 GB)

- **calculate_similarities.py** 🠊 based on the pickles that are created from the above script, we compute the Cosine similarity scores of each sentence per different abstract with either the top-100 most frequent novel labels (choice == 1) or the existing known labels per instance (choice == 2). The produced 5 .pickles per case are equal in total to 6.75 GB and 213 MB, respectively

- **add_noisy_labels.py** 🠊 the process under which the noisy_labels_70percent.pickle file is produced


### on-the-fly baseline

- **NN_baseline proprocess.py** 🠊 this script creates the appropriate .pickle files per examined input .pickle (batch) where each instance corresponds to one dictionary structure (dictA): 
* dictA: keys -> instanceX (e.g. 'instance0'), value: another dictionary structure (dictX)
* dictX: keys -> investigated labels (top100 labels e.g. 'Flexural Strength'), value: a list with i) the Manhattan distance (~1.5GB) or ii) the corresponding cosine similarity (~2.3 GB) from the bioBERT embedding of each instance's sentence and the specific label (length of list is equal to  **p** which depends on sentence's length)

- **NN_summary.py** 🠊 given the path with the .pickles created by the above script, we concatenate the necessary information (decisions, best 3 scores) into a common .pickle per different distance function for the whole examined test set. The 'best 3 scores' values are not exploited further into this work

- **NN_bioBERT_evaluate** 🠊 given the two summarization files from the above script into one folder, it computes the Coverage and 1-error metrics, as well as plots and provided some information about the achieved rankings

### ONZSL

- **record_label_similaritiy_scores.py** 🠊 this files examines the known label vector of each given abstract (implementing 3 different assumptions: i) all known labels, ii) 70% of the known labels, iii) 70% of the known labels along with some noisy labels are provided) and exports a .pickle which contains for each examined instance a pandas DataFrame with the relative similarities of the investigated novel labels and the existing ones, respectively

- **compute_weights_per_instance.py** 🠊 this script exploits the pickles with the pre-computed similarities and obtains the stores the max similarity score per instance for every examined label, as the Equation 2 of the original paper presents

- **weighted_unweighted_approaches.py** 🠊 we combine the the label dependencies through the similarities that are stored previously either along with the weights from the above file (weighted version) or without (unweighted version) for implementing the RankScore of Equation 3 into the original work

- **occurence.py** 🠊 this script computes the last step of the proposed algorithm, examining if the label names are detected into each abstract segment, otherwise it returns None. Its produced file is found in the pre-comouted folder (predictions_label_occurence.pickle)

- **apply_occurence.py** 🠊 this file combines the predictions from label occurence with any provided ranking from the (un)weighted stages, storing the final decisions of the proposed algorithm into one proper .pickle file per time 

- **ONZSL_evaluate.py** 🠊 the script through which the Coverage and 1-error metric are computed for the proposed algorithm, as well as for the rest ones, apart for the NN_baselines, as they are recorded into the original work
                            
## Process:

(To do)


## Requirements/Dependencies

Our code has been tested on Windows10 using python 3.7.6. The mentioned time responses correspond to a working station embedded with Intel Core i7-9700 (3GHz) processor and 32 GB RAM. The next libaries are necessary:

- Numpy
- bioBERT
- Spacy
- Pandas
- Seaborn and Matplotlib (for graphing)


## Developed by: 

|           Name  (English/Greek)            |      e-mail          |
| -------------------------------------------| ---------------------|
| Nikolaos Mylonas    (Νικόλαος Μυλωνάς)     | myloniko@csd.auth.gr |
| Stamatis Karlos     (Σταμάτης Κάρλος)      | stkarlos@csd.auth.gr |
| Grigorios Tsoumakas (Γρηγόριος Τσουμάκας)  | greg@csd.auth.gr     |

## Funded by

The research work was supported by the Hellenic Foundation forResearch and Innovation (H.F.R.I.) under the “First Call for H.F.R.I.Research Projects to support Faculty members and Researchers and the procurement of high-cost research equipment grant” (ProjectNumber: 514).

## Additional resources

- [AMULET project](https://www.linkedin.com/showcase/amulet-project/about/)
- [Academic Team's page](https://intelligence.csd.auth.gr/#)
 
 ![amulet-logo](https://user-images.githubusercontent.com/6009931/87019683-9204ad00-c1db-11ea-9394-855d1d3b41b3.png)
