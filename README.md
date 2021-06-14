# Instance-Based Zero-Shot-Learning (IBZSL)

## Instance-Based Zero-Shot Learning for Semi-Automatic MeSH Indexing

This repository constitutes an implementation of the submitted paper to the Pattern Recognition Letters ([PRLetters](https://www.journals.elsevier.com/pattern-recognition-letters)) journal with the above title by our research team.
It has been created for facilitating the reproducubility of the proposed online Zero-shot Learning algorithm, applied on data mined from the MeSH 2020 database, as it is defined by the coresponding [BioASQ challenge](http://bioasq.org) regarding the Biomedical indexing.

![PRLetters_mesh_github](https://user-images.githubusercontent.com/6009931/121053214-cb66e380-c7c3-11eb-92b8-14817d2684b7.jpg)


Some brief documentation is provided here for running all the necessary steps, since uploading all the source file demands several of GBs (a permanent link is going to be provided). 


### Source data 

- **abstracts from MeSH 2020 with top100 labels appeared.7z** 🠊 contains 5 .txt files (split per 10,000 instances) which include the abstracts from MeSH 2020 that at least one of the top-100 most frequent novel labels (*frLabel*) appears into its label space.
- **frequency_novel_labels_test_set_MeSH2020.csv**  🠊 the names of the *frLabel* in the examined version of the BioASQ dataset, along with the frequency of appearence per each one into the examined test set.

- **pure_zero_shot_test_set_top100.7z** 🠊 it actually contains all the examined test abstracts into one *.txt* file.

### Pre-computed files

Here are added some files for accelerating the execution of several needed computations:

- **dict_similarities_novel_known_labels.pickle** 🠊 a dictionary structure whose: 
 1. *keys*: are the names of the *frLabel* set,
 2. *values*: the Cosine similarity score of the each key with all of the actual known labels, so as to avoid computing such scores per instance by making simple searches.

- **dict_similarities_novel_MTI_labels.pickle** 🠊 a dictionary structure whose: 
 1. *keys*: are the names of the *frLabel* set,
 2. *values*: the Cosine similarity score of the each key with all of the predicted existing labels based on the MTI tool, so as to avoid computing such scores per instance by making simple searches.

- **novel_labels_embeddings.pickle** 🠊 a dictionary structure whose:
 1.*keys*: are the names of the *frLabel* set,
 2.*values*: the bioBERT embedding vector of each key stored as a Numpy array *(768,)*.

- **known_labels_embeddings.pickle** 🠊 a dictionary structure whose:
 1. *keys*: are the names of all the known labels for the examined test set, 
 2. *values*: their bioBERT embedding vector of each key stored as a Numpy array *(768,)*.

- **MTI_labels_embeddings.pickle** 🠊 a dictionary structure whose:
 1. *keys*: are the names of all the predicted labels for the examined test set by the MTI tool, 
 2. *values*: their bioBERT embedding vector of each key stored as a Numpy array *(768,)*.

- **known_labels.pickle** 🠊 the known labels of the whole test set, being stored as a list with list items per instance, which contain the separate labels inside them. We depict here the first 5 items of this list:

```
[ ['Fluorodeoxyglucose F18', 'Glycolysis', 'Humans', 'Lymphoma, Extranodal NK-T-Cell', 'Positron Emission Tomography Computed Tomography','Positron-Emission         Tomography','Prognosis','Radiopharmaceuticals','Retrospective Studies','Survival Analysis','Tumor Burden'],
  ['Bacillus subtilis','China','Fermentation','Glucosidases','Peptide Hydrolases','RNA, Ribosomal, 16S','Soy Foods'],
  ['Catheter Ablation','Delivery, Obstetric','Female','Fetofetal Transfusion','Gestational Age','Humans','Infant, Newborn','Pregnancy','Pregnancy Outcome','Pregnancy Reduction, Multifetal','Pregnancy, Twin','Retrospective Studies','Treatment Outcome','Twins, Monozygotic'] ]
```

- **predictions_label_occurence.pickle** 🠊 the output of the label occurence stage for the whole test size into a list structure, where each item is into the next format: 
 'label_1#label_2#...#label_r'. We depict here the first 3 items of this list:

```
['Data Analysis#Progression-Free Survival',
 'None',
 'Information Technology#Gestational Weight Gain#Radiofrequency Ablation']
```

- **noisy_labels_70percent.pickle** 🠊 a dictionary structure whose (currently there are 11,791 different keys, but this number can vary based on the random character of the artificial injected noise):
 1. *keys*: the known labels that are replaced during the imperfect oracle scenario, 
 2. *values*: a dictionary with 20 randomly selected MeSH terms in the role of keys, and their corresponding *Cosine similarity score* with the original key of upper level.

- **novel_labels_actual.pickle** 🠊 the actual novel labels of the whole test set, being stored as a list with list items per instance, which contain the separate labels inside them. We depict here the first 3 items of this list:

```
[['Progression-Free Survival'],
 ['Fermented Foods and Beverages'],
 ['Radiofrequency Ablation']]
 ```

- **top_100_labels.txt** 🠊 the *frLabel* set into *.txt* format.

- **known_y_labels.csv** 🠊 the known labels of the whole examined set into a *.csv* format.


### Pre-processing stages

We describe the necessary files that need to be executed for producing the official results of the proposed **IBZSL** algorithm. These steps concern the full examined test set, while the proposed algorithm can be applied to each one arrived test instance, providing thus independent predictions when exposed to a new test instance, thus avoiding the computational burden of building a model based on any available training data. 


- **obtain_text_embeddings.py** 🠊 this script computes the embeddings of the *input text files* and saves them into corresponding pickles with the bioBERT embeddings at a sentence-level. The context of each pickle is a *Series* object, whose each item is a list with **p** Numpy arrays of dimension (768,) (total size: 1.58 GB).

- **calculate_similarities.py** 🠊 based on the pickles that are created from the above script, we compute the Cosine similarity scores of each sentence per different abstract with either the *frLabel* set (choice == 1) or the existing known labels per instance (choice == 2). The 5 produced *.pickle* files per case are equal in total to 6.75 GB and 213 MB, respectively.

- **add_noisy_labels.py** 🠊 the process under which the **noisy_labels_70percent.pickle** file is produced, which is later used for simulating the scenario under which noisy predictions regarding the existing labels are provided before the proposed **IBZSL** approach is applied.

- **compute_label_similarities_per_pair.py** 🠊 this script computes the *Cosine similarity score* of each pair between the labels into the *frLabel* set and all the distinct labels that appear at least once into the total predictions. Currently, there are two choices: i) All the actual labels, ii) The labels that are predicted by the MTI tool. Thus, the dictionary object that is produced contains 100 keys (the number of the novel labels, which equals to the size of the *frLabel* set) and *k* values per key, where *k* equals to the number of all the different existing labels. For the two implemented cases, *k*  equals to 17,482 and 22,227, respectively. At the same time, we store here the embeddings vectors of each label into the *frLabel* set (**novel_labels_embeddings.pickle**), as well as the same vectors for the existing labels per case (**known_labels_embeddings.pickle**, **MTI_labels_embeddings.pickle**)


### On-the-fly baseline

- **NN_baseline proprocess.py** 🠊 this script creates the appropriate output (storing into *.pickle* format) per examined input batch (stored also in *.pickle* format) where each instance corresponds to one dictionary structure (dictA): 
  * dictA: keys -> instanceX (e.g. 'instance0'), value: another dictionary structure (dictX),
  * dictX: keys -> investigated labels (*frLabel* set e.g. 'Flexural Strength'), value: a list with i) the Manhattan distance (~1.5GB) or ii) the corresponding cosine similarity   (~2.3 GB) from the bioBERT embedding of each instance's sentence and the specific label (length of list is equal to  **p** which depends on sentence's length).

- **NN_summary.py** 🠊 given the path with the *.pickle* files created by the above script, we concatenate the necessary information (decisions, best 3 scores) into a common *.pickle* file per different distance function for the whole examined test set. The 'best 3 scores' values are not exploited further into this work.

- **NN_bioBERT_evaluate** 🠊 given the two summarization files from the above script into one folder, it computes the metrics of *Coverage* and *1-error*, as well as produces some appropriate histograms plots and prints additional information about the achieved rankings.

### IBZSL

- **record_label_similaritiy_scores.py** 🠊 this files examines the known label vector of each given abstract (implementing 3 different assumptions: i) all known labels, ii) 70% of the known labels, iii) 70% of the known labels along with some noisy labels are provided) and exports a *.pickle* file which contains for each examined instance a *Pandas DataFrame* with the relative similarities of the investigated novel labels and the existing ones, respectively.

- **compute_weights_per_instance.py** 🠊 this script exploits the pickles with the pre-computed similarities and obtains the stored *max similarity score* per instance for every examined label, according to the *Equation 6* of the original paper.

- **weighted_unweighted_approaches.py** 🠊 we combine the label dependencies through the similarities that are stored previously either along with the weights from the above file (weighted version) or without (unweighted version) for implementing the RankScore of Equation 3 into the original work.

- **occurence.py** 🠊 this script computes the last step of the proposed algorithm, examining if the label names are detected into each abstract segment, otherwise it returns None. Its produced file is found in the pre-comouted folder (*predictions_label_occurence.pickle*).

- **apply_occurence.py** 🠊 this file combines the predictions from label occurence with any provided ranking from the (un)weighted stages, storing the final decisions of the proposed algorithm into a proper *.pickle* file per time.

- **IBZSL_evaluate.py** 🠊 the script through which the *Coverage* and the *1-error* metrics are computed for the proposed algorithm, as well as for the rest ones, apart for the NN_baselines, since these computations are computed into their corresponding files. Additionally, appropriate histograms and useful stats (frequency of correct predictions) are computed here, which are not recorded into the manuscript due to lack of space.
                            
### Results

Here are added the finally produced .pickle files which facilitate the reproducibility of the results reported in Table 1 of the manuscript.

- **ideal oracle** and **imperfect oracle** 🠊 exploited under the **IBZSL_evaluate.py**
- **baselines** 🠊 exploited under the **NN_bioBERT_evaluate.py** (first unzip the existing *.7z* files)



## Requirements/Dependencies

Our code has been tested on Windows10 using python 3.7.6. The next libaries are necessary:

- Numpy
- bioBERT
- Spacy
- Pandas
- Seaborn and Matplotlib (for graphing)


## Developed by: 

|           Name  (English/Greek)            |      e-mail          |
| -------------------------------------------| ---------------------|
| Stamatis Karlos     (Σταμάτης Κάρλος)      | stkarlos@csd.auth.gr |
| Nikolaos Mylonas    (Νικόλαος Μυλωνάς)     | myloniko@csd.auth.gr |
| Grigorios Tsoumakas (Γρηγόριος Τσουμάκας)  |  greg@csd.auth.gr    |

## Funded by

The research work was supported by the Hellenic Foundation forResearch and Innovation (H.F.R.I.) under the “First Call for H.F.R.I.Research Projects to support Faculty members and Researchers and the procurement of high-cost research equipment grant” (ProjectNumber: 514).

## Additional resources

- [AMULET project](https://www.linkedin.com/showcase/amulet-project/about/)
- [Academic Team's page](https://intelligence.csd.auth.gr/#)
 
 ![amulet-logo](https://user-images.githubusercontent.com/6009931/87019683-9204ad00-c1db-11ea-9394-855d1d3b41b3.png)
