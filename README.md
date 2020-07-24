# AMULET_SCIS

## Semi-automatic indexing method for BioASQ task through on-the-fly ZSL method.

link: https://mc03.manuscriptcentral.com/scis


## Process:

 - run NN_baseline_preprocess.py 
 
 Input: path, 'top_100_labels_embeddings.pickle', 44k pickles folder with 5 pickles (~2GB)
 
 - run record_label_similaritiy_scores.py -> produce pickle (1.5 GB) with dataframes for eq1
 
 Input: path, top_100_labels.txt, pure_zero_shot_test_set_top100.txt, noisy_labels_70percent.pickle
