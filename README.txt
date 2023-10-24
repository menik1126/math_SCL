Please overwrite the file prune.py with the /lib/python3.6/site-packages/torch/nn/utils/prune.py file in the dependencies.




If you want to view the running results of the competitor network on Math23k, please set is_prune2test to true, (is_RDrop)  to true and USE_APE to false. 
Next run run_seq2tree_bert_ultimate_mix.py

If you want To view the running results of the source network on Math23k, please set is_prune2test to false, (is_RDrop)  to true and USE_APE to false.
Next run run_seq2tree_bert_ultimate_mix.py

If you want to view the results of the five-fold cross-validation of the competitor and source networks on Math23k, please follow the above settings and run:
run_seq2tree_bert_ultimate_5fold_best.py

If you want to view the running results on Ape210k, please set (is_RDrop) in the config.py file to true and USE_APE to True. 
By changing is_prune2test to view the running results of competitor and source network on Ape210k
Next run run_seq2tree_APE_early_SP.py

If you want to view Source network w/o pruning and Competitor network w/o pruning, please set is_RDrop in the config file to true, is_prune2test to false, and USE_APE to false.
Next run run_seq2tree_bert_ultimate_divide_epoch.py

If you want to view Source network w/o MT and Competitor network w/o MT, please set is_RDrop to false and is_prune2test to false and true respectively.
Next run run_seq2tree_bert_ultimate_comp.py