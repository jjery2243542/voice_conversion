# Voice conversion via disentangle context representation (Code for paper:https://arxiv.org/pdf/1804.02812.pdf )

Steps for training:
1.) Use make_dataset_vctk.py to convert the VCTK corpus into training data.
2.) Use make_single_samples.py to make the data feed dictionary/log used in training.
3.) Train using main.py
4.) Generate converted samples using convert.py

