# Multi-target Voice Conversion without Parallel Data by Adversarially Learning Disentangled Audio Representations

This is the source code for [Multi-target Voice Conversion without Parallel Data by Adversarially Learning Disentangled Audio Representations](https://arxiv.org/pdf/1804.02812), which is accepted in Interspeech 2018, and selected as the finallist of best student paper award.

You can find the conversion sample at [here](https://jjery2243542.github.io/voice_conversion_demo/).
Pretrained model is available at [here](http://speech.ee.ntu.edu.tw/~jjery2243542/model.pkl).

If you want to trained the model by yourself, please refer to new-branch, the hyperparameters are hps/vctk.json.
#### training steps:
- preprocess/make\_dataset\_vctk.py to generated the feature (you need to install h5py package).
- preprocess/make\_single\_samples.py to generate the training segments and testing segments (need to change the variable in the code to switch to testing data).
- train the model with main.py (hps/vctk.json).
- generate the samples with convert.py.

The source code is currently a little messy, if you have any problem, feel free to email me (jjery2243542@gmail.com).
