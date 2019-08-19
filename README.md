# Multi-target Voice Conversion without Parallel Data by Adversarially Learning Disentangled Audio Representations
This is the official implementation of the paper [Multi-target Voice Conversion without Parallel Data by Adversarially Learning Disentangled Audio Representations](https://arxiv.org/abs/1804.02812).
You can find the demo webpage [here](https://jjery2243542.github.io/voice_conversion_demo/), and the pretrained model [here](http://speech.ee.ntu.edu.tw/~jjery2243542/resource/model/is18/model.pkl).

# Dependency
- python 3.6+
- pytorch 1.0.1
- h5py 2.8
- tensorboardX
We also use some preprocess script from [Kyubyong/tacotron](https://github.com/Kyubyong/tacotron).

# Preprocess
Our model is trained on [CSTR VCTK Corpus](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html).

### Feature extraction
We use the code from [Kyubyong/tacotron](https://github.com/Kyubyong/tacotron) to extract feature. The default paprameters can be found at ```preprocess/tacotron/norm_utils.py```.

The configuration for preprocess is at ```preprocess/vctk.config```, where: 
- **data_root_dir**: the path of VCTK Corpus (VCTK-Corpus).
- **h5py_path**: the path to store extracted features.
- **index_path**: the path to store sampled segments.
- **traini_proportion**: the proportion of training utterances. Default: 0.9.
- **n_samples**: the number of sampled samples. Default: 500000.
- **seg_len**: the length of sampled segments. Default: 128.
- **speaker_used_path**: the path of used speaker list. Our speakers set used in the paper is [here](http://speech.ee.ntu.edu.tw/~jjery2243542/resource/model/is18/en_speaker_used.txt).

Once you edited the config file, you can run ```preprocess.sh``` to preprocess the dataset.

# Training
You can start training by running ```main.py```. The arguments are listed below.
- **--load_model**: whether to load the model from checkpoint.
- **-flag**: flag of this training episode for tensorboard. Default: train.
- **-hps_path**: the path of hyper-parameters set. You can find the default setting at ```vctk.json```.
- **--load_model_path**: If **--load_model** is on, it will load the model parameters from this path.
- **-dataset_path**: the path of processed features (.h5).
- **-index_path**: the path of sampled segment indexes (.json).
- **-output_model_path**: the path to store trained model. 

# Testing
You can inference by running ```python3 test.py```. The arguments are listed below.
- **-hps**: the path of hyper-parameter set. Default: vctk.json
- **-m**: the path of model checkpoint to load.
- **-s**: the path of source .wav file.
- **-t**: the index of target speaker id (integer). Same order as the speaker list (```en_speaker_used.txt```).
- **-o**: output .wav path.
- **-sr**: sample rate of the output .wav file. Default: 16000.
- **--use_gen**: if the flag is on, inference will use generator. Default: True.

# Reference
Please cite our paper if you find this repository useful.
```
@article{chou2018multi,
  title={Multi-target voice conversion without parallel data by adversarially learning disentangled audio representations},
  author={Chou, Ju-chieh and Yeh, Cheng-chieh and Lee, Hung-yi and Lee, Lin-shan},
  journal={arXiv preprint arXiv:1804.02812},
  year={2018}
}
```

# Contact
If you have any question about the paper or the code, feel free to email me at [jjery2243542@gmail.com](jjery2243542@gmail.com).
