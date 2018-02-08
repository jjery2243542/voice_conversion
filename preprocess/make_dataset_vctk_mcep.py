import h5py
import numpy as np
import sys
import os
import glob
import re
from collections import defaultdict
#from tacotron.audio import load_wav, spectrogram, melspectrogram
#from tacotron.utils import get_spectrograms 
from tacotron.mcep import wav2mcep

root_dir='/storage/datasets/VCTK/VCTK-Corpus/wav48/'
speaker_info_path='/storage/datasets/VCTK/VCTK-Corpus/speaker-info.txt'
train_split=0.9

def read_info(speaker_info_path):
    info_dict = defaultdict(lambda : []) 
    with open(speaker_info_path, 'r') as f_in:
        lines = f_in.readlines()[1:]
        infos = [line.strip().split() for line in lines]
        for info in infos:
            info_dict[info[3]].append(info[0])
    return info_dict

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: python3 make_dataset_vctk.py [h5py_path]')
        exit(0)
    h5py_path=sys.argv[1]
    filename_groups = defaultdict(lambda : [])
    accent_list = read_info(speaker_info_path)
    eps = 1e-10
    with h5py.File(h5py_path, 'w') as f_h5:
        filenames = sorted(glob.glob(os.path.join(root_dir, '*/*.wav')))
        for filename in filenames:
            # divide into groups
            sub_filename = filename.strip().split('/')[-1]
            # format: p{speaker}_{sid}.wav
            speaker_id, utt_id = re.match(r'p(\d+)_(\d+)\.wav', sub_filename).groups()
            filename_groups[speaker_id].append(filename)
        for speaker_id in accent_list['English'] + accent_list['America']:
            print('processing {}'.format(speaker_id))
            filenames = filename_groups[speaker_id]
            train_size = int(len(filenames) * train_split)
            for i, filename in enumerate(filenames):
                print(filename)
                sub_filename = filename.strip().split('/')[-1]
                # format: p{speaker}_{sid}.wav
                speaker_id, utt_id = re.match(r'p(\d+)_(\d+)\.wav', sub_filename).groups()
                f0, ap, mc = wav2mcep(filename)
                log_f0 = np.log(f0 + eps)
                f0_mean, f0_std = np.mean(log_f0), np.std(log_f0)
                if i < train_size:
                    datatype = 'train'
                else:
                    datatype = 'test'
                f_h5.create_dataset(f'{datatype}/{speaker_id}/{utt_id}/log_f0', \
                    data=log_f0, dtype=np.float32)
                f_h5.create_dataset(f'{datatype}/{speaker_id}/{utt_id}/f0_mean', \
                    data=f0_mean, dtype=np.float32)
                f_h5.create_dataset(f'{datatype}/{speaker_id}/{utt_id}/f0_std', \
                    data=f0_std, dtype=np.float32)
                f_h5.create_dataset(f'{datatype}/{speaker_id}/{utt_id}/ap', \
                    data=ap, dtype=np.float32)
                f_h5.create_dataset(f'{datatype}/{speaker_id}/{utt_id}/mc', \
                    data=mc, dtype=np.float32)
                del f0, log_f0, f0_mean, f0_std, ap, mc

