import h5py
import numpy as np
import sys
import os
import glob
import re
from collections import defaultdict
#from tacotron.audio import load_wav, spectrogram, melspectrogram
from tacotron.utils import get_spectrograms 

root_dir='/storage/datasets/VCTK/VCTK-Corpus/wav48'
train_split=0.9

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: python3 make_dataset_vctk.py [h5py_path]')
        exit(0)
    h5py_path=sys.argv[1]
    filename_groups = defaultdict(lambda : [])
    with h5py.File(h5py_path, 'w') as f_h5:
        filenames = sorted(glob.glob(os.path.join(root_dir, '*/*.wav')))
        for filename in filenames:
            # divide into groups
            sub_filename = filename.strip().split('/')[-1]
            # format: p{speaker}_{sid}.wav
            speaker_id, utt_id = re.match(r'p(\d+)_(\d+)\.wav', sub_filename).groups()
            filename_groups[speaker_id].append(filename)
        for speaker_id, filenames in filename_groups.items():
            print('processing {}'.format(speaker_id))
            train_size = int(len(filenames) * train_split)
            for i, filename in enumerate(filenames):
                print(filename)
                sub_filename = filename.strip().split('/')[-1]
                # format: p{speaker}_{sid}.wav
                speaker_id, utt_id = re.match(r'p(\d+)_(\d+)\.wav', sub_filename).groups()
                #wav_data = load_wav(filename)
                #lin_spec = spectrogram(wav_data).astype(np.float32).T
                #mel_spec = melspectrogram(wav_data).astype(np.float32).T
                mel_spec, lin_spec = get_spectrograms(filename)
                eps = 1e-10
                log_mel_spec, log_lin_spec = np.log(mel_spec+eps), np.log(lin_spec+eps)
                if i < train_size:
                    datatype = 'train'
                else:
                    datatype = 'test'
                f_h5.create_dataset(f'{datatype}/{speaker_id}/{utt_id}/mel', \
                    data=log_mel_spec, dtype=np.float32)
                f_h5.create_dataset(f'{datatype}/{speaker_id}/{utt_id}/lin', \
                    data=log_lin_spec, dtype=np.float32)
