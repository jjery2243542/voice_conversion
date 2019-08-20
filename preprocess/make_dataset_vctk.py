import h5py
import numpy as np
import sys
import os
import glob
import re
from collections import defaultdict
#from tacotron.audio import load_wav, spectrogram, melspectrogram
from tacotron.norm_utils import get_spectrograms 

def read_speaker_info(path='/storage/datasets/VCTK/VCTK-Corpus/speaker-info.txt'):
    accent2speaker = defaultdict(lambda: [])
    with open(path) as f:
        splited_lines = [line.strip().split() for line in f][1:]
        speakers = [line[0] for line in splited_lines]
        regions = [line[3] for line in splited_lines]
        for speaker, region in zip(speakers, regions):
            accent2speaker[region].append(speaker)
    return accent2speaker


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('usage: python3 make_dataset_vctk.py [data root directory (VCTK-Corpus)] [h5py path] '
                '[training proportion]')
        exit(0)

    root_dir = sys.argv[1]
    h5py_path = sys.argv[2]
    proportion = float(sys.argv[3])

    accent2speaker = read_speaker_info(os.path.join(root_dir, 'speaker-info.txt'))
    filename_groups = defaultdict(lambda : [])
    with h5py.File(h5py_path, 'w') as f_h5:
        filenames = sorted(glob.glob(os.path.join(root_dir, 'wav48/*/*.wav')))
        for filename in filenames:
            # divide into groups
            sub_filename = filename.strip().split('/')[-1]
            # format: p{speaker}_{sid}.wav
            speaker_id, utt_id = re.search(r'p(\d+)_(\d+)\.wav', sub_filename).groups()
            filename_groups[speaker_id].append(filename)
        for speaker_id, filenames in filename_groups.items():
            # only use the speakers who are English accent.
            if speaker_id not in accent2speaker['English']:
                continue
            print('processing {}'.format(speaker_id))
            train_size = int(len(filenames) * proportion)
            for i, filename in enumerate(filenames):
                sub_filename = filename.strip().split('/')[-1]
                # format: p{speaker}_{sid}.wav
                speaker_id, utt_id = re.search(r'p(\d+)_(\d+)\.wav', sub_filename).groups()
                _, lin_spec = get_spectrograms(filename)
                if i < train_size:
                    datatype = 'train'
                else:
                    datatype = 'test'
                f_h5.create_dataset(f'{datatype}/{speaker_id}/{utt_id}', \
                    data=lin_spec, dtype=np.float32)
