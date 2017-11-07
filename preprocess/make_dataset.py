import h5py
import numpy as np
import sys
import os
import glob
import re
from collections import defaultdict
from utils import get_spectrograms

'''DEPRECATE
def sort_key(x):
    sub = x.split('/')[-1]
    l = re.split('_|-', sub.strip('.npy'))
    if len(l[-1]) == 1:
        l[-1] = '0{}'.format(l[-1])
    return ''.join(l)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: python3 make_dataset.py [h5py_path] [numpy_dir]')
        exit(0)
    h5py_path = sys.argv[1]
    np_dir = sys.argv[2]

    with h5py.File(h5py_path, 'w') as f_h5:
        for dataset, dir_name in zip(['train', 'test'], ['train-clean-100', 'train-clean-100_infer']):
            utt = []
            # tuple: (speaker, chapter, utt)
            prev_utt = None
            np_sub_dir = os.path.join(np_dir, dir_name)   
            for i, filename in enumerate(sorted(glob.glob(os.path.join(np_sub_dir, '*')), key=sort_key)):
                speaker_id, chapter_id, other = filename.split('/')[-1].split('-')
                utt_id, seg_id = other[:-4].split('_')
                current_utt = (speaker_id, chapter_id, utt_id)
                d = np.load(filename)
                utt.append(d)
                if prev_utt != current_utt and i != 0:
                    print('dump {}'.format(prev_utt))
                    data=np.array(utt, dtype=np.float32)
                    print(data.shape)
                    grp = f_h5.create_dataset(
                        '{}/{}/{}-{}'.format(dataset, speaker_id, chapter_id, utt_id),
                        data=data,
                        dtype=np.float32,
                    )
                    utt = []
                prev_utt = current_utt
            # last utt
            if len(utt) > 0:
                print('dump {}'.format(prev_utt))
                data=np.array(utt, dtype=np.float32)
                print(data.shape)
                grp = f_h5.create_dataset(
                    '{}/{}/{}-{}'.format(dataset, speaker_id, chapter_id, utt_id),
                    data=data,
                    dtype=np.float32,
                )

'''

root_dir='/storage/LibriSpeech/LibriSpeech/train-clean-100'

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: python3 make_dataset.py [h5py_path]')
        exit(0)
    h5py_path=sys.argv[1]
    filename_groups = defaultdict(lambda : [])
    with h5py.File(h5py_path, 'w') as f_h5:
        grps = [f_h5.create_group('train'), f_h5.create_group('test')]
        filenames = sorted(glob.glob(os.path.join(root_dir, '*/*/*.flac')))
        for filename in filenames:
            # divide into groups
            speaker_id, chapter_id, segment_id = filename.strip().split('/')[-1].strip('.flac').split('-')
            filename_groups[speaker_id].append(filename)
        for speaker_id, filenames in filename_groups.items():
            print('processing {}'.format(speaker_id))
            for filename in filenames[:-1]:
                print(filename)
                speaker_id, chapter_id, segment_id = filename.strip().split('/')[-1].strip('.flac').split('-')
                mel_spec, lin_spec = get_spectrograms(filename)
                grps[0].create_dataset('{}/{}-{}/mel'.format(speaker_id, chapter_id, segment_id), \
                    data=mel_spec, dtype=np.float32)
                grps[0].create_dataset('{}/{}-{}/lin'.format(speaker_id, chapter_id, segment_id),\
                    data=lin_spec, dtype=np.float32)
            # the last segment put into testset
            filename = filenames[-1]
            speaker_id, chapter_id, segment_id = filename.strip().split('/')[-1].strip('.flac').split('-')
            mel_spec, lin_spec = get_spectrograms(filename)
            grps[1].create_dataset('{}/{}-{}/mel'.format(speaker_id, chapter_id, segment_id), \
                data=mel_spec, dtype=np.float32)
            grps[1].create_dataset('{}/{}-{}/lin'.format(speaker_id, chapter_id, segment_id), \
                data=lin_spec, dtype=np.float32)

