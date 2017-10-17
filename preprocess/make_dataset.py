import h5py
import numpy as np
import sys
import os
import glob

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: python3 make_dataset.py [h5py_path] [numpy_dir]')
        exit(0)
    h5py_path = sys.argv[1]
    np_dir = sys.argv[2]

    with h5py.File(h5py_path, 'w') as f_h5:
        for dataset, dir_name in zip(['train', 'test'], ['train-clean-100', 'train-clean-100_infer']):
            np_sub_dir = os.path.join(np_dir, dir_name)   
            for filename in sorted(glob.glob(os.path.join(np_sub_dir, '*'))):
                speaker_id, chapter_id, other = filename.split('/')[-1].split('-')
                utt_id, seg_id = other[:-4].split('_')
                print('{}, {}, {}, {}'.format(speaker_id, chapter_id, utt_id, seg_id))
                d = np.load(filename)
                grp = f_h5.create_dataset(
                    '{}/{}/{}/{}/{}'.format(dataset, speaker_id, chapter_id, utt_id, seg_id),
                    data=d,
                    dtype=np.float32,
                )

        
