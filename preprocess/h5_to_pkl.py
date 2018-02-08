import h5py
import numpy as np
import pickle 

h5_path='/storage/feature/voice_conversion/vctk/trim_log_vctk.h5'
pkl_path = '/storage/feature/voice_conversion/vctk/trim_log_vctk.pkl'
speakers = ['225', '226', '227', '228', '229', '230', '232', '237']

dictionary = {}

with h5py.File(h5_path, 'r') as f_h5:
    for dset in ['train', 'test']:
        for speaker in speakers:
            print(f'processing speaker-{speaker}')
            for utt_id in f_h5[f'{dset}/{speaker}']:
                dictionary[f'{dset}/{speaker}/{utt_id}/mel'] = f_h5[f'{dset}/{speaker}/{utt_id}/mel'][()]
                dictionary[f'{dset}/{speaker}/{utt_id}/lin'] = f_h5[f'{dset}/{speaker}/{utt_id}/lin'][()]

with open(pkl_path, 'wb') as f:
    pickle.dump(dictionary, f)
