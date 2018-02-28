import h5py
import numpy as np
import pickle 

h5_path='/storage/feature/voice_conversion/vcc/trim_log.h5'
pkl_path = '/storage/feature/voice_conversion/vcc/trim_log.pkl'
speakers = ['SF1', 'SF2', 'SF3', 'SM1', 'SM2', 'TF1', 'TF2', 'TM1', 'TM2', 'TM3']

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
