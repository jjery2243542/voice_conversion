import h5py
import numpy as np
import pickle 

h5_path='/storage/feature/voice_conversion/teacher_voice/trim_tv.h5'
pkl_path = '/storage/feature/voice_conversion/teacher_voice/trim_tv.pkl'

speakers = ['hsuan_tien_lin', 'hung_yi_lee', 'lin_shan_lee', 'yun_nung_chen']

dictionary = {}

with h5py.File(h5_path, 'r') as f_h5:
    for dset in ['train', 'test']:
        for speaker in speakers:
            print(f'processing speaker {speaker}')
            utt_list = list(f_h5[f'{dset}/{speaker}'].keys())
            # take 1/3
            utt_list = utt_list[:len(utt_list) // 5]
            for utt_id in utt_list:
                #dictionary[f'{dset}/{speaker}/{utt_id}/mel'] = f_h5[f'{dset}/{speaker}/{utt_id}/mel'][()]
                dictionary[f'{dset}/{speaker}/{utt_id}/lin'] = f_h5[f'{dset}/{speaker}/{utt_id}/lin'][()]

with open(pkl_path, 'wb') as f:
    pickle.dump(dictionary, f)
