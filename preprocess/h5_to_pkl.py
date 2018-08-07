import h5py
import numpy as np
<<<<<<< HEAD
import pickle 
=======
import pickle
import sys
>>>>>>> 1fbdb8b8865b71091d2c4e69d0e0e77d0f96f13c

h5_path='/storage/feature/voice_conversion/LibriSpeech/libri.h5'
pkl_path = '/storage/feature/voice_conversion/LibriSpeech/libri.pkl'

<<<<<<< HEAD
#speakers = ['226', '227', '228', '248', '251', '376']

dictionary = {}

with h5py.File(h5_path, 'r') as f_h5:
    for dset in ['train', 'test']:
        speakers = list(f_h5[f'{dset}'].keys())
        for speaker in speakers:
            print(f'processing speaker {speaker}')
            utt_list = list(f_h5[f'{dset}/{speaker}'].keys())
            # take 1/3
            for utt_id in utt_list:
                #dictionary[f'{dset}/{speaker}/{utt_id}/mel'] = f_h5[f'{dset}/{speaker}/{utt_id}/mel'][()]
                dictionary[f'{dset}/{speaker}/{utt_id}/lin'] = f_h5[f'{dset}/{speaker}/{utt_id}/lin'][()]

with open(pkl_path, 'wb') as f:
    pickle.dump(dictionary, f)
=======
dictionary = {}


def convert(h5_path, pkl_path):
    with h5py.File(h5_path, 'r') as f_h5:
        for dset in ['train', 'test']:
            speakers = list(f_h5[f'{dset}'].keys())
            for speaker in speakers:
                print(f'processing speaker {speaker}')
                utt_list = list(f_h5[f'{dset}/{speaker}'].keys())
                # take 1/3
                for utt_id in utt_list:
                    #dictionary[f'{dset}/{speaker}/{utt_id}/mel'] = f_h5[f'{dset}/{speaker}/{utt_id}/mel'][()]
                    dictionary[f'{dset}/{speaker}/{utt_id}/lin'] = f_h5[f'{dset}/{speaker}/{utt_id}/lin'][()]

    with open(pkl_path, 'wb') as f:
        pickle.dump(dictionary, f)

if __name__ == "__main__":
    h5_path = sys.argv[1]
    pkl_path = sys.argv[2]

    print("Converting.......")
    convert(h5_path, pkl_path)
    print("Done!!")



>>>>>>> 1fbdb8b8865b71091d2c4e69d0e0e77d0f96f13c
