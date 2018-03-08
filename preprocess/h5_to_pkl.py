import h5py
import numpy as np
import pickle 

#h5_path='/storage/feature/voice_conversion/vcc/trim_norm_mc.h5'
#pkl_path = '/storage/feature/voice_conversion/vcc/trim_norm_mc.pkl'
#speakers = ['SF1', 'SF2', 'SF3', 'SM1', 'SM2', 'TF1', 'TF2', 'TM1', 'TM2', 'TM3']

def read_speaker(speaker_file='/storage/datasets/VCTK/VCTK-Corpus/speaker-info.txt'):
    with open(speaker_file, 'r') as f:
        # no header
        lines = [line for line in f][1:]
        line_splited = [line.split() for line in lines]
        f_en_ids = [line.strip().split()[0] for line in lines if line.strip().split()[3] == 'English' and line.strip().split()[2] == 'F'][:10]
        m_en_ids = [line.strip().split()[0] for line in lines if line.strip().split()[3] == 'English' and line.strip().split()[2] == 'M'][:10]
        return f_en_ids + m_en_ids

h5_path='/storage/feature/voice_conversion/vctk/trim_log_add_one_vctk.h5'
pkl_path = '/storage/feature/voice_conversion/vctk/trim_log_add_one_vctk.pkl'
speaker_used_path = '/storage/feature/voice_conversion/vctk/en_speaker_used.txt'
speakers = read_speaker()
dictionary = {}

with h5py.File(h5_path, 'r') as f_h5:
    for dset in ['train', 'test']:
        for speaker in speakers:
            print(f'processing speaker-{speaker}')
            for utt_id in f_h5[f'{dset}/{speaker}']:
                dictionary[f'{dset}/{speaker}/{utt_id}/lin'] = f_h5[f'{dset}/{speaker}/{utt_id}/lin'][()]

#with open(speaker_used_path, 'w') as f:
#    for speaker in speakers:
#        f.write(f'{speaker}\n')

with open(pkl_path, 'wb') as f:
    pickle.dump(dictionary, f)
