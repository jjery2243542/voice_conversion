import h5py

#with h5py.File('/storage/raw_feature/voice_conversion/tacotron_feature/train-clean-100.h5') as f_h5:
#    print(f_h5['train/911/130578-0019/mel'].shape)
#    print(list(f_h5.keys()))
cnt = 0
total = 0
print('speaker utterence length')
with h5py.File('/storage/feature/voice_conversion/vctk/log_vctk.h5') as f_h5:
    for dset in ['train', 'test']:
        for speaker in f_h5[dset]:
            for utt in f_h5[f'{dset}/{speaker}']:
                length = f_h5[f'{dset}/{speaker}/{utt}/lin'].shape[0]
                print(f'{speaker} {utt} {length}')
#print(cnt / total)
