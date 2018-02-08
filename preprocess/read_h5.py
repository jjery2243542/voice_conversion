import h5py 

with h5py.File('/storage/feature/voice_conversion/vctk/log_vctk.h5') as f_h5:
    for dset in f_h5:
        for speaker in f_h5[dset]:
            for utt in f_h5[f'{dset}/{speaker}']:
                print(f_h5[f'{dset}/{speaker}/{utt}/lin'][:].min(), f_h5[f'{dset}/{speaker}/{utt}/lin'][:].max())
