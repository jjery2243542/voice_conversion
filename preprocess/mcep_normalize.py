import h5py

if __name__ == '__main__':
    in_h5py_path=sys.argv[1]
    out_h5py_path=sys.argv[2]

    with h5py.File(in_h5py_path, 'r+') as f_in:
        for speaker in f_in['train'].keys():
            print(f'processing speaker_id={speaker}')
            # normalized f0
            utts_f0 = [f_in[f'train/{speaker}/{utt_id}/log_f0'][:] for utt_id in f_in[f'train/{speaker}'].keys()]
            all_f0 = np.concatenate(utt_f0, axis=0)
            f0_mean, f0_std = np.mean(all_f0), np.std(all_f0)
            print(f0_mean.shape, f0_std.shape)
            f_in.create_dataset(f'train/{speaker}/f0_mean', data=f0_mean, dtype=np.float32)
            f_in.create_dataset(f'train/{speaker}/f0_std', data=f0_std, dtype=np.float32)
            # processing MCEP
            utts_mc = [f_in[f'train/{speaker}/{utt_id}/mc'][:] for utt_id in f_in[f'train/{speaker}'].keys()]
            all_mc = np.concatenate(utt_mc, axis=0)
            mc_mean, mc_std = np.mean(all_mc), np.std(all_mc)
            print(mc_mean.shape, mc_std.shape)
            f_in.create_dataset(f'train/{speaker}/mc_mean', data=mc_mean, dtype=np.float32)
            f_in.create_dataset(f'train/{speaker}/mc_std', data=mc_std, dtype=np.float32)

        
