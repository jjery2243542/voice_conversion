import h5py
import numpy as np
import sys

if __name__ == '__main__':
    in_h5py_path=sys.argv[1]
    out_h5py_path=sys.argv[2]

    with h5py.File(in_h5py_path, 'r+') as f_in, h5py.File(out_h5py_path, 'w') as f_out:
        for speaker in f_in['train'].keys():
            print(f'processing speaker_id={speaker}')
            # normalized f0
            utt_f0 = [f_in[f'train/{speaker}/{utt_id}/log_f0'][()] for utt_id in f_in[f'train/{speaker}'].keys()]
            all_f0 = np.concatenate(utt_f0, axis=0)
            index = np.where(all_f0 > 1e-10)[0]
            print(all_f0.shape, index.shape)
            f0_mean, f0_std = np.mean(all_f0[index]), np.std(all_f0[index])
            f_out.create_dataset(f'{speaker}/f0_mean', data=f0_mean, dtype=np.float32)
            f_out.create_dataset(f'{speaker}/f0_std', data=f0_std, dtype=np.float32)
            # processing MCEP
            utt_mc = [f_in[f'train/{speaker}/{utt_id}/mc'][()] for utt_id in f_in[f'train/{speaker}'].keys()]
            all_mc = np.concatenate(utt_mc, axis=0)
            print(all_mc.shape)
            mc_mean, mc_std = np.mean(all_mc, axis=0), np.std(all_mc, axis=0)
            print(mc_mean.shape, mc_std.shape)
            f_out.create_dataset(f'{speaker}/mc_mean', data=mc_mean, dtype=np.float32)
            f_out.create_dataset(f'{speaker}/mc_std', data=mc_std, dtype=np.float32)

        for speaker in f_in['train'].keys():
            print(f'normalize speaker_id={speaker}')
            mc_mean = f_out[f'{speaker}/mc_mean'][:] 
            mc_std = f_out[f'{speaker}/mc_std'][:]
            for dset in ['train', 'test']:
                for utt_id in f_in[f'{dset}/{speaker}']:
                    mc = f_in[f'{dset}/{speaker}/{utt_id}/mc'][:]
                    norm_mc = (mc - mc_mean) / mc_std
                    f_in.create_dataset(f'{dset}/{speaker}/{utt_id}/norm_mc', data=norm_mc, dtype=np.float32)
