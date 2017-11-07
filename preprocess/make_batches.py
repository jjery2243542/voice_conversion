import sys
sys.path.append('../')
from utils import Sampler
import h5py
import numpy as np

max_step=5
seg_len=128
mel_band=80
lin_band=1025
batch_size=16
n_batches=50000

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: python3 make_batches.py [in_h5py_path] [out_h5py_path]')
        exit(0)
    sampler = Sampler(sys.argv[1], max_step=max_step, seg_len=seg_len)
    with h5py.File(sys.argv[2], 'w') as f_h5:
        for i in range(n_batches):
            samples = {
                'X_i_t':{
                    'mel':np.empty(shape=(batch_size, seg_len, mel_band), dtype=np.float32), 
                    'lin':np.empty(shape=(batch_size, seg_len, lin_band), dtype=np.float32)
                },
                'X_i_tk':{
                    'mel':np.empty(shape=(batch_size, seg_len, mel_band), dtype=np.float32), 
                    'lin':np.empty(shape=(batch_size, seg_len, lin_band), dtype=np.float32)
                },
                'X_i_tk_prime':{
                    'mel':np.empty(shape=(batch_size, seg_len, mel_band), dtype=np.float32), 
                    'lin':np.empty(shape=(batch_size, seg_len, lin_band), dtype=np.float32)
                },
                'X_j':{
                    'mel':np.empty(shape=(batch_size, seg_len, mel_band), dtype=np.float32), 
                    'lin':np.empty(shape=(batch_size, seg_len, lin_band), dtype=np.float32)
                },
            }
            for j in range(batch_size):
                sample = sampler.sample()
                samples['X_i_t']['mel'][j,:] = sample[0]
                samples['X_i_t']['lin'][j,:] = sample[1]
                samples['X_i_tk']['mel'][j,:] = sample[2]
                samples['X_i_tk']['lin'][j,:] = sample[3]
                samples['X_i_tk_prime']['mel'][j,:] = sample[4]
                samples['X_i_tk_prime']['lin'][j,:] = sample[5]
                samples['X_j']['mel'][j,:] = sample[6]
                samples['X_j']['lin'][j,:] = sample[7]

            for data_name in samples:
                for data_type in samples[data_name]:
                    data = samples[data_name][data_type]
                    f_h5.create_dataset(
                        '{}/{}/{}'.format(i, data_name, data_type),
                        data=data,
                        dtype=np.float32,
                    )
            if i % 5 == 0:
                print('process [{}/{}]'.format(i, n_batches))

