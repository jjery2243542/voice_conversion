import sys
sys.path.append('../')
from utils import Sampler
import h5py
import numpy as np

max_step=5
seg_len=128
batch_size=16
n_batches=100000

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: python3 make_batches.py [in_h5py_path] [out_h5py_path]')
        exit(0)
    sampler = Sampler(sys.argv[1], max_step=max_step, seg_len=seg_len)
    with h5py.File(sys.argv[2], 'w') as f_h5:
        for i in range(n_batches):
            samples = {
                'X_i_t':[{'mel':[], 'lin':[]}],
                'X_i_tk':[{'mel':[], 'lin':[]}],
                'X_i_tk_prime':[{'mel':[], 'lin':[]}],
                'X_j':[{'mel':[], 'lin':[]}],
            }
            for j in range(batch_size):
                sample = sampler.sample()
                samples['X_i_t']['mel'].append(sample[0])
                samples['X_i_t']['lin'].append(sample[1])
                samples['X_i_tk']['mel'].append(sample[2])
                samples['X_i_tk']['lin'].append(sample[3])
                samples['X_i_tk_prime']['mel'].append(sample[4])
                samples['X_i_tk_prime']['lin'].append(sample[5])
                samples['X_j']['mel'].append(sample[6])
                samples['X_j']['lin'].append(sample[7])
            for data_name in samples:
                for data_type in samples[data_name]:
                    f_h5.create_dataset(
                        '{}/{}/{}'.format(i, data_name, data_type),
                        data=np.array(samples[data_name][data_type], dtype=np.float32),
                        dtype=np.float32,
                    )
            if i % 10 == 0:
                print('process [{}/{}]'.format(i, n_batches))

