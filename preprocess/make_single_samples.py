import sys
sys.path.append('../')
from utils import Sampler
import h5py
import numpy as np
import json 

max_step=5
seg_len=128
mel_band=80
lin_band=513
n_samples=2000000
dset='train'

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: python3 make_single_samples.py [in_h5py_path] [out_json_path]')
        exit(0)
    sampler = Sampler(sys.argv[1], max_step=max_step, seg_len=seg_len, dset=dset)
    samples = [sampler.sample_single()._asdict() for _ in range(n_samples)]
    with open(sys.argv[2], 'w') as f_json:
        json.dump(samples, f_json, indent=4, separators=(',', ': '))
