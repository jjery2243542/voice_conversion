import h5py
import numpy as np
import sys
import glob

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: python3 make_dataset.py [h5py_path] [numpy_dir]')
        exit(0)
    h5py_path = sys.argv[1]
    np_dir = sys.argv[2]
    with h5py.File(h5py_path, 'w') as f_h5:
        for filename in sorted(glob.glob(os.path.join(np_dir, '*'))):
            print(filename)
        
