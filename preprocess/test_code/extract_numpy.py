import h5py
import numpy as np

with h5py.File('/storage/feature/voice_conversion/vctk/trim_log_vctk.h5') as f_h5:
    lin_spec = f_h5['test/225/366/lin']
    lin_spec2 = f_h5['test/226/366/lin']
    np.savetxt('./lin.npy', lin_spec)
    np.savetxt('./lin2.npy', lin_spec2)
