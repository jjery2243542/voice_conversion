from utils import ispecgram
from scipy.io import wavfile
import h5py
import numpy as np

rate=20000

def specs2wav(specs, wav_filename):
    '''
    input: spectrograms, shape=[n_segments, 257, 64, 1]
    '''
    specs = specs.transpose([1, 0, 2, 3])
    specs = specs.reshape([257, -1, 1])
    print(specs.shape)
    wav_arr = ispecgram(specs)
    wavfile.write(wav_filename, rate=rate, data=wav_arr)

if __name__ == '__main__':
    with h5py.File('/storage/raw_feature/voice_conversion/libre_equal.h5') as f_h5:
        specs = f_h5['test/911/130578-0020'][:]
        specs = specs.transpose([0, 2, 3, 1])[:,:,:,:1]
        specs2wav(specs, 'test.wav')
