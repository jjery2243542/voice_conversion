import sys
sys.path.append('/home/jjery2243542/research/vc/preprocess/tacotron/')
from another_repo.audio import load_wav, spectrogram, melspectrogram, inv_spectrogram, save_wav
from utils import get_spectrograms, spectrogram2wav
import numpy as np
from scipy.io.wavfile import write
import pysptk
import pyworld as pw
import librosa
import soundfile as sf

order=25
alpha=0.42
sr=16000
#wav = load_wav('/storage/datasets/VCTK/VCTK-Corpus/wav48/p225/p225_236.wav')
#S = spectrogram(wav).astype(np.float32)
#M = melspectrogram(wav).astype(np.float32)
#output = inv_spectrogram(S)
#save_wav(output, 'test.wav')
#x, fs = librosa.load('/storage/datasets/VCTK/VCTK-Corpus/wav48/p225/p225_366.wav', sr=sr, dtype=np.float64)
#f0, sp, ap = pw.wav2world(x, fs)
#_f0, t = pw.dio(x, fs, f0_floor=40.0, f0_ceil=1000.0,
#                channels_in_octave=2,
#                frame_period=1,
#                speed=1)
#_sp = pw.cheaptrick(x, _f0, t, fs)
#_ap = pw.d4c(x, _f0, t, fs)
#_y = pw.synthesize(_f0, _sp, _ap, fs, 1)
## harvest
#_f0_h, t_h = pw.harvest(x, fs)
#f0_h = pw.stonemask(x, _f0_h, t_h, fs)
#sp_h = pw.cheaptrick(x, f0_h, t_h, fs)
#ap_h = pw.d4c(x, f0_h, t_h, fs)
# mc code
#mc = pysptk.sp2mc(sp, order=order, alpha=alpha)
#sp_new = pysptk.mc2sp(mc, alpha=0.42, fftlen=1024)
#y = pw.synthesize(f0, sp, ap, fs, pw.default_frame_period)
# tacotron code
S, M = get_spectrograms('/storage/datasets/VCTK/VCTK-Corpus/wav48/p225/p225_236.wav')
print(M.shape)
#print(M.max())
#mc = pysptk.sp2mc(M, order=25, alpha=0.42)
#M_new = pysptk.mc2sp(mc, alpha=0.42, fftlen=1024)
#print(M_new.max())
wav_data = spectrogram2wav(M)
sf.write('test.wav', wav_data, 16000)
