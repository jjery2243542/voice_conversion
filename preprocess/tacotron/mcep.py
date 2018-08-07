import librosa 
import pysptk
from pysptk.synthesis import LMADF, MLSADF, Synthesizer
from scipy.io.wavfile import write
import numpy as np
import copy 
import soundfile as sf
import pyworld as pw

def wav2mcep(wav_path, sr=16000, order=25, alpha=0.41):
    x, fs = librosa.load(wav_path, sr=sr, dtype=np.float64)
    #f0, sp, ap = pw.wav2world(x, fs)
    # 2-3 Harvest with F0 refinement (using Stonemask)
    _f0_h, t_h = pw.harvest(x, fs)
    f0_h = pw.stonemask(x, _f0_h, t_h, fs)
    sp_h = pw.cheaptrick(x, f0_h, t_h, fs)
    ap_h = pw.d4c(x, f0_h, t_h, fs)
    # convert to MCEP
    mc = pysptk.conversion.sp2mc(sp_h, order=order, alpha=alpha)
    return f0_h, ap_h, mc

def mc2wav(log_f0, src_f0_mean, src_f0_std, tar_f0_mean, tar_f0_std, ap, mc, mc_mean, mc_std, sr=16000):
    # mc reconstruction
    new_mc = mc * mc_std + mc_mean
    new_log_f0 = ((log_f0 - src_f0_mean) / src_f0_std) * tar_f0_std + tar_f0_mean
    new_f0 = np.exp(new_log_f0)
    sp = pysptk.conversion.mc2sp(new_mc, alpha=0.41, fftlen=1024)
    y = pw.synthesize(
            new_f0.astype(np.float64),
            sp.astype(np.float64),
            ap.astype(np.float64), 
            sr, 
            pw.default_frame_period)
    return y

if __name__ == '__main__':
    f0, ap, mc = wav2mcep('/media/arshsing/Storage/ML/_tensorflow3/VCTK-Corpus/wav48/p225/p225_236.wav')
    sp_new = pysptk.conversion.mc2sp(mc, alpha=0.41, fftlen=1024)
    log_f0 = np.log(f0 + 1e-10)
    src_f0_mean, src_f0_std = np.mean(log_f0), np.std(log_f0)
    f0, _, _ = wav2mcep('/media/arshsing/Storage/ML/_tensorflow3/VCTK-Corpus/wav48/p226/p226_236.wav')
    tar_log_f0 = np.log(f0 + 1e-10)
    tar_f0_mean, tar_f0_std = np.mean(tar_log_f0), np.std(tar_log_f0)
    new_log_f0 = ((log_f0 - src_f0_mean) / src_f0_std) * tar_f0_std + tar_f0_mean
    new_f0 = np.exp(new_log_f0)
    print(sp_new.shape)
    y_h = pw.synthesize(
            new_f0.astype(np.float64), 
            np.array(sp_new, dtype=np.float64), 
            ap.astype(np.float64), 
            16000, 
            pw.default_frame_period)
    sf.write('test.wav', y_h, 16000)
