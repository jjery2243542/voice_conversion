import torch
from torch import optim
from torch.autograd import Variable
import numpy as np
import pickle
from utils import Hps
from utils import DataLoader
from utils import Logger
from utils import myDataset
from utils import Indexer
from solver import Solver
from preprocess.tacotron.norm_utils import spectrogram2wav
#from preprocess.tacotron.audio import inv_spectrogram, save_wav
from scipy.io.wavfile import write
#from preprocess.tacotron.mcep import mc2wav
import h5py 
import os 
import soundfile as sf
#import pysptk
#import pyworld as pw

def sp2wav(sp): 
    #exp_sp = np.exp(sp)
    exp_sp = sp
    wav_data = spectrogram2wav(exp_sp)
    return wav_data

def get_world_param(f_h5, src_speaker, utt_id, tar_speaker, tar_speaker_id, solver, dset='test', gen=True):
    mc = f_h5[f'{dset}/{src_speaker}/{utt_id}/norm_mc'][()]
    converted_mc = convert_mc(mc, tar_speaker_id, solver, gen=gen)
    #converted_mc = mc
    mc_mean = f_h5[f'train/{tar_speaker}'].attrs['mc_mean']
    mc_std = f_h5[f'train/{tar_speaker}'].attrs['mc_std']
    converted_mc = converted_mc * mc_std + mc_mean
    log_f0 = f_h5[f'{dset}/{src_speaker}/{utt_id}/log_f0'][()]
    src_mean = f_h5[f'train/{src_speaker}'].attrs['f0_mean']
    src_std = f_h5[f'train/{src_speaker}'].attrs['f0_std']
    tar_mean = f_h5[f'train/{tar_speaker}'].attrs['f0_mean']
    tar_std = f_h5[f'train/{tar_speaker}'].attrs['f0_std']
    index = np.where(log_f0 > 1e-10)[0]
    log_f0[index] = (log_f0[index] - src_mean) * tar_std / src_std + tar_mean
    log_f0[index] = np.exp(log_f0[index])
    f0 = log_f0
    ap = f_h5[f'{dset}/{src_speaker}/{utt_id}/ap'][()]
    converted_mc = converted_mc[:ap.shape[0]]
    sp = pysptk.conversion.mc2sp(converted_mc, alpha=0.41, fftlen=1024)
    return f0, sp, ap

def synthesis(f0, sp, ap, sr=16000):
    y = pw.synthesize(
            f0.astype(np.float64),
            sp.astype(np.float64),
            ap.astype(np.float64), 
            sr, 
            pw.default_frame_period)
    return y

def convert_sp(sp, c, solver, gen=True):
    c_var = Variable(torch.from_numpy(np.array([c]))).cuda()
    sp_tensor = torch.from_numpy(np.expand_dims(sp, axis=0))
    sp_tensor = sp_tensor.type(torch.FloatTensor)
    converted_sp = solver.test_step(sp_tensor, c_var, gen=gen)
    converted_sp = converted_sp.squeeze(axis=0).transpose((1, 0))
    return converted_sp

def convert_mc(mc, c, solver, gen=True):
    c_var = Variable(torch.from_numpy(np.array([c]))).cuda()
    mc_tensor = torch.from_numpy(np.expand_dims(mc, axis=0))
    mc_tensor = mc_tensor.type(torch.FloatTensor)
    converted_mc = solver.test_step(mc_tensor, c_var, gen=gen)
    converted_mc = converted_mc.squeeze(axis=0).transpose((1, 0))
    return converted_mc

def get_model(hps_path='./hps/vctk.json', model_path='/storage/model/voice_conversion/vctk/clf/model.pkl-109999'):
    hps = Hps()
    hps.load(hps_path)
    hps_tuple = hps.get_tuple()
    solver = Solver(hps_tuple, None)
    solver.load_model(model_path)
    return solver

def convert_all_sp(h5_path, src_speaker, tar_speaker, gen=True, 
        dset='test', speaker_used_path='/storage/feature/voice_conversion/vctk/dataset_used/en_speaker_used.txt',
        root_dir='/storage/result/voice_conversion/vctk/p226_to_p225/',
        model_path='/storage/model/voice_conversion/vctk/clf/wo_tanh/model_0.001.pkl-79999'):
    # read speaker id file
    with open(speaker_used_path) as f:
        speakers = [line.strip() for line in f]
        speaker2id = {speaker:i for i, speaker in enumerate(speakers)}
    solver = get_model(hps_path='hps/vctk.json', 
            model_path=model_path)
    with h5py.File(h5_path, 'r') as f_h5:
        for utt_id in f_h5[f'{dset}/{src_speaker}']:
            sp = f_h5[f'{dset}/{src_speaker}/{utt_id}/lin'][()]
            converted_sp = convert_sp(sp, speaker2id[tar_speaker], solver, gen=gen)
            wav_data = sp2wav(converted_sp)
            wav_path = os.path.join(root_dir, f'{src_speaker}_{tar_speaker}_{utt_id}.wav')
            sf.write(wav_path, wav_data, 16000, 'PCM_24')

def convert_all_mc(h5_path, src_speaker, tar_speaker, gen=False, 
        dset='test', speaker_used_path='/storage/feature/voice_conversion/vctk/mcep/en_speaker_used.txt',
        root_dir='/storage/result/voice_conversion/vctk/p226_to_p225',
        model_path='/storage/model/voice_conversion/vctk/clf/wo_tanh/model_0.001.pkl-79999'):
    # read speaker id file
    with open(speaker_used_path) as f:
        speakers = [line.strip() for line in f]
        speaker2id = {speaker:i for i, speaker in enumerate(speakers)}
    solver = get_model(hps_path='hps/vctk.json', 
            model_path=model_path)
    with h5py.File(h5_path, 'r') as f_h5:
        for utt_id in f_h5[f'{dset}/{src_speaker}']:
            f0, sp, ap = get_world_param(f_h5, src_speaker, utt_id, tar_speaker, tar_speaker_id=speaker2id[tar_speaker], solver=solver, dset='test', gen=gen)
            wav_data = synthesis(f0, sp, ap)
            wav_path = os.path.join(root_dir, f'{src_speaker}_{tar_speaker}_{utt_id}.wav')
            sf.write(wav_path, wav_data, 16000, 'PCM_24')

if __name__ == '__main__':
    h5_path = '/storage/feature/voice_conversion/vctk/dataset_used/norm_vctk.h5'
    root_dir = '/storage/result/voice_conversion/vctk/norm/clf_gen'
    #h5_path = '/storage/feature/voice_conversion/LibriSpeech/libri.h5'
    #h5_path = '/storage/feature/voice_conversion/vctk/mcep/trim_mc_vctk_backup.h5'
    #convert_all_mc(h5_path, '226', '225', root_dir='./test_mc/', gen=False, 
    #        model_path='/storage/model/voice_conversion/vctk/mcep/clf/model.pkl-129999')
    #convert_all_mc(h5_path, '225', '226', root_dir='./test_mc/', gen=False, 
    #        model_path='/storage/model/voice_conversion/vctk/mcep/clf/model.pkl-129999')
    #convert_all_mc(h5_path, '225', '228', root_dir='./test_mc/', gen=False, 
    #        model_path='/storage/model/voice_conversion/vctk/mcep/clf/model.pkl-129999')
    #model_path = '/storage/model/voice_conversion/vctk/mcep/clf/model.pkl-129999'
    model_path = '/storage/model/voice_conversion/vctk/clf/norm/wo_tanh/model_0.001_no_ins.pkl-124000'
    #model_path = '/storage/model/voice_conversion/librispeech/ls_1e-3.pkl-99999'
    speakers = ['225', '226', '227']
    for speaker_A in speakers:
        for speaker_B in speakers:
            if speaker_A == speaker_B:
                continue
            else:
                dir_path = os.path.join(root_dir, f'p{speaker_A}_p{speaker_B}')
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                convert_all_sp(h5_path,speaker_A,speaker_B,
                        root_dir=dir_path, 
                        gen=True, model_path=model_path)
    # diff accent
    #dir_path = os.path.join(root_dir, '163_6925')
    #if not os.path.exists(dir_path):
    #    os.makedirs(dir_path)
    #convert_all_sp(h5_path,'163','6925',root_dir=dir_path, 
    #        gen=False, model_path=model_path)
    #dir_path = os.path.join(root_dir, '460_1363')
    #if not os.path.exists(dir_path):
    #    os.makedirs(dir_path)
    #convert_all_sp(h5_path,'460','1363',root_dir=dir_path, 
    #        gen=False, model_path=model_path)
    #convert_all_sp(h5_path,'363','256',root_dir=os.path.join(root_dir, 'p363_p256'), 
    #        gen=True, model_path=model_path)
    #convert_all_sp(h5_path,'340','251',root_dir=os.path.join(root_dir, 'p340_p251'), 
    #        gen=True, model_path=model_path)
    #convert_all_sp(h5_path,'285','251',root_dir=os.path.join(root_dir, 'p285_p251'), 
    #        gen=True, model_path=model_path)
