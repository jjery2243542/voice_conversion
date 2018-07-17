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
from preprocess.tacotron.norm_utils import spectrogram2wav, get_spectrograms
#from preprocess.tacotron.audio import inv_spectrogram, save_wav
from scipy.io.wavfile import write
#from preprocess.tacotron.mcep import mc2wav
import glob
import os

if __name__ == '__main__':
    feature = 'sp'
    hps = Hps()
    hps.load('./hps/vctk.json')
    hps_tuple = hps.get_tuple()
    solver = Solver(hps_tuple, None)
    solver.load_model('/storage/model/voice_conversion/vctk/clf/norm/8_speakers_1e-3.pkl-149999')
    if feature == 'mc':
        # indexer to extract data
        indexer = Indexer()
        src_mc = indexer.index(speaker_id='225', utt_id='366', dset='test', feature='norm_mc')
        tar_mc = indexer.index(speaker_id='226', utt_id='366', dset='test', feature='norm_mc')
        expand_src_mc = np.expand_dims(src_mc, axis=0)
        expand_tar_mc = np.expand_dims(tar_mc, axis=0)
        src_mc_tensor = torch.from_numpy(expand_src_mc).type(torch.FloatTensor)
        tar_mc_tensor = torch.from_numpy(expand_tar_mc).type(torch.FloatTensor)
        c1 = Variable(torch.from_numpy(np.array([0]))).cuda()
        c2 = Variable(torch.from_numpy(np.array([1]))).cuda()
        results = [src_mc]
        result = solver.test_step(src_mc_tensor, c1)
        result = result.squeeze(axis=0).transpose((1, 0))
        results.append(result)
        result = solver.test_step(src_mc_tensor, c2)
        result = result.squeeze(axis=0).transpose((1, 0))
        results.append(result)
        results.append(tar_mc)
        result = solver.test_step(tar_mc_tensor, c2)
        result = result.squeeze(axis=0).transpose((1, 0))
        results.append(result)
        result = solver.test_step(tar_mc_tensor, c1)
        result = result.squeeze(axis=0).transpose((1, 0))
        results.append(result)
        print(np.mean((result[-1] - result[-3])**2), np.mean((result[-2] - result[-3])**2))
        speaker_pairs = [('225', '225'), ('225', '225'), ('225', '225'), ('226', '226'), ('226', '226'), ('226', '226')]
        for i, (result, (src_speaker, tar_speaker)) in enumerate(zip(results, speaker_pairs)):
            src_f0_mean, src_f0_std = indexer.get_mean_std(src_speaker, 'f0')
            tar_f0_mean, tar_f0_std = indexer.get_mean_std(tar_speaker, 'f0')
            mc_mean, mc_std = indexer.get_mean_std(tar_speaker, 'mc')
            ap = indexer.index(speaker_id=src_speaker, utt_id='366', dset='test', feature='ap')
            log_f0 = indexer.index(speaker_id=src_speaker, utt_id='366', dset='test', feature='log_f0')
            truncated_result = result[:ap.shape[0]]
            wav_data = mc2wav(log_f0, src_f0_mean, src_f0_std, tar_f0_mean, tar_f0_std, ap, truncated_result, mc_mean, mc_std)
            write(f'output{i+1}.wav', rate=16000, data=wav_data)
    else:
        one = False
        if one == True:
            filename = '/storage/datasets/teacher_voice/data/lin_shan_lee/wav/20060530-1-002380.wav' 
            #spec = np.loadtxt(filename)
            _, spec = get_spectrograms(filename)
            spec_expand = np.expand_dims(spec, axis=0)
            spec_tensor = torch.from_numpy(spec_expand).type(torch.FloatTensor)
            c = Variable(torch.from_numpy(np.array([1]))).cuda()
            result = solver.test_step(spec_tensor, c, gen=True)
            result = result.squeeze(axis=0).transpose((1, 0))
            print(result.shape)
            wav_data = spectrogram2wav(result)
            write('result.wav', rate=16000, data=wav_data)
        else:
            directory = './for_VC/hy/'
            for filename in glob.glob(os.path.join(directory, '*.wav')):
                _, sub_filename = filename.rsplit('/', maxsplit=1)
                _, spec = get_spectrograms(filename)
                spec_expand = np.expand_dims(spec, axis=0)
                spec_tensor = torch.from_numpy(spec_expand).type(torch.FloatTensor)
                c = Variable(torch.from_numpy(np.array([1]))).cuda()
                if spec_tensor.size(1) > 1000:
                    result_list = []
                    for small_spec_tensor in spec_tensor.split(split_size=400, dim=1):
                        print(small_spec_tensor.size())
                        if small_spec_tensor.size(1) >= 10:
                            result = solver.test_step(small_spec_tensor, c, gen=True)
                            result = result.squeeze(axis=0).transpose((1, 0))
                            result_list.append(result)
                    result = np.concatenate(result_list, axis=0)
                    print(result.shape)
                    wav_data = spectrogram2wav(result)
                else:
                    result = solver.test_step(spec_tensor, c, gen=True)
                    result = result.squeeze(axis=0).transpose((1, 0))
                    print(result.shape)
                    wav_data = spectrogram2wav(result)
                write(os.path.join('./for_VC/hy/result/', sub_filename), rate=16000, data=wav_data)
        #result = solver.test_step(spec_tensor, c1, gen=True)
        #spec = np.loadtxt('preprocess/test_code/vcc/lin.npy')
        #spec2 = np.loadtxt('preprocess/test_code/vcc/lin2.npy')
        #spec_expand = np.expand_dims(spec, axis=0)
        #spec_tensor = torch.from_numpy(spec_expand)
        #spec_tensor = spec_tensor.type(torch.FloatTensor)
        #spec2_expand = np.expand_dims(spec2, axis=0)
        #spec2_tensor = torch.from_numpy(spec2_expand)
        #spec2_tensor = spec2_tensor.type(torch.FloatTensor)
        #c1 = Variable(torch.from_numpy(np.array([0]))).cuda()
        #c2 = Variable(torch.from_numpy(np.array([7]))).cuda()
        #c3 = Variable(torch.from_numpy(np.array([5]))).cuda()
        #c4 = Variable(torch.from_numpy(np.array([9]))).cuda()
        #results = [spec, spec2]
        #result = solver.test_step(spec_tensor, c1, gen=True)
        #result = result.squeeze(axis=0).transpose((1, 0))
        #result = np.concatenate([result, np.ones((100, 513), dtype=np.float32) * -10])
        #results.append(result)
        #result = solver.test_step(spec2_tensor, c2, gen=True)
        #result = result.squeeze(axis=0).transpose((1, 0))
        #results.append(result)
        #result = solver.test_step(spec2_tensor, c1, gen=True)
        #result = result.squeeze(axis=0).transpose((1, 0))
        #results.append(result)
        #result = solver.test_step(spec_tensor, c2, gen=True)
        #result = result.squeeze(axis=0).transpose((1, 0))
        #results.append(result)
        #result = solver.test_step(spec_tensor, c3, gen=True)
        #result = result.squeeze(axis=0).transpose((1, 0))
        #results.append(result)
        #result = solver.test_step(spec_tensor, c4, gen=True)
        #result = result.squeeze(axis=0).transpose((1, 0))
        #results.append(result)
        #for i, result in enumerate(results):
        #    result = np.power(np.e, result)**1.2
        #    wav_data = spectrogram2wav(result)
        #    write(f'output{i+1}.wav', rate=16000, data=wav_data)
