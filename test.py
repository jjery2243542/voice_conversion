import torch
from torch import optim
from torch.autograd import Variable
import numpy as np
import pickle
from model import Encoder
from model import Discriminator
from utils import Hps
from utils import DataLoader
from utils import Logger
from utils import myDataset
from solver import Solver
from preprocess.tacotron.utils import spectrogram2wav
from scipy.io.wavfile import write

if __name__ == '__main__':
    hps = Hps()
    hps.load('./hps/v5.json')
    hps_tuple = hps.get_tuple()
    dataset = myDataset('/storage/raw_feature/voice_conversion/vctk/vctk.h5',\
            '/storage/raw_feature/voice_conversion/vctk/128_513_2000k.json')
    data_loader = DataLoader(dataset)
    solver = Solver(hps_tuple, data_loader)
    solver.load_model('/storage/model/voice_conversion/v6/model.pkl-59999')
    spec = np.loadtxt('./preprocess/test_code/lin.npy')
    spec2 = np.loadtxt('./preprocess/test_code/lin2.npy')
    spec_expand = np.expand_dims(spec, axis=0)
    spec_tensor = torch.from_numpy(spec_expand)
    spec_tensor = spec_tensor.type(torch.FloatTensor)
    spec2_expand = np.expand_dims(spec2, axis=0)
    spec2_tensor = torch.from_numpy(spec2_expand)
    spec2_tensor = spec2_tensor.type(torch.FloatTensor)
    c1 = Variable(torch.from_numpy(np.array([0]))).cuda()
    c2 = Variable(torch.from_numpy(np.array([4]))).cuda()
    result1 = solver.test_step(spec_tensor, c1)
    result1 = result1.squeeze(axis=0).transpose((1, 0))
    result2 = solver.test_step(spec2_tensor, c2)
    result2 = result2.squeeze(axis=0).transpose((1, 0))
    result3 = solver.test_step(spec2_tensor, c1)
    result3 = result3.squeeze(axis=0).transpose((1, 0))
    result4 = solver.test_step(spec_tensor, c2)
    result4 = result4.squeeze(axis=0).transpose((1, 0))
    wav_data = spectrogram2wav(spec)
    write('output.wav', rate=16000, data=wav_data)
    wav_data = spectrogram2wav(spec2)
    write('output2.wav', rate=16000, data=wav_data)
    wav_data = spectrogram2wav(result1)
    write('output3.wav', rate=16000, data=wav_data)
    wav_data = spectrogram2wav(result2)
    write('output4.wav', rate=16000, data=wav_data)
    wav_data = spectrogram2wav(result3)
    write('output5.wav', rate=16000, data=wav_data)
    wav_data = spectrogram2wav(result4)
    write('output6.wav', rate=16000, data=wav_data)


