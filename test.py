import torch
from torch import optim
from torch.autograd import Variable
import numpy as np
import pickle
from utils import Hps
from utils import DataLoader
from utils import Logger
from utils import myDataset
from solver import Solver
from preprocess.tacotron.utils import spectrogram2wav
#from preprocess.tacotron.audio import inv_spectrogram, save_wav
from scipy.io.wavfile import write

if __name__ == '__main__':
    hps = Hps()
    hps.load('./hps/v18.json')
    hps_tuple = hps.get_tuple()
    dataset = myDataset('/storage/feature/voice_conversion/vctk/log_vctk.h5',\
            '/storage/feature/voice_conversion/vctk/128_513_2000k.json')
    data_loader = DataLoader(dataset)
    solver = Solver(hps_tuple, data_loader)
    solver.load_model('/storage/model/voice_conversion/v18/model.pkl-44999')
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
    results = [spec, spec2]
    result = solver.test_step(spec_tensor, c1)
    result = result.squeeze(axis=0).transpose((1, 0))
    results.append(result)
    result = solver.test_step(spec2_tensor, c2)
    result = result.squeeze(axis=0).transpose((1, 0))
    results.append(result)
    result = solver.test_step(spec2_tensor, c1)
    result = result.squeeze(axis=0).transpose((1, 0))
    results.append(result)
    result = solver.test_step(spec_tensor, c2)
    result = result.squeeze(axis=0).transpose((1, 0))
    results.append(result)
    for i, result in enumerate(results):
        result = np.power(np.e, result)**1.2
        wav_data = spectrogram2wav(result)
        write(f'output{i+1}.wav', rate=16000, data=wav_data)


