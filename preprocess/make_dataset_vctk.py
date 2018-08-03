import h5py
import numpy as np
import sys
import os
import glob
import re
from collections import defaultdict
#from tacotron.audio import load_wav, spectrogram, melspectrogram
from tacotron.norm_utils import get_spectrograms
from tacotron.mcep import wav2mcep
import json
import pickle
import random


'''
Samples N specfied speakers from the VCTK datset and then computes the linear spectogram inh5py fotmat.
Note: Assumes VCTK's directory structure as is when VCTK is downloaded from: http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html
'''

female_ids = []
male_ids = []
accents = defaultdict(list)

class Speaker(json.JSONEncoder):
    global female_ids, accents, male_ids
    def __init__(self,id,files,gender,accent):
        self.id = str(id)
        self.files = list([str(f) for f in files])
        self.gender = str(gender)
        self.accent = str(accent)
        accents[id] = self.accent
        if self.gender == 'M':
            male_ids.append(self.id)
        elif self.gender == 'F':
            female_ids.append(self.id)

    def default(self, o):
        d = { 'id' : self.id, 'files': 'f'}
        return json.loads(d)

def getFileList(dir_path, extension):
    return list(glob.glob(str(dir_path)+"/*."+str(extension)))

#root_dir='/media/arshsing/Storage/ML/_tensorflow3/VCTK-Corpus/wav48'
#train_split=0.9

def getSpeakerIdDict(speaker_info_txt_path):
    speakers_info = defaultdict(Speaker)
    speaker_id_by_gender = defaultdict(list)
    root_dir_parts = speaker_info_txt_path.strip().split('/')
    root_dir = '/'.join(root_dir_parts[1:-1]) + '/wav16/p'
    root_dir = '/' + root_dir
    with open(speaker_info_txt_path,'r') as f_info:
        lines = f_info.readlines()
        lines = lines[1:]
        for l in lines:
            id, age, gender, accent = l.strip().split()[:4]
            speakers_info[id] = Speaker(id,getFileList(root_dir+str(id),'wav'),gender,accent)
    speaker_id_by_gender['f'] = female_ids
    speaker_id_by_gender['m'] = male_ids
    speaker_id_by_gender["accents"] = accents
    return speakers_info, speaker_id_by_gender

def read_speaker_info(path='/media/arshsing/Storage/ML/_tensorflow3/VCTK-Corpus/speaker-info.txt'):
    accent2speaker = defaultdict(lambda: [])
    with open(path) as f:
        splited_lines = [line.strip().split() for line in f][1:]
        speakers = [line[0] for line in splited_lines]
        regions = [line[3] for line in splited_lines]
        for speaker, region in zip(speakers, regions):
            accent2speaker[region].append(speaker)
    return accent2speaker


def sample_speakerIds(female_ids,male_ids,N=20):
    fN = N//2
    mN = N//2
    f_ids = random.sample(female_ids,fN)
    m_ids = random.sample(male_ids,mN)
    speakers = f_ids + m_ids
    return speakers,f_ids,m_ids

root_dir='/storage/datasets/VCTK/VCTK-Corpus/wav48'
train_split=0.9
N_speakers = 2

def read_speaker_info(path='/media/arshsing/Storage/ML/_tensorflow3/VCTK-Corpus/speaker-info.txt'):
    accent2speaker = defaultdict(lambda: [])
    with open(path) as f:
        splited_lines = [line.strip().split() for line in f][1:]
        speakers = [line[0] for line in splited_lines]
        regions = [line[3] for line in splited_lines]
        for speaker, region in zip(speakers, regions):
            accent2speaker[region].append(speaker)
    return accent2speaker

'''
For future reference:
Female VCTK Speaker Ids = ['225', '228', '229', '230', '231', '233', '234', '236', '238', '239', '240', '244', '248', '249', '250', '253', '257', '261', '262', '264', '265', '266', '267', '268', '269', '276', '277', '282', '283', '288', '293', '294', '295', '297', '299', '300', '301', '303', '305', '306', '307', '308', '310', '312', '313', '314', '317', '318', '323', '329', '330', '333', '335', '336', '339', '340', '341', '343', '351', '361', '362']
Male VCTK Speaker Ids = ['226', '227', '232', '237', '241', '243', '245', '246', '247', '251', '252', '254', '255', '256', '258', '259', '260', '263', '270', '271', '272', '273', '274', '275', '278', '279', '281', '284', '285', '286', '287', '292', '298', '302', '304', '311', '315', '316', '326', '334', '345', '347', '360', '363', '364', '374', '376']
'''

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('usage: python3 make_dataset_vctk.py [N_speakers] [output h5py_path] [vctl_speaker_info_path] [VCTK wav48 wav files directory path]')
        exit(0)
    N_speakers = int(sys.argv[1])
    h5py_path=sys.argv[2]
    h5py_name = h5py_path.split('/')[-1].split('.')[0]
    vctk_speakerInfo_path = sys.argv[3]
    root_dir = sys.argv[4]
    #accent2speaker = read_speaker_info(vctk_speaker_path)

    speaker_info, speaker_id_by_gender = getSpeakerIdDict(vctk_speakerInfo_path)

    with open('speaker_info.pkl', 'wb') as outfile:
        pickle.dump(speaker_info, outfile)
        x = pickle.load(open('speaker_info.pkl', 'rb'))
        print(x)

    with open('speaker_id_by_gender.json', 'w') as outfile:
        json.dump(speaker_id_by_gender, outfile)

    filename_groups = defaultdict(lambda : [])
    speaker_list, females, males = sample_speakerIds(female_ids, male_ids, N_speakers)
    print(f'Using randomnly sampled ids:{speaker_list}\nFemales:{females},\nMales{males}')
    with h5py.File(h5py_path, 'w') as f_h5:
        filenames = sorted(glob.glob(os.path.join(root_dir, '*/*.wav')))
        for filename in filenames:
            # divide into groups
            sub_filename = filename.strip().split('/')[-1]
            # format: p{speaker}_{sid}.wav
            speaker_id, utt_id = re.match(r'p(\d+)_(\d+)\.wav', sub_filename).groups()
            filename_groups[speaker_id].append(filename)
        for speaker_id, filenames in filename_groups.items():
            if speaker_id not in speaker_list:
                continue
            print('processing {}'.format(speaker_id))
            print(f'Using randomly sampled ids:{speaker_list}\nFemales:{females},\nMales{males}')
            train_size = int(len(filenames) * train_split)
            for i, filename in enumerate(filenames):
                print(filename)
                sub_filename = filename.strip().split('/')[-1]
                # format: p{speaker}_{sid}.wav
                speaker_id, utt_id = re.match(r'p(\d+)_(\d+)\.wav', sub_filename).groups()
                #wav_data = load_wav(filename)
                #lin_spec = spectrogram(wav_data).astype(np.float32).T
                #mel_spec = melspectrogram(wav_data).astype(np.float32).T
                mel_spec, lin_spec = get_spectrograms(filename)
                f0, ap, mc = wav2mcep(filename)
                #eps = 1e-10
                #log_mel_spec, log_lin_spec = np.log(mel_spec+eps), np.log(lin_spec+eps)
                if i < train_size:
                    datatype = 'train'
                else:
                    datatype = 'test'
                f_h5.create_dataset(f'{datatype}/{speaker_id}/{utt_id}/mel', \
                    data=mel_spec, dtype=np.float32)
                f_h5.create_dataset(f'{datatype}/{speaker_id}/{utt_id}/lin', \
                    data=lin_spec, dtype=np.float32)
    with open(h5py_name+'_speakers_used.txt','w') as su_f:
        for f_id in females:
            su_f.write(f_id + "F\n")
        for m_id in males:
            su_f.write(m_id + "M\n")
