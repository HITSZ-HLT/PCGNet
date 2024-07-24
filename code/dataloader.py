import pandas as pd
import pickle

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import re
import pandas as pd
import copy
import numpy as np
import csv

class IEMOCAPDataset(Dataset):
    def __init__(self, path=None, train=True, use_multiemo=False):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        _, _, self.roberta1, self.roberta2, self.roberta3, self.roberta4,\
        _, _, _, _ = pickle.load(open('data/iemocap/iemocap_features_roberta.pkl', 'rb'), encoding='latin1')
        
        ############################
        # prepare persona data
        self.persona = pickle.load(open("data/personaERC/IEMOCAP_persona.pkl", 'rb'), encoding='latin1')      
        ############################

        ## construct speaker infomation
        self.speakers = {}
        self.pattern = re.compile(r'(Ses\d{2}).*?([MF]\d{3})')
        for vid in self.videoIDs.keys():
            self.speakers[vid] = []
            for item in self.videoIDs[vid]:
                matches = self.pattern.search(item)
                if matches:
                    temp = matches.groups()
                    session = temp[0].replace("Ses", "")
                    # 偶数为male, 奇数为female
                    gender = 0 if 'M' in temp[1] else 1
                    self.speakers[vid].append((int(session)-1)*2 + gender)


        if use_multiemo:
            self.videoText = pickle.load(open('data/MultiEMO/IEMOCAP/TextFeatures.pkl', 'rb'))
            self.videoAudio = pickle.load(open('data/MultiEMO/IEMOCAP/AudioFeatures.pkl', 'rb'))
            self.videoVisual = pickle.load(open('data/MultiEMO/IEMOCAP/VisualFeatures.pkl', 'rb'))

            self.roberta1 = self.videoText
            self.roberta2 = self.videoText
            self.roberta3 = self.videoText
            self.roberta4 = self.videoText




    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.roberta1[vid]),\
            torch.FloatTensor(self.roberta2[vid]),\
            torch.FloatTensor(self.roberta3[vid]),\
            torch.FloatTensor(self.roberta4[vid]),\
            torch.FloatTensor(self.videoVisual[vid]),\
            torch.FloatTensor(self.videoAudio[vid]),\
            torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in self.videoSpeakers[vid]]),\
            torch.FloatTensor([1]*len(self.videoLabels[vid])),\
            torch.LongTensor(self.videoLabels[vid]),\
            torch.FloatTensor(self.persona[vid]),\
            self.speakers[vid],\
            vid

    def __len__(self):
        return len(self.keys)

    def collate_fn(self, data):
        dat = pd.DataFrame(data)  
        return [pad_sequence(dat[i]) if i<8 else pad_sequence(dat[i], True) if i<10 else dat[i].tolist() for i in dat]


class MELDDataset(Dataset):

    def __init__(self, path=None, train=True, shift=False, use_multiemo=False):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid,self.aaa = pickle.load(open(path, 'rb'),encoding='latin1')
        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.shift = shift
        self.trainOrigin = pd.read_csv("data/meld/train_sent_emo.csv").to_dict(orient='index')
        self.testOrigin = pd.read_csv("data/meld/test_sent_emo.csv").to_dict(orient='index')
        self.devOrigin = pd.read_csv("data/meld/dev_sent_emo.csv").to_dict(orient='index')
        self.originData = {}

        ############################
        # prepare persona data
        self.persona = pickle.load(open("data/personaERC/MELD_persona.pkl", 'rb'), encoding='latin1')
        ############################

        # to get speaker info 
        for _, value in self.trainOrigin.items():
            if value['Dialogue_ID'] not in self.originData.keys():
                self.originData[value['Dialogue_ID']] = []
            self.originData[value['Dialogue_ID']].append(value)

        for _, value in self.devOrigin.items():
            value['Dialogue_ID'] = value['Dialogue_ID'] + 1039
            if (value['Dialogue_ID']) not in self.originData.keys():
                self.originData[(value['Dialogue_ID'])] = []
            self.originData[(value['Dialogue_ID'])].append(value)
        for _, value in self.testOrigin.items():
            value['Dialogue_ID'] = value['Dialogue_ID'] + 1153
            if (value['Dialogue_ID']) not in self.originData.keys():
                self.originData[(value['Dialogue_ID'])] = []
            self.originData[(value['Dialogue_ID'])].append(value)
            
        self.speakers = []
        # 存储所有的speaker的信息，具体到每一个人
        self.speaker_list = {}
        for index in self.originData.keys():
            value = self.originData[index]
            cur_speaker = []
            for utterance in value:
                cur_speaker.append(utterance['Speaker'])
            self.speaker_list[index] = cur_speaker
            assert len(cur_speaker) == len(self.videoIDs[index])
            self.speakers.extend(cur_speaker)
        self.speakersIndex2Speaker = {i: speaker for i, speaker in enumerate(list(set(self.speakers)))}
        self.speakersSpeaker2Index = {v: k for k, v in self.speakersIndex2Speaker.items()}
        for key in self.keys:
            value = self.speaker_list[key]
            value_Index = []
            for item in value:
                value_Index.append(self.speakersSpeaker2Index[item])
            self.speaker_list[key] = value_Index
        self.len = len(self.keys)
        _, _, _, self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
            _, _, _, _ \
            = pickle.load(open("data/meld/meld_features_roberta.pkl", 'rb'), encoding='latin1')

        self.get_jsp_embedding()
        self.get_sp_jsp_list()


        if use_multiemo:
            self.videoText = pickle.load(open('data/MultiEMO/MELD/TextFeatures.pkl', 'rb'))
            self.videoAudio = pickle.load(open('data/MultiEMO/MELD/AudioFeatures.pkl', 'rb'))
            self.videoVisual = pickle.load(open('data/MultiEMO/MELD/VisualFeatures.pkl', 'rb'))
            self.roberta1 = self.videoText
            self.roberta2 = self.videoText
            self.roberta3 = self.videoText
            self.roberta4 = self.videoText


    def get_jsp_embedding(self):
        job_embedding = np.loadtxt(open('data/meld/job_embedding.txt'), delimiter=" ", skiprows=0, dtype=np.float32)
        sex_embedding = np.loadtxt(open('data/meld/sex_embedding.txt'), delimiter=" ", skiprows=0, dtype=np.float32)
        personality_embedding = np.loadtxt(open('data/meld/personality_embedding.txt'), delimiter=" ", skiprows=0, dtype=np.float32)
        self.job_embedding = torch.from_numpy(job_embedding)
        self.sex_embedding = torch.from_numpy(sex_embedding)
        self.personality_embedding = torch.from_numpy(personality_embedding)
        

    def get_jsp_list(self): # job, sex, personality
        job = []
        sex = []
        personality = []
        with open('data/meld/speaker_information.csv', 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            fieldnames = next(reader)
            csv_reader = csv.DictReader(f, fieldnames=fieldnames)
            for row in csv_reader:
                if row['job'] not in job:
                    job.append(row['job'])
                if row['sex'] not in sex:
                    sex.append(row['sex'])
                if row['personality'] not in personality:
                    personality.append(row['personality'])
        return job, sex, personality
    
    def get_sp_jsp_list(self):
        job, sex, personality = self.get_jsp_list()
        sp_jsp_list = {}
        with open('./data/meld/speaker_information.csv', 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            fieldnames = next(reader)
            csv_reader = csv.DictReader(f, fieldnames=fieldnames)
            for row in csv_reader:
                sp_jsp_list[self.speakersSpeaker2Index[row['name']]] = [job.index(row['job']), sex.index(row['sex']), personality.index(row['personality'])]
                # sp_jsp_list.append([job.index(row['job']), sex.index(row['sex']), personality.index(row['personality'])])
        self.sp_jsp_list = sp_jsp_list


    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.roberta1[vid]),\
            torch.FloatTensor(self.roberta2[vid]),\
            torch.FloatTensor(self.roberta3[vid]),\
            torch.FloatTensor(self.roberta4[vid]),\
            torch.FloatTensor(self.videoVisual[vid]),\
            torch.FloatTensor(self.videoAudio[vid]),\
            torch.FloatTensor(self.videoSpeakers[vid]),\
            torch.FloatTensor([1]*len(self.videoLabels[vid])),\
            torch.LongTensor(self.videoLabels[vid]),\
            torch.FloatTensor(self.persona[vid]), \
            self.speaker_list[vid],\
            (torch.stack([self.job_embedding[self.sp_jsp_list[speaker][0]] for speaker in self.speaker_list[vid]]), torch.stack([self.job_embedding[self.sp_jsp_list[speaker][1]] for speaker in self.speaker_list[vid]]), torch.stack([self.job_embedding[self.sp_jsp_list[speaker][2]] for speaker in self.speaker_list[vid]]) ),\
            vid

    def __len__(self):
        return self.len

    def return_labels(self):
        return_label = []
        for key in self.keys:
            return_label+=self.videoLabels[key]
        return return_label

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<8 else pad_sequence(dat[i], True) if i<10 else dat[i].tolist() for i in dat]


class DailyDialogueDataset(Dataset):

    def __init__(self, split, path):
        
        self.Speakers, self.Features, \
        self.ActLabels, self.EmotionLabels, self.trainId, self.testId, self.validId = pickle.load(open(path, 'rb'))
        
        if split == 'train':
            self.keys = [x for x in self.trainId]
        elif split == 'test':
            self.keys = [x for x in self.testId]
        elif split == 'valid':
            self.keys = [x for x in self.validId]

        self.len = len(self.keys)

    def __getitem__(self, index):
        conv = self.keys[index]
        
        return  torch.FloatTensor(self.Features[conv]), \
                torch.FloatTensor([[1,0] if x=='0' else [0,1] for x in self.Speakers[conv]]),\
                torch.FloatTensor([1]*len(self.EmotionLabels[conv])), \
                torch.LongTensor(self.EmotionLabels[conv]), \
                conv

    def __len__(self):
        return self.len
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<2 else pad_sequence(dat[i], True) if i<4 else dat[i].tolist() for i in dat]
