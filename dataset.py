#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:05:38 2020

@author: krishna
"""

import os
import numpy as np
import glob
import pickle
import argparse
#### Import pickle file


emotion_id = {'hap':0,'ang':1,'sad':2,'neu':3}
gender_id = {'M':0,'F':1}

class IEMOCAP(object):
    def __init__(self,config):
        super(IEMOCAP, self).__init__()
        self.pickle_path = config.pickle_filepath
        self.dataset_path = config.dataset_root
        self.store_meta = config.store_meta

    def create_dict(self):
        f = open(self.pickle_path, 'rb')
        p_data = pickle.load(f,encoding="latin1")   
        label_dict = {}
        for row in p_data:
            filename = row['id']
            if row['emotion']=='exc':
                label_dict[filename] = 'hap'
            else:
                label_dict[filename] = row['emotion']
        return label_dict
    
    def sess_map(self,session_list):
        final_sess_list=[]
        for sess in session_list:
            if sess=='s1':
                final_sess_list.append('Ses01')
            elif sess =='s2':
                final_sess_list.append('Ses02')
            elif sess =='s3':
                final_sess_list.append('Ses03')
            elif sess =='s4':
                final_sess_list.append('Ses04')
            elif sess =='s5':
                final_sess_list.append('Ses05')
            else:
                print('Wrong session id')
        return final_sess_list
    
    
    def create_meta_train(self,session_list):
        label_dict = self.create_dict()
        filename = 'training_'+'_'.join(session_list)+'.txt'
        filepath = os.path.join(self.store_meta,filename)
        fid = open(filepath,'w')
        all_audio_files = sorted(glob.glob(self.dataset_path+'/*.wav'))
        final_session_list = self.sess_map(session_list)
        for filepath in all_audio_files:
            filename =  filepath.split('/')[-1]
            check_name = filename.split('_')[-1]
            if check_name=='noise.wav':
                continue
            check_sv = filename.split('_')[-1]
            if (check_sv[0]=='s' or check_sv[0]=='v'):
                check_filename = '_'.join(filepath.split('/')[-1].split('_')[:-1])
                emotion=label_dict[check_filename]
                gender = check_filename.split('_')[0][-1]
                to_write = filepath+' '+str(emotion_id[emotion])+' '+str(gender_id[gender])
                check_session = check_filename.split('_')[0][:-1]
                if check_session in final_session_list:
                    fid.write(to_write+'\n')
                print(emotion,check_filename,gender)
            else:
                check_filename=filepath.split('/')[-1][:-4]
                emotion = label_dict[check_filename]
                gender = check_filename.split('_')[0][-1]
                to_write = filepath+' '+str(emotion_id[emotion])+' '+str(gender_id[gender])
                check_session = check_filename.split('_')[0][:-1]
                if check_session in final_session_list:
                    fid.write(to_write+'\n')
                print(emotion,check_filename,gender)
        fid.close()
        
        
        
    def create_meta_test(self,session_list):
        label_dict = self.create_dict()
        filename = 'testing_'+'_'.join(session_list)+'.txt'
        filepath = os.path.join(self.store_meta,filename)
        fid = open(filepath,'w')
        all_audio_files = sorted(glob.glob(self.dataset_path+'/*.wav'))
        final_session_list = self.sess_map(session_list)
        for filepath in all_audio_files:
            filename =  filepath.split('/')[-1]
            check_name = filename.split('_')[-1]
            if check_name=='noise.wav':
                continue
            check_sv = filename.split('_')[-1]
            if (check_sv[0]=='s' or check_sv[0]=='v'):
                continue
            else:
                check_filename=filepath.split('/')[-1][:-4]
                emotion = label_dict[check_filename]
                gender = check_filename.split('_')[0][-1]
                to_write = filepath+' '+str(emotion_id[emotion])+' '+str(gender_id[gender])
                check_session = check_filename.split('_')[0][:-1]
                if check_session in final_session_list:
                    fid.write(to_write+'\n')
                print(emotion,check_filename,gender)
        fid.close()
        
        
        
                
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Configuration for data preparation")
    parser.add_argument("--pickle_filepath", default="/media/newhd/IEMOCAP_dataset/data_collected_full.pickle", type=str,help='Dataset path')
    parser.add_argument("--dataset_root", default="/media/newhd/IEMOCAP_dataset/raw_data", type=str,help='Save directory after processing')
    parser.add_argument("--store_meta", default="meta/", type=str,help='CMU pronounciation directory path')

    config = parser.parse_args()
    dataset = IEMOCAP(config)
    ### 5 folder cross validation
    train_list=['s1','s2','s3','s4']
    test_list =['s5']
    dataset.create_meta_train(train_list)
    dataset.create_meta_test(test_list)
    
    #######
    train_list=['s2','s3','s4','s5']
    test_list =['s1']
    dataset.create_meta_train(train_list)
    dataset.create_meta_test(test_list)
    
    ########
    train_list=['s1','s3','s4','s5']
    test_list =['s2']
    dataset.create_meta_train(train_list)
    dataset.create_meta_test(test_list)
    
    ########
    train_list=['s1','s2','s4','s5']
    test_list =['s3']
    dataset.create_meta_train(train_list)
    dataset.create_meta_test(test_list)
    
    ######
    train_list=['s1','s2','s3','s5']
    test_list =['s4']
    dataset.create_meta_train(train_list)
    dataset.create_meta_test(test_list)


