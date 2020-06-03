#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 20:22:26 2020

@author: krishna
"""



import torch
import numpy as np
from torch.utils.data import DataLoader   
from SpeechDataGenerator import SpeechDataGenerator
import torch.nn as nn
import os
import numpy as np
from torch import optim
import argparse
from models.Emo_Raw_TDNN_StatPool import Emo_Raw_TDNN
from sklearn.metrics import accuracy_score
from utils.utils_wav import speech_collate
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')
import glob

########## Argument parser
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-training_filepath',type=str,default='meta/training_s1_s2_s3_s4.txt')
parser.add_argument('-testing_filepath',type=str, default='meta/testing_s5.txt')

parser.add_argument('-input_dim', action="store_true", default=1)
parser.add_argument('-num_classes', action="store_true", default=4)
parser.add_argument('-lamda_val', action="store_true", default=0.1)
parser.add_argument('-batch_size', action="store_true", default=64)
parser.add_argument('-use_gpu', action="store_true", default=True)
parser.add_argument('-num_epochs', action="store_true", default=100)
args = parser.parse_args()

### Data related

dataset_test = SpeechDataGenerator(manifest=args.testing_filepath,mode='test')
dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size,shuffle=True,collate_fn=speech_collate) 

## Model related
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Emo_Raw_TDNN(args.input_dim, args.num_classes).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0, betas=(0.9, 0.98), eps=1e-9)
loss_fun = nn.CrossEntropyLoss()



def test(dataloader_test):
    model.eval()
    with torch.no_grad():
        val_loss_list=[]
        full_preds=[]
        full_gts=[]
        for i_batch, sample_batched in enumerate(dataloader_test):
            features = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[0]])).float()
            labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]]))
            features, labels = features.to(device),labels.to(device)
            pred_logits = model(features)
            #### CE loss
            loss = loss_fun(pred_logits,labels)
            val_loss_list.append(loss.item())
            #train_acc_list.append(accuracy)
            predictions = np.argmax(pred_logits.detach().cpu().numpy(),axis=1)
            for pred in predictions:
                full_preds.append(pred)
            for lab in labels.detach().cpu().numpy():
                full_gts.append(lab)
                
        mean_acc = accuracy_score(full_gts,full_preds)
        mean_loss = np.mean(np.asarray(val_loss_list))
        #print('Total Test loss {} and Test accuracy {}'.format(mean_loss,mean_acc))
    return mean_acc
      
if __name__ == '__main__':
    all_models = sorted(glob.glob('save_model/*'))
    for model_path in all_models:
        model.load_state_dict(torch.load(model_path)['model'])
        acc = test(dataloader_test)
        print('Accuracy {} for model {}'.format(acc,model_path))
    
    
