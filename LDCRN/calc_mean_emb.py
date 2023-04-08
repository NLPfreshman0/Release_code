from config import parse_args
from model import Model
from load_datasets import create_dataloaders, create_hard_dataloaders, create_MultiNLI_hard_dataloaders
import logging
import os
import time
import torch
from tqdm import tqdm
from optimizer import build_optimizer
import scipy.stats
import random
import numpy as np
from transformers import AutoTokenizer
import pandas as pd

def setup_device(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()
    
def setup_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

args = parse_args()
setup_device(args)
setup_seed(args)

train_dataloader, val_dataloader, test_dataloader = create_dataloaders(args)
hard_dataloader = create_hard_dataloaders(args)

model_path = "/data/zhangdacao/save/snli_baseline/bert-base-uncased_pair/model.bin"

tokenizer = AutoTokenizer.from_pretrained(args.bert_dir, use_fast=True)
model = Model(args)

model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.cuda()
#if args.device == 'cuda':
 #   model = torch.nn.parallel.DataParallel(model.to(args.device))

model.eval()


def calc_mean_emb(dataloader):
    all_mean = torch.zeros(1, 768)
    all_num = 0
    en_num, ne_num, con_num = 0, 0, 0
    """en_mean = torch.zeros(1, 768)
    ne_mean = torch.zeros(1, 768)
    con_mean = torch.zeros(1, 768)
    en_num, ne_num, con_num = 0, 0, 0"""
    with torch.no_grad():
        loop = tqdm(dataloader, total = len(dataloader))
        for batch in loop:
            embs, labels = model(batch)
            all_mean += torch.sum(embs, dim=0).cpu()
            all_num += embs.shape[0]
            """for i in range(embs.shape[0]):
                if labels[i] == 0:
                    en_num += 1
                elif labels[i] == 1:
                    ne_num += 1
                else:
                    con_num += 1"""
        all_mean /= all_num
    print(all_num, en_num, ne_num, con_num)
    np.save('data/multinli_all_mean.npy', all_mean)
    #np.save('data/en_mean.npy', en_mean)
    #np.save('data/ne_mean.npy', ne_mean)
    #$np.save('data/con_mean.npy', con_mean)

#calc_mean_emb(train_dataloader)

def calc_topk_word_emb(task_name):
    if task_name == 'SNLI':
        df = pd.read_csv('data/SNLI_word_counts_without_stopwords.csv')
    elif task_name == 'MultiNLI':
        df = pd.read_csv('data/MultiNLI_word_counts_without_stopwords.csv')
    entailment = df['full_hy_entailment']
    neutral = df['full_hy_neutral']
    contradiction = df['full_hy_contradiction']
    labels = [entailment, neutral, contradiction]
    names = ['entailment', 'neutral', 'contradiction']
    for label, name in zip(labels, names):
        mean_emb = torch.zeros(1, 768)
        num = 0
        for i in range(len(label)):
            word, _ = eval(label[i])
            with torch.no_grad():
                encoded_inputs = tokenizer([word], max_length=10, padding='max_length', truncation=True)
                input_ids = torch.LongTensor(encoded_inputs['input_ids']).cuda()
                mask = torch.LongTensor(encoded_inputs['attention_mask']).cuda()
                embs = model.bert(input_ids, mask)['last_hidden_state'][:, 0, :]
                mean_emb += embs.cpu()
                num += 1
        mean_emb /= num
        np.save('data/bert_' + task_name + '_' + name + '.npy', mean_emb)

#calc_topk_word_emb('SNLI')
calc_topk_word_emb('MultiNLI')
                
        
        
        
        
        
        