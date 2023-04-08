import torch
from config import parse_args
from model import Model
from transformers import AutoTokenizer
from load_datasets import create_dataloaders, create_hard_dataloaders
import logging
import os
import time
import torch
from tqdm import tqdm
from optimizer import build_optimizer
import scipy.stats
import random
import numpy as np
from tqdm import tqdm

hard_sample = 1000
hard_ratio = 0.6
def setup_device(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()
    
def setup_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def gen_hard(args, tokenizer, model):
    GTE_test_data = np.load('dataset/GTE/test.npy', allow_pickle='TRUE')
    hard_e, hard_n, hard_c = [], [], []
    hard = []
    loop = tqdm(dataset, total = len(dataset))
    for data, GTE in zip(loop, GTE_test_data):
        pre = data['premise']    
        hy = data['hypothesis']  
        label = data['label']
        encoded_inputs = tokenizer([["", hy]], max_length=args.bert_seq_length, padding='max_length', truncation=True)
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        inputs = {'image':None, 'text':{'input_ids':input_ids.cuda(), 'attention_mask':mask.cuda()}, 'label':label}
        logits, pred, _ = model(inputs, inference=True)
        #print({'premise':pre, 'hypothesis':hy, 'label':label})
        #print({'premise':GTE['premise'], 'pre_img': GTE['pre_img'], 'hypothesis':GTE['hypothesis'], 'label':GTE['label']})
        if label == 0:
            if logits[0][0].item() < hard_ratio:
                hard_e.append({'premise':GTE['premise'], 'pre_img': GTE['pre_img'], 'hypothesis':GTE['hypothesis'], 'label':GTE['label']})
        elif label == 1:
            if logits[0][1].item() < hard_ratio:
                hard_n.append({'premise':GTE['premise'], 'pre_img': GTE['pre_img'], 'hypothesis':GTE['hypothesis'], 'label':GTE['label']})
        elif label == 2:
            if logits[0][2].item() < hard_ratio:
                hard_c.append({'premise':GTE['premise'], 'pre_img': GTE['pre_img'], 'hypothesis':GTE['hypothesis'], 'label':GTE['label']})
    hard = hard_e + hard_n + hard_c
    print(len(hard_e), len(hard_n), len(hard_c))
    return hard
   
if __name__ == '__main__':
    args = parse_args()
    setup_device(args)
    setup_seed(args)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_dir, use_fast=True)
    model = Model(args)
    model_path = "save/only_hy/bert-base-uncased/model.bin"
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.cuda()
    model.eval()
    
    hard = gen_hard(args, tokenizer, model)
    np.save('dataset/GTE/hard.npy', hard)
