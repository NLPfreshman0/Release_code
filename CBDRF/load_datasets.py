import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
import numpy as np
import jsonlines
import re

class Datasets(Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        self.confactual = self.args.confactual
        self.task_name = args.task_name
        self.bert_seq_length = args.bert_seq_length
        self.pair = args.pair
        self.pattern = r"\b\w+\b"
        if self.task_name == 'GTE':
            self.image = np.load('dataset/GTE/clip_image.npy', allow_pickle='TRUE').item()
            self.data = np.load('dataset/GTE/'+split+'.npy', allow_pickle='TRUE')
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_dir, use_fast=True)
            
    def __len__(self):
        return len(self.data)
        
    def tokenize_text(self, text):
        encoded_inputs = self.tokenizer(text, max_length=self.bert_seq_length, padding='max_length', truncation=True)
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        return input_ids, mask
    
    def __getitem__(self, idx):
        if self.task_name == 'GTE':
            if self.data[idx]['pre_img'] not in self.image.keys():
                print(self.data[idx]['pre_img'])
            image = torch.Tensor(self.image[self.data[idx]['pre_img']])
            text1, text2, label = self.data[idx]['premise'], self.data[idx]['hypothesis'], self.data[idx]['label']
            premise_input_ids, premise_mask = self.tokenize_text([text1])
            hypothesis_input_ids, hypothesis_mask = self.tokenize_text([['', text2]])
            con_input_ids, con_mask = self.tokenize_text([[text1, text2]])
            return {'image':image, 
                    'text':{'premise':{'input_ids':premise_input_ids, 'attention_mask':premise_mask}, 
                            'hypothesis':{'input_ids':hypothesis_input_ids, 'attention_mask':hypothesis_mask}, 
                            'con_text':{'input_ids':con_input_ids, 'attention_mask':con_mask}},
                    'label':label}


def create_GTE_hard(args):
    dataset = Datasets(args, split='hard')
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                              batch_size=args.test_batch_size,
                              sampler=sampler,
                              drop_last=False,
                              pin_memory=True,
                              num_workers=args.num_workers,
                              prefetch_factor=args.prefetch)
    
    return dataloader
    
      
def create_dataloaders(args, confactual=0):
    args.confactual = confactual
    train_dataset = Datasets(args, split='train')
    val_dataset = Datasets(args, split='validation')
    test_dataset = Datasets(args, split='test')
    
    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    test_sampler = SequentialSampler(test_dataset)
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  drop_last=True,
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  prefetch_factor=args.prefetch)
    
    val_dataloader = DataLoader(val_dataset,
                                  batch_size=args.val_batch_size,
                                  sampler=val_sampler,
                                  drop_last=False,
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  prefetch_factor=args.prefetch)
    
    test_dataloader = DataLoader(test_dataset,
                                  batch_size=args.test_batch_size,
                                  sampler=test_sampler,
                                  drop_last=False,
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  prefetch_factor=args.prefetch)
    
    return train_dataloader, val_dataloader, test_dataloader


    
