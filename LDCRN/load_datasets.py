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
        self.mean_pre = np.load('data/all_mean.npy', allow_pickle=True).reshape(-1)
        if self.pair:
            self.bert_seq_length *= 2
        if self.task_name == 'SNLI':
            self.data = np.load('/root/dataset/snli/'+split+'.npy', allow_pickle='TRUE')
        elif self.task_name == 'SNLI-VE':
            self.image = np.load('/root/dataset/snli_ve/image.npy', allow_pickle='TRUE').item()
            self.data = np.load('/root/dataset/snli_ve/'+split+'.npy', allow_pickle='TRUE')
        elif self.task_name == 'MultiNLI':
            if split == 'train':
                self.data = np.load('/root/dataset/multi_nli/train.npy', allow_pickle='TRUE')
            elif split == 'validation':
                self.data = np.load('/root/dataset/multi_nli/validation_matched.npy', allow_pickle='TRUE')
            else:
                self.data = np.load('/root/dataset/multi_nli/validation_mismatched.npy', allow_pickle='TRUE')
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_dir, use_fast=True)
            
    def __len__(self):
        return len(self.data)
        
    def tokenize_text(self, text):
        encoded_inputs = self.tokenizer(text, max_length=self.bert_seq_length, padding='max_length', truncation=True)
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        return input_ids, mask
    
    def __getitem__(self, idx):
        if self.args.only_pre:
            text1, label = self.data[idx]['premise'], self.data[idx]['label']
            text2 = ''
            if self.args.pair:
                input_ids, mask = self.tokenize_text([[text1, text2]])
            else:
                input_ids, mask = self.tokenize_text([text1, text2])
            return {'input_ids':input_ids, 'attention_mask':mask, 'label':label}
        if self.args.only_hy:
            text2, label = self.data[idx]['hypothesis'], self.data[idx]['label']
            text1 = ''
            if self.args.pair:
                input_ids, mask = self.tokenize_text([[text1, text2]])
            else:
                input_ids, mask = self.tokenize_text([text1, text2])
            return {'input_ids':input_ids, 'attention_mask':mask, 'label':label}
        if self.args.confactual_train:
            text1, text2, label = self.data[idx]['premise'], self.data[idx]['hypothesis'], self.data[idx]['label']
            confactual_text1 = ''
            if self.pair:
                factual_input_ids, factual_mask = self.tokenize_text([[text1, text2]])
                confactual_input_ids, confactual_mask = self.tokenize_text([[confactual_text1, text2]])
            else:
                factual_input_ids, factual_mask = self.tokenize_text([text1, text2])
                confactual_input_ids, confactual_mask = self.tokenize_text([confactual_text1, text2])
            return {'factual_text':{'input_ids':factual_input_ids, 'attention_mask':factual_mask}, 'confactual_text':{'input_ids':confactual_input_ids, 'attention_mask':confactual_mask}, 'label':label}
        if self.task_name == 'SNLI' or 'MultiNLI':
            text1, text2, label = self.data[idx]['premise'], self.data[idx]['hypothesis'], self.data[idx]['label']
            if self.confactual == 1:
                text1 = re.sub(self.pattern, "[MASK]", text1)
            elif self.confactual == 2:
                text1 = ""
            if self.args.pair:
                input_ids, mask = self.tokenize_text([[text1, text2]])
            else:
                input_ids, mask = self.tokenize_text([text1, text2])
            return {'input_ids':input_ids, 'attention_mask':mask, 'label':label}
        elif self.task_name == 'SNLI-VE':
            image = torch.Tensor(self.image[self.data[idx]['premise']])
            text = self.data[idx]['hypothesis']
            input_ids, mask = self.tokenize_text([text])
            label = self.data[idx]['label']
            return {'image':image, 'text':{'input_ids':input_ids, 'attention_mask':mask}, 'label':label}

class MultiNLI_Hard(Dataset):
    def __init__(self, args, matched=True):
        self.args = args
        self.confactual = self.args.confactual
        self.bert_seq_length = args.bert_seq_length
        self.pair = args.pair
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_dir, use_fast=True)
        self.pattern = r"\b\w+\b"
        if matched:
            self.data = np.load('/root/dataset/multi_nli/matched_hard.npy', allow_pickle='TRUE')
        else:
            self.data = np.load('/root/dataset/multi_nli/mismatched_hard.npy', allow_pickle='TRUE')
            
    def __len__(self):
        return len(self.data)
        
    def tokenize_text(self, text):
        encoded_inputs = self.tokenizer(text, max_length=self.bert_seq_length, padding='max_length', truncation=True)
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        return input_ids, mask
    
    def __getitem__(self, idx):
        if self.args.confactual_train:
            text1, text2, label = self.data[idx]['premise'], self.data[idx]['hypothesis'], self.data[idx]['label']
            confactual_text1 = ''
            if self.pair:
                factual_input_ids, factual_mask = self.tokenize_text([[text1, text2]])
                confactual_input_ids, confactual_mask = self.tokenize_text([[confactual_text1, text2]])
            else:
                factual_input_ids, factual_mask = self.tokenize_text([text1, text2])
                confactual_input_ids, confactual_mask = self.tokenize_text([confactual_text1, text2])
            return {'factual_text':{'input_ids':factual_input_ids, 'attention_mask':factual_mask}, 'confactual_text':{'input_ids':confactual_input_ids, 'attention_mask':confactual_mask}, 'label':label}
        text1, text2, label = self.data[idx]['premise'], self.data[idx]['hypothesis'], self.data[idx]['label']
        if self.confactual == 1:
            text1 = re.sub(self.pattern, "[MASK]", text1)
        elif self.confactual == 2:
            text1 = ""
        if self.pair:
            input_ids, mask = self.tokenize_text([[text1, text2]])
        else:
            input_ids, mask = self.tokenize_text([text1, text2])
        return {'input_ids':input_ids, 'attention_mask':mask, 'label':label}
        
class SNLI_Hard(Dataset):
    def __init__(self, args):
        self.args = args
        self.confactual = self.args.confactual
        self.bert_seq_length = args.bert_seq_length
        self.pair = args.pair
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_dir, use_fast=True)
        self.pattern = r"\b\w+\b"
        self.data = []
        label_dict = {'entailment':0, 'neutral':1, 'contradiction':2}
        f = open('/root/dataset/snli/test_hard.jsonl', 'r+', encoding='utf-8')
        for line in jsonlines.Reader(f): 
            premise = str(line['sentence1'])
            hypothesis = str(line['sentence2'])
            label = label_dict[str(line['gold_label'])]
            self.data.append({'premise':premise, 'hypothesis':hypothesis, 'label':label})
            
    def __len__(self):
        return len(self.data)
        
    def tokenize_text(self, text):
        encoded_inputs = self.tokenizer(text, max_length=self.bert_seq_length, padding='max_length', truncation=True)
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        return input_ids, mask
    
    def __getitem__(self, idx):
        if self.args.only_pre:
            text1, label = self.data[idx]['premise'], self.data[idx]['label']
            text2 = ''
            if self.args.pair:
                input_ids, mask = self.tokenize_text([[text1, text2]])
            else:
                input_ids, mask = self.tokenize_text([text1, text2])
            return {'input_ids':input_ids, 'attention_mask':mask, 'label':label}
        if self.args.only_hy:
            text2, label = self.data[idx]['hypothesis'], self.data[idx]['label']
            text1 = ''
            if self.args.pair:
                input_ids, mask = self.tokenize_text([[text1, text2]])
            else:
                input_ids, mask = self.tokenize_text([text1, text2])
            return {'input_ids':input_ids, 'attention_mask':mask, 'label':label}
        if self.args.confactual_train:
            text1, text2, label = self.data[idx]['premise'], self.data[idx]['hypothesis'], self.data[idx]['label']
            confactual_text1 = ''
            if self.pair:
                factual_input_ids, factual_mask = self.tokenize_text([[text1, text2]])
                confactual_input_ids, confactual_mask = self.tokenize_text([[confactual_text1, text2]])
            else:
                factual_input_ids, factual_mask = self.tokenize_text([text1, text2])
                confactual_input_ids, confactual_mask = self.tokenize_text([confactual_text1, text2])
            return {'factual_text':{'input_ids':factual_input_ids, 'attention_mask':factual_mask}, 'confactual_text':{'input_ids':confactual_input_ids, 'attention_mask':confactual_mask}, 'label':label}
        text1, text2, label = self.data[idx]['premise'], self.data[idx]['hypothesis'], self.data[idx]['label']
        if self.args.only_pre:
            text2 = ''
        if self.args.only_hy:
            text1 = ''
        if self.confactual == 1:
            text1 = re.sub(self.pattern, "[MASK]", text1)
        elif self.confactual == 2:
            text1 = ""
        if self.pair:
            input_ids, mask = self.tokenize_text([[text1, text2]])
        else:
            input_ids, mask = self.tokenize_text([text1, text2])
        return {'input_ids':input_ids, 'attention_mask':mask, 'label':label}

def create_hard_dataloaders(args, confactual=0):  
    args.confactual = confactual
    hard_dataset = SNLI_Hard(args)
    test_sampler = SequentialSampler(hard_dataset)
    hard_dataloader = DataLoader(hard_dataset,
                                  batch_size=args.test_batch_size,
                                  sampler=test_sampler,
                                  drop_last=False,
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  prefetch_factor=args.prefetch)
    return hard_dataloader
    
def create_MultiNLI_hard_dataloaders(args, confactual=0):
    args.confactual = confactual
    matched_hard = MultiNLI_Hard(args)
    mismatched_hard = MultiNLI_Hard(args, matched=False)
    matched_sampler = SequentialSampler(matched_hard)
    mismatched_sampler = SequentialSampler(mismatched_hard)
    matched_dataloader = DataLoader(matched_hard,
                                  batch_size=args.test_batch_size,
                                  sampler=matched_sampler,
                                  drop_last=False,
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  prefetch_factor=args.prefetch)
    mismatched_dataloader = DataLoader(mismatched_hard,
                                  batch_size=args.test_batch_size,
                                  sampler=mismatched_sampler,
                                  drop_last=False,
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  prefetch_factor=args.prefetch)
    return matched_dataloader, mismatched_dataloader
    
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


    
