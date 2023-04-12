import math
import os

import numpy as np
import pandas as pd
import torch

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer
from tqdm import tqdm

from data.data_utils import *
from parser import *

import pretty_errors




class MindDataset(Dataset):
    """
    parse user file and news file to generate data
    one line represent one sample
    """
    def __init__(self, args, basic_path, title_size, his_size, npratio, negative_split=None, split='train'):
        self.args = args
        self.basic_path = basic_path
        self.title_size = title_size
        self.his_size = his_size
        self.npratio = npratio
        self._negative_split = negative_split
        self.split = split
        super(MindDataset, self).__init__()

        if not self.args.use_bert:
            self.word_dict = load_dict(os.path.join(self.args.basic_path, 'word2idx.pkl'))

        self.uid2idx = load_dict(os.path.join(basic_path, 'uid2idx.pkl'))
        self.nid2idx = load_dict(os.path.join(basic_path, 'nid2idx.pkl'))
        self.nid2idx[0] = 0

        self.user_data_path = os.path.join(self.basic_path, f'convert_{self.split}_behavior.tsv')
        self.news_data_path = os.path.join(self.basic_path, f'convert_{self.split}_news.tsv')

        if self.args.use_bert:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.pre_trained_model,
                do_lower_case=True,
                cache_dir=self.args.cache_dir
            )

        if not hasattr(self, 'news_info'):
            self.init_news()

        if not hasattr(self, 'user_info'):
            self.init_users()

        if self.npratio > 0:
            self.init_behaviors_sample()
        else:
            self.init_behaviors_nosample()

    def __getitem__(self, index):
        if self.split.lower() in ['dev', 'valid', 'test']:
            current_samples = self.user_info[index]
            info_idx = index
            uindex = current_samples['uindex']
            impressId = current_samples['impressId']
            pos = current_samples['positive_news']
            negs = current_samples['negative_news']
            label = [1] * len(pos) + [0] * len(negs)
            history = current_samples['history']
            category_kl = current_samples['category_kl']
            subcategory_kl = current_samples['subcategory_kl']

            click_title_index = [self.nid2info[i]['title_index'] for i in history]
            click_title_ctr = torch.tensor([self.nid2info[i]['ctr'] for i in history], dtype=torch.float)
            click_title_prob = torch.tensor([[self.nid2info[i]['prob'], self.nid2info[i]['ctprob'], self.nid2info[i]['sctprob']] for i in history], dtype=torch.float)

            candidate_news_index = self._obtain_news_idx(pos + negs)
            candidate_title_index = [self.nid2info[i]['title_index'] for i in pos + negs]
            candidate_title_ctr = torch.tensor([self.nid2info[i]['ctr'] for i in pos + negs], dtype=torch.float)
            candidate_title_prob = torch.tensor([[self.nid2info[i]['prob'], self.nid2info[i]['ctprob'], self.nid2info[i]['sctprob']] for i in pos + negs], dtype=torch.float)

            all_candidate_ctr = current_samples['candidate_ctr']
        else:
            if self.npratio == 0:
                current_samples = self.behavior_list[index]
                info_idx, news, click = current_samples
                uindex = self.user_info[info_idx]['uindex']
                impressId = self.user_info[info_idx]['impressId']
                label = click
                history = self.user_info[info_idx]['history']
                category_kl = self.user_info[info_idx]['category_kl']
                subcategory_kl = self.user_info[info_idx]['subcategory_kl']
                candidate_news_index = self._obtain_news_idx(news)
                candidate_title_index = self.nid2info[news]['title_index']
                candidate_title_ctr = self.nid2info[news]['ctr']
                candidate_title_prob = [self.nid2info[news]['prob'], self.nid2info[news]['ctprob'], self.nid2info[news]['sctprob']]

                click_title_index = [self.nid2info[i]['title_index'] for i in history]
                click_title_ctr = torch.tensor([self.nid2info[i]['ctr'] for i in history], dtype=torch.float)
                click_title_prob = torch.tensor([[self.nid2info[i]['prob'], self.nid2info[i]['ctprob'], self.nid2info[i]['sctprob']] for i in history], dtype=torch.float)

                all_candidate_ctr = self.user_info[info_idx]['candidate_ctr']
            elif self.args.net.lower() == 'dice':
                current_samples = self.behavior_list[index]
                info_idx, news = current_samples
                uindex = self.user_info[info_idx]['uindex']
                impressId = self.user_info[info_idx]['impressId']
                label = [0]
                negs = self.user_info[info_idx]['negative_news']
                history = self.user_info[info_idx]['history']

                threshold = self.nid2info[news]['ctr']
                negs_ctr = [self.nid2info[i]['ctr'] for i in negs]
                n = conditionsample(negs, negs_ctr, threshold, self.npratio, condition=self._negative_split)
    
                n = [news] + n
                candidate_news_index = self._obtain_news_idx(n)
                candidate_title_index = [self.nid2info[i]['title_index'] for i in n]
                candidate_title_ctr = torch.tensor([self.nid2info[i]['ctr'] for i in n], dtype=torch.float)
                candidate_title_prob = torch.tensor([[self.nid2info[i]['prob'], self.nid2info[i]['ctprob'], self.nid2info[i]['sctprob']] for i in n], dtype=torch.float)

                click_title_index = [self.nid2info[i]['title_index'] for i in history]
                click_title_ctr = torch.tensor([self.nid2info[i]['ctr'] for i in history], dtype=torch.float)
                click_title_prob = torch.tensor([[self.nid2info[i]['prob'], self.nid2info[i]['ctprob'], self.nid2info[i]['sctprob']] for i in history], dtype=torch.float)

                category_kl = self.user_info[info_idx]['category_kl']
                subcategory_kl = self.user_info[info_idx]['subcategory_kl']
                all_candidate_ctr = self.user_info[info_idx]['candidate_ctr']
            else:
                current_samples = self.behavior_list[index]
                info_idx, news = current_samples
                uindex = self.user_info[info_idx]['uindex']
                impressId = self.user_info[info_idx]['impressId']

                label = [0]
                negs = self.user_info[info_idx]['negative_news']
                history = self.user_info[info_idx]['history']
                n = newsample(negs, self.npratio)

                n = [news] + n
                candidate_news_index = self._obtain_news_idx(n)
                candidate_title_index = [self.nid2info[i]['title_index'] for i in n]
                candidate_title_ctr = torch.tensor([self.nid2info[i]['ctr'] for i in n], dtype=torch.float)
                candidate_title_prob = torch.tensor([[self.nid2info[i]['prob'], self.nid2info[i]['ctprob'], self.nid2info[i]['sctprob']] for i in n], dtype=torch.float)

                click_title_index = [self.nid2info[i]['title_index'] for i in history]
                click_title_ctr = torch.tensor([self.nid2info[i]['ctr'] for i in history], dtype=torch.float)
                click_title_prob = torch.tensor([[self.nid2info[i]['prob'], self.nid2info[i]['ctprob'], self.nid2info[i]['sctprob']] for i in history], dtype=torch.float)

                category_kl = self.user_info[info_idx]['category_kl']
                subcategory_kl = self.user_info[info_idx]['subcategory_kl']
                all_candidate_ctr = self.user_info[info_idx]['candidate_ctr']

        return label, uindex, candidate_title_index, candidate_title_ctr, \
               click_title_index, click_title_ctr, category_kl, subcategory_kl, \
               candidate_title_prob, click_title_prob, \
               candidate_news_index, all_candidate_ctr, impressId

    def _obtain_news_idx(self, news_id):
        result = []
        for id in news_id:
            if id not in self.nid2idx.keys():
                result.append(self.nid2idx["<unk>"])
            else:
                result.append(self.nid2idx[id])

        return result

    def __len__(self):
        return self.total_length

    def _get_news_count(self):
        return len(self.nid2idx)

    def _get_user_count(self):
        return len(self.user_info)

    def init_news(self):
        self.nid2info = {}
        with open(self.news_data_path, 'r', encoding='utf8') as read:
            lines = read.readlines()

            for line in lines:
                nid, vert, subvert, title, ab, ctr, history_click, prob, ctprob, sctprob = line.strip("\n").split('\t')
                title = word_tokenize(title)
                vert = vert.lower()
                subvert = subvert.lower()
                info = {'vert': vert, 'subvert': subvert, 'ctr': float(ctr), 'prob': float(prob), 'ctprob': float(ctprob), 'sctprob': float(sctprob)}
                if not self.args.use_bert:
                    news_titile = np.zeros(shape=[self.title_size, ])
                    for idx in range(min(self.title_size, len(title))):
                        if title[idx] in self.word_dict:
                            news_titile[idx] = self.word_dict[title[idx].lower()]
                    news_titile = torch.tensor(news_titile, dtype=torch.long)
                else:
                    news_titile = title[:self.title_size]
                    news_titile = self.process_sent(' '.join(news_titile))
                info.update({'title_index': news_titile})

                if not nid in self.nid2info.keys():
                    self.nid2info[nid] = info

                if 0 not in self.nid2info.keys():
                    fake_info = {'vert': vert, 'subvert': subvert, 'ctr': 0., 'prob': 0., 'ctprob': 0., 'sctprob': 0.}
                    fake_title = torch.zeros([self.title_size,], dtype=torch.long)
                    if self.args.use_bert:
                        news_titile = {
                            'input_ids': fake_title,
                            'token_type_ids': torch.zeros_like(fake_title, dtype=torch.long),
                            'attention_mask': torch.ones_like(fake_title, dtype=torch.long)
                        }
                    else:
                        news_titile = fake_title
                    fake_info.update({'title_index': news_titile})
                    self.nid2info[0] = fake_info

    def init_users(self):
        self.user_info = {}

        with open(self.user_data_path, 'r', encoding='utf8') as read:
            lines = read.readlines()
            for line in lines:
                tokens = line.strip().split('\t')
                uid = tokens[0]
                uindex = self.uid2idx[uid] if uid in self.uid2idx else 0
                impressId = int(tokens[1])
                if len(tokens[3]) == 0:
                    history = [0] * self.his_size
                else:
                    history = [i for i in tokens[3].split(' ')]
                    history = [0] * (self.his_size - len(history)) + history[:self.his_size]
                impr_news = [i.split('-')[0] for i in tokens[4].split(' ')]
                impr_label = [int(i.split('-')[1]) for i in tokens[4].split(' ')]

                category_kl = float(tokens[5])
                subcategory_kl = float(tokens[6])
                candidate_ctr = float(tokens[7])

                negative_news = []
                positive_news = []
                for news, click in zip(impr_news, impr_label):
                    if click == 0:
                        negative_news.append(news)
                    else:
                        positive_news.append(news)

                info_index = len(self.user_info)
                self.user_info[info_index] = {
                    'uindex': uindex,
                    'impressId': impressId,
                    'history': history,
                    'positive_news': positive_news,
                    'negative_news': negative_news,
                    'category_kl': category_kl,
                    'subcategory_kl': subcategory_kl,
                    'candidate_ctr': candidate_ctr
                }

        self.total_length = len(self.user_info)

    def init_behaviors_sample(self):
        self.behavior_list = []

        for info_index, info in self.user_info.items():
            positive_news = info['positive_news']

            for news in positive_news:
                self.behavior_list.append([info_index, news])


        self.total_behavior_length = len(self.behavior_list)


    def init_behaviors_nosample(self):
        self.behavior_list = []

        for info_index, info in self.user_info.items():
            positive_news = info['positive_news']
            negative_news = info['negative_news']

            for news in positive_news:
                self.behavior_list.append([info_index, news, 1])
            for news in negative_news:
                self.behavior_list.append([info_index, news, 0])

        self.total_behavior_length = len(self.behavior_list)


    def process_sent(self, sent):
        results = self.tokenizer.encode_plus(sent)

        sentence_ids = torch.tensor(results['input_ids'])
        attention_mask_ids = torch.tensor(results['attention_mask'])
        if 'token_type_ids' in results.keys():
            segment_ids = torch.tensor(results['token_type_ids'])
            values = [sentence_ids, segment_ids, attention_mask_ids]
        else:
            values = [sentence_ids, attention_mask_ids]

        return values


class MINDdata(Dataset):
    def __init__(self, args):
        self.args = args

        if self.args.debug:
            train_split = 'valid'
        else:
            train_split = 'train'
        
        self.train_set = MindDataset(
            args=args,
            basic_path=self.args.basic_path,
            title_size=self.args.title_size,
            his_size=self.args.his_size,
            npratio=self.args.npratio,
            split=train_split,
            negative_split=0
        )
        if self.args.net.lower() == 'dice':
            self.positive_train_set = MindDataset(
                args=args,
                basic_path=self.args.basic_path,
                title_size=self.args.title_size,
                his_size=self.args.his_size,
                npratio=self.args.npratio,
                split=train_split,
                negative_split=1
            )

        if self.args.test:
            test_split = 'test'
        else:
            test_split = 'valid'
        self._test_set = MindDataset(
                args=args,
                basic_path=self.args.basic_path,
                title_size=self.args.title_size,
                his_size=self.args.his_size,
                npratio=self.args.npratio,
                split=test_split
            )

        self._dev_set = MindDataset(
            args=args,
            basic_path=self.args.basic_path,
            title_size=self.args.title_size,
            his_size=self.args.his_size,
            npratio=self.args.npratio,
            split='valid'
        )

    def get_loader(self, batch_size, type='dev', shuffle=True, num_worker=4, pin_memory=True, drop_last=True):
        if type == 'train':
            current_dataset = self.train_set
        elif type == 'valid' or type == 'dev':
            current_dataset = self._dev_set
        elif type == 'test':
            current_dataset = self._test_set
        elif type == 'positive_train' and self.args.net.lower() == 'dice':
            current_dataset = self.positive_train_set
        else:
            raise ValueError(f'[Error!] wrong value for type or net name, please try again')

        loader = DataLoader(
            current_dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_worker,
            pin_memory=True,
            drop_last=drop_last,
            collate_fn=self._collate_fn
        )

        return loader

    def _pad_sequence(self, content_list):
        if isinstance(content_list[0], torch.Tensor):
            results = torch.stack(content_list, dim=0)
        elif isinstance(content_list[0], list):
            results = []
            for content in content_list:
                result = pad_sequence(content, batch_first=True, padding_value=0)
                results.append(torch.transpose(result, 1, 0))
            results = pad_sequence(results, batch_first=True, padding_value=0)
            results = torch.transpose(results, 2, 1)
        elif isinstance(content_list[0], float):
            results = torch.tensor(content_list, dtype=torch.float)
        else:
            raise ValueError('wrong content list, please try again')

        return results

    def _collate_fn(self, batch):
        input_data = []
        labels = torch.tensor([item[0] for item in batch]).squeeze(dim=1)
        user_index = torch.tensor([item[1] for item in batch])
        if self.args.use_bert:
            if len(batch[0][2][0]) == 3:
                input_ids = []
                segment_ids = []
                attention_mask = []
                for item in batch:
                    input_ids.append([content[0] for content in item[2]])
                    segment_ids.append([content[1] for content in item[2]])
                    attention_mask.append([content[2] for content in item[2]])

                candidate_ids = self._pad_sequence(input_ids)
                candidate_segment_ids = self._pad_sequence(segment_ids)
                candidate_attention_mask = self._pad_sequence(attention_mask)

                candidate_titles = {
                    'input_ids': candidate_ids,
                    'token_type_ids': candidate_segment_ids,
                    'attention_mask': candidate_attention_mask
                }
                input_ids = []
                segment_ids = []
                attention_mask = []
                for item in batch:
                    input_ids.append([content[0] for content in item[4]])
                    segment_ids.append([content[1] for content in item[4]])
                    attention_mask.append([content[2] for content in item[4]])

                history_ids = self._pad_sequence(input_ids)
                history_segment_ids = self._pad_sequence(segment_ids)
                history_attention_mask = self._pad_sequence(attention_mask)

                history_titles = {
                    'input_ids': history_ids,
                    'token_type_ids': history_segment_ids,
                    'attention_mask': history_attention_mask
                }
            elif len(batch[0][2][0]) == 2:
                input_ids = []
                attention_mask = []
                for item in batch:
                    input_ids.append([content[0] for content in item[2]])
                    attention_mask.append([content[1] for content in item[2]])
                candidate_ids = self._pad_sequence(input_ids)
                candidate_attention_mask = self._pad_sequence(attention_mask)

                candidate_titles = {
                    'input_ids': candidate_ids,
                    'attention_mask': candidate_attention_mask
                }

                input_ids = []
                attention_mask = []
                for item in batch:
                    input_ids.append([content[0] for content in item[4]])
                    attention_mask.append([content[1] for content in item[4]])

                history_ids = self._pad_sequence(input_ids)
                history_attention_mask = self._pad_sequence(attention_mask)

                history_titles = {
                    'input_ids': history_ids,
                    'attention_mask': history_attention_mask
                }
            else:
                candidate_titles = self._pad_sequence([item[2] for item in batch])
                history_titles = self._pad_sequence([item[4] for item in batch])
        else:
            candidate_titles = self._pad_sequence([item[2] for item in batch])
            history_titles = self._pad_sequence([item[4] for item in batch])

        candidate_ctr = self._pad_sequence([item[3] for item in batch])
        history_ctr = self._pad_sequence([item[5] for item in batch])

        category_kl = torch.tensor([item[6] for item in batch], dtype=torch.float)
        subcategory_kl = torch.tensor([item[7] for item in batch], dtype=torch.float)

        candidate_news_index = torch.tensor([item[-3] for item in batch])
        all_candidate_ctr = torch.tensor([item[-2] for item in batch], dtype=torch.float)

        impressId = torch.tensor([item[-1] for item in batch])

        click_title_prob = self._pad_sequence([item[-4] for item in batch])
        candidate_title_prob = self._pad_sequence([item[-5] for item in batch])

        return labels, user_index, candidate_titles, candidate_ctr, history_titles, history_ctr, \
               category_kl, subcategory_kl, click_title_prob, candidate_title_prob, \
               candidate_news_index, all_candidate_ctr, impressId