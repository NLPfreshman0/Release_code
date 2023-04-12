import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import AutoModel, AutoConfig

from class_utils import *


class TextEmbedding(nn.Module):
    def __init__(self, args):
        self._args = args
        super(TextEmbedding, self).__init__()

        use_cuda = self._args.gpu != '' and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        train_flag = self._args.embedding_train
        embedding_freeze = not self._args.embedding_train

        if self._args.use_bert:
            self.encoding = AutoModel.from_pretrained(
                self._args.pre_trained_model,
                cache_dir=self._args.cache_dir,
                output_attentions=False
            )
            config = AutoConfig.from_pretrained(
                self._args.pre_trained_model,
                cache_dir=self._args.cache_dir,
            )
            self.word_embedding_size = config.hidden_size

            for param in self.encoding.parameters():
                param.requires_grad = train_flag
        else:
            embedding_matrix = np.load(os.path.join(self._args.basic_path, 'word2vec.npy'))
            pre_trained_embedding = torch.from_numpy(embedding_matrix).float()
            self.encoding = nn.Embedding.from_pretrained(
                pre_trained_embedding,
                freeze=embedding_freeze
            )
            self.word_embedding_size = embedding_matrix.shape[-1]

    def forward(self, text):
        if self._args.use_bert:
            sizes = text['input_ids'].shape
            input_ids = torch.reshape(text['input_ids'], (-1, sizes[-1]))
            attention_masks = torch.reshape(text['attention_mask'], (-1, sizes[-1]))
            if len(text) == 3:
                segment_ids = torch.reshape(text['token_type_ids'], (-1, sizes[-1]))
                outputs = self.encoding(
                    input_ids=input_ids,
                    attention_mask=attention_masks,
                    token_type_ids=segment_ids,
                )
            else:
                outputs = self.encoding(
                    input_ids=input_ids,
                    attention_mask=attention_masks,
                )
            outputs = outputs['last_hidden_state']
            new_sizes = outputs.shape

            outputs = outputs.reshape(sizes[0], -1, sizes[2], new_sizes[-1])
        else:
            outputs = self.encoding(text)
            outputs = F.dropout(outputs, self._args.dropout, training=self.training)

        return outputs

    def get_embedding_size(self):
        return self.word_embedding_size


class TextEncoder(nn.Module):
    def __init__(self,
                 args,
                 num_attention_heads=20,
                 query_vector_dim=200,
                 dropout_rate=0.2,
                 enable_gpu=True):
        super(TextEncoder, self).__init__()
        self._args = args
        self._dropout_rate = dropout_rate
        self.dropout_rate = 0.2

        self.embedding = TextEmbedding(
            args=self._args,
        )

        self.word_embedding_dim = self.embedding.get_embedding_size()

        self.multihead_attention = MultiHeadAttention(self.word_embedding_dim, num_attention_heads, 20, 20)
        self.additive_attention = AdditiveAttention(num_attention_heads * 20, query_vector_dim)

    def forward(self, text):

        if isinstance(text, dict):
            batch_size, num_sentence, seq_len = text['input_ids'].shape
        else:
            batch_size, num_sentence, seq_len = text.shape

        text_vectors = self.embedding(text)

        text_vectors = torch.reshape(text_vectors, (-1,  seq_len, self.word_embedding_dim))

        multihead_text_vector = self.multihead_attention(
            text_vectors, text_vectors, text_vectors)

        multihead_text_vector = F.dropout(multihead_text_vector,
                                          p=self.dropout_rate,
                                          training=self.training)
        text_vector = self.additive_attention(multihead_text_vector)
        return text_vector


class UserEncoder(nn.Module):
    def __init__(self,
                 news_embedding_dim=400,
                 num_attention_heads=20,
                 query_vector_dim=200
                 ):
        super(UserEncoder, self).__init__()
        self.multihead_attention = MultiHeadAttention(news_embedding_dim, num_attention_heads, 20, 20)
        self.additive_attention = AdditiveAttention(num_attention_heads * 20, query_vector_dim)

        self.neg_multihead_attention = MultiHeadAttention(news_embedding_dim, num_attention_heads, 20, 20)

    def forward(self, clicked_news_vecs):
        multi_clicked_vectors = self.multihead_attention(
            clicked_news_vecs, clicked_news_vecs, clicked_news_vecs
        )
        pos_user_vector = self.additive_attention(multi_clicked_vectors)

        user_vector = pos_user_vector
        return user_vector


class NRMS(nn.Module):
    def __init__(self, args):
        self._args = args
        super(NRMS, self).__init__()
        self.text_encoder = TextEncoder(self._args)
        self.user_encoder = UserEncoder()

        self.req_grad_params = self.get_req_grad_params(debug=self._args.debug)

    def forward(self, input_line):

        candidate_titles = input_line['candidate_titles']
        history_titles = input_line['history_titles']

        if isinstance(candidate_titles, dict):
            batch_size, npratio, word_num = candidate_titles['input_ids'].shape
        else:
            batch_size, npratio, word_num = candidate_titles.shape

        candidate_vector = self.text_encoder(candidate_titles).view(batch_size, npratio, -1)

        if isinstance(history_titles, dict):
            batch_size, click_news_num, word_num = history_titles['input_ids'].shape
        else:
            batch_size, click_news_num, word_num = history_titles.shape
        click_news_vec = self.text_encoder(history_titles).view(batch_size, click_news_num, -1)

        user_vector = self.user_encoder(click_news_vec)

        score = torch.bmm(candidate_vector, user_vector.unsqueeze(-1)).squeeze(dim=-1)

        return [score]

    def get_req_grad_params(self, debug=False):
        print(f'# {self._args.name} parameters: ', end='')
        params = list()
        total_size = 0

        def multiply_iter(p_list):
            out = 1
            for _p in p_list:
                out *= _p

            return out

        for name, p in self.named_parameters():
            if p.requires_grad:
                params.append(p)
                n_params = multiply_iter(p.size())
                total_size += n_params
                if True:
                    print(name, p.requires_grad, p.size(), multiply_iter(p.size()), sep='\t')

        print(f'{total_size},')
        return params


class NRMS_CI(nn.Module):
    def __init__(self, args):
        self._args = args
        super(NRMS_CI, self).__init__()
        self.text_encoder = TextEncoder(self._args)
        self.user_encoder = UserEncoder()

        if self._args.kl_usage in ['category', 'subcategory','ctr']:
            emb_size = 1
        elif self._args.kl_usage == 'kl_ctr':
            emb_size = 3
        else:
            emb_size = 2

        if self._args.prob_usage == -1:
            current_prob_usage = 3
        elif self._args.prob_usage == 5:
            current_prob_usage = 4
        else:
            current_prob_usage = self._args.prob_usage
        self.conformity_cal = ConformtyCalv3(self._args, emb_size=emb_size, news_prob_size=current_prob_usage)

        self.req_grad_params = self.get_req_grad_params(debug=self._args.debug)

    def forward(self, input_line, is_train=False):
         
        candidate_titles = input_line['candidate_titles']
        history_titles = input_line['history_titles']


        if isinstance(candidate_titles, dict):
            batch_size, npratio, word_num = candidate_titles['input_ids'].shape
        else:
            batch_size, npratio, word_num = candidate_titles.shape



        candidate_vector = self.text_encoder(candidate_titles).view(batch_size, npratio, -1)

        if isinstance(history_titles, dict):
            batch_size, click_news_num, word_num = history_titles['input_ids'].shape
        else:
            batch_size, click_news_num, word_num = history_titles.shape
        click_news_vec = self.text_encoder(history_titles).view(batch_size, click_news_num, -1)

        user_vector = self.user_encoder(click_news_vec)

        score = torch.bmm(candidate_vector, user_vector.unsqueeze(-1)).squeeze(dim=-1)

        candidate_ctr = input_line['candidate_ctr']
        history_ctr = input_line['history_ctr']
        user_category_kl = input_line['user_category_kl']
        user_subcategory_kl = input_line['user_subcategory_kl']

        candidate_titles_prob = input_line['candidate_title_prob']
        click_title_prob = input_line['click_title_prob']
        all_candidate_ctr = input_line['all_candidate_ctr']

        if self._args.prob_usage == 1:
            used_candidate_ctr = torch.unsqueeze(candidate_ctr, dim=-1)
            used_history_ctr = torch.unsqueeze(candidate_ctr, dim=-1)

        elif self._args.prob_usage == -1:
            used_candidate_ctr = candidate_titles_prob
            used_history_ctr = click_title_prob

        elif self._args.prob_usage == 2:
            used_candidate_ctr = torch.cat([torch.unsqueeze(candidate_ctr, dim=-1), candidate_titles_prob[:,:,:1]], dim=-1)
            used_history_ctr = torch.cat([torch.unsqueeze(history_ctr, dim=-1), click_title_prob[:,:,:1]], dim=-1)
        
        elif self._args.prob_usage == 3:
            used_candidate_ctr = torch.cat([torch.unsqueeze(candidate_ctr, dim=-1), candidate_titles_prob[:,:,:2]], dim=-1)
            used_history_ctr = torch.cat([torch.unsqueeze(history_ctr, dim=-1), click_title_prob[:,:,:2]], dim=-1)
        
        elif self._args.prob_usage == 4:
            used_candidate_ctr = torch.cat([torch.unsqueeze(candidate_ctr, dim=-1), candidate_titles_prob[:,:,:3]], dim=-1)
            used_history_ctr = torch.cat([torch.unsqueeze(history_ctr, dim=-1), click_title_prob[:,:,:3]], dim=-1)
        
        elif self._args.prob_usage == 5:
            used_candidate_ctr = torch.cat([torch.unsqueeze(candidate_ctr, dim=-1), candidate_titles_prob], dim=-1)
            used_history_ctr = torch.cat([torch.unsqueeze(history_ctr, dim=-1),click_title_prob],dim=-1)
            
        else:
            raise ValueError(f'[Error!] wrong prob usage number {self._args.prob_usage}, please try again!')

        if self._args.kl_usage == 'category':
            user_kl = torch.unsqueeze(user_category_kl, dim=-1)
        elif self._args.kl_usage == 'subcategory':
            user_kl = torch.unsqueeze(user_subcategory_kl, dim=-1)
        elif self._args.kl_usage == 'all': 
            user_kl = torch.stack([user_category_kl, user_subcategory_kl], dim=-1)
        elif self._args.kl_usage == 'kl_ctr':
            user_kl = torch.stack([user_category_kl,user_subcategory_kl,all_candidate_ctr], dim=-1)

        conformity = self.conformity_cal(
            candidate_ctr=used_candidate_ctr,
            history_ctr=used_history_ctr,
            user_kl=user_kl,
            fuse=self._args.fusing,
            hyper_alpha=self._args.user_weight
        )

        if isinstance(conformity, list):
            conformity_factor = F.softplus(conformity[0]+conformity[1])
        else:
            conformity_factor = F.softplus(conformity)

        weighted_score = torch.multiply(conformity_factor, score)
        
        return [weighted_score, score, conformity_factor]

    def get_req_grad_params(self, debug=False):
        print(f'# {self._args.name} parameters: ', end='')
        params = list()
        total_size = 0

        def multiply_iter(p_list):
            out = 1
            for _p in p_list:
                out *= _p

            return out

        for name, p in self.named_parameters():
            if p.requires_grad:
                params.append(p)
                n_params = multiply_iter(p.size())
                total_size += n_params
                if debug:
                    print(name, p.requires_grad, p.size(), multiply_iter(p.size()), sep='\t')

        print(f'{total_size},')
        return params