import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from layers import Conv1D,Attention

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
                 num_filters=300,
                 window_size=3,
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
        
        self.cnn_kernel_num = self._args.cnn_kernel_num
        self.conv = Conv1D(self._args.cnn_method,  self.word_embedding_dim, self._args.cnn_kernel_num,  self._args.cnn_window_size)
        self.dropout_ = nn.Dropout(p=self.dropout_rate, inplace=False)
        self.additive_attention = AdditiveAttention(query_vector_dim, num_filters)
        self.title_attention = Attention(self._args.cnn_kernel_num, self._args.attention_dim)
       

    def forward(self, text):

        if isinstance(text, dict):
            batch_size, num_sentence, seq_len = text['input_ids'].shape
        else:
            batch_size, num_sentence, seq_len = text.shape

        text_vectors = self.embedding(text)

        text_vectors = torch.reshape(text_vectors, (-1,  seq_len, self.word_embedding_dim))

        convoluted_text_vector = self.dropout_(self.conv(text_vectors.permute(0, 2, 1)).permute(0, 2, 1))         
       
        text_vector = self.title_attention(
            convoluted_text_vector).view([batch_size, num_sentence, self._args.cnn_kernel_num]) 

        return text_vector
    

class UserEncoder(nn.Module):
    def __init__(self,
                 news_embedding_dim=400,
                 query_vector_dim=200
                 ):
        super(UserEncoder, self).__init__()
        self.additive_attention = Attention(news_embedding_dim, query_vector_dim)

    def forward(self, clicked_news_vecs):
        pos_user_vector = self.additive_attention(clicked_news_vecs)
        user_vector = pos_user_vector
        return user_vector


class NAML(nn.Module):
    def __init__(self, args):
        self._args = args
        super(NAML, self).__init__()
        self.text_encoder = TextEncoder(self._args)
        self.user_encoder = UserEncoder()
        self.category_embedding = nn.Embedding(num_embeddings=self._args.category_num, embedding_dim=self._args.category_embedding_dim)
        self.subcategory_embedding = nn.Embedding(num_embeddings=self._args.subcategory_num, embedding_dim=self._args.subcategory_embedding_dim)
        
        self.category_affine = nn.Linear(in_features=self._args.category_embedding_dim, out_features=self._args.cnn_kernel_num, bias=True)
        self.subcategory_affine = nn.Linear(in_features=self._args.subcategory_embedding_dim, out_features=self._args.cnn_kernel_num, bias=True)
        
        self.affine1 = nn.Linear(in_features=self._args.cnn_kernel_num, out_features=self._args.attention_dim, bias=True)
        self.affine2 = nn.Linear(in_features=self._args.attention_dim, out_features=1, bias=False)
        
        self.req_grad_params = self.get_req_grad_params(debug=self._args.debug)

    def forward(self, input_line):

        candidate_titles = input_line['candidate_titles']
        history_titles = input_line['history_titles']
        history_category = input_line['click_vert_idx']
        history_subcategory = input_line['click_subvert_idx']

        candidate_category = input_line['candidate_vert_idx']
        candidate_subcategory = input_line['candidate_subvert_idx']

        click_category_embedding = self.category_embedding(history_category)
        click_subcategory_embedding = self.subcategory_embedding(history_subcategory)

        candidate_category_embedding = self.category_embedding(candidate_category)
        candidate_subcategory_embedding = self.subcategory_embedding(candidate_subcategory)

        if isinstance(candidate_titles, dict):
            batch_size, npratio, word_num = candidate_titles['input_ids'].shape
        else:
            batch_size, npratio, word_num = candidate_titles.shape

        candidate_title_representation = self.text_encoder(candidate_titles)
        
        candidate_category_representation = F.relu(self.category_affine(candidate_category_embedding), inplace=True)                              # [batch_size, news_num, cnn_kernel_num]
        candidate_subcategory_representation = F.relu(self.subcategory_affine(candidate_subcategory_embedding), inplace=True) 
        candidate_feature = torch.stack([candidate_title_representation, candidate_category_representation, candidate_subcategory_representation], dim=2)    # [batch_size, news_num, 4, cnn_kernel_num]
        candidate_alpha = F.softmax(self.affine2(torch.tanh(self.affine1(candidate_feature))), dim=2)                                                            # [batch_size, news_num, 4, 1]
        candidate_vector = (candidate_feature * candidate_alpha).sum(dim=2, keepdim=False)                                                                    # [batch_size, news_num, cnn_kernel_num]
        
       
        if isinstance(history_titles, dict):
            batch_size, click_news_num, word_num = history_titles['input_ids'].shape
        else:
            batch_size, click_news_num, word_num = history_titles.shape
        click_title_representation = self.text_encoder(history_titles)
        
        click_category_representation = F.relu(self.category_affine(click_category_embedding), inplace=True)                              # [batch_size, news_num, cnn_kernel_num]
        click_subcategory_representation = F.relu(self.subcategory_affine(click_subcategory_embedding), inplace=True) 
        click_feature = torch.stack([click_title_representation, click_category_representation, click_subcategory_representation], dim=2)    # [batch_size, news_num, 4, cnn_kernel_num]
        click_alpha = F.softmax(self.affine2(torch.tanh(self.affine1(click_feature))), dim=2)                                                            # [batch_size, news_num, 4, 1]
        click_news_vec = (click_feature * click_alpha).sum(dim=2, keepdim=False)  

        user_vector = self.user_encoder(click_news_vec)
    
        score = torch.bmm(candidate_vector, user_vector.unsqueeze(-1)).squeeze(dim=-1)

        return [user_vector, candidate_vector, score]

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


class NAML_CI(nn.Module):
    def __init__(self, args):
        self._args = args
        super(NAML_CI, self).__init__()
        self.backbone = NAML(args)
        if self._args.kl_usage in ['category', 'subcategory']:
            emb_size = 1
        else:
            emb_size = 2

        if self._args.prob_usage == -1:
            current_prob_usage = 3
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
        
        user_vector, candidate_vector, score = self.backbone(input_line)

        if isinstance(history_titles, dict):
            batch_size, click_news_num, word_num = history_titles['input_ids'].shape
        else:
            batch_size, click_news_num, word_num = history_titles.shape

        candidate_ctr = input_line['candidate_ctr']
        history_ctr = input_line['history_ctr']
        user_category_kl = input_line['user_category_kl']
        user_subcategory_kl = input_line['user_subcategory_kl']

        candidate_titles_prob = input_line['candidate_title_prob']
        click_title_prob = input_line['click_title_prob']

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
        else:
            raise ValueError(f'[Error!] wrong prob usage number {self._args.prob_usage}, please try again!')

        if self._args.kl_usage == 'category':
            user_kl = torch.unsqueeze(user_category_kl, dim=-1)
        elif self._args.kl_usage == 'subcategory':
            user_kl = torch.unsqueeze(user_subcategory_kl, dim=-1)
        else:
            user_kl = torch.stack([user_category_kl, user_subcategory_kl], dim=-1)

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