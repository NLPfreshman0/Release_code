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
            # output.shape = [batch_size, number_text, seq_len, hidden_size]
            outputs = outputs.reshape(sizes[0], -1, sizes[2], new_sizes[-1])
        else:
            # require tensor [batch_size, sentence_num, length]
            outputs = self.encoding(text)
            # output.shape = [batch_size, number_text, seq_len, emb_size]
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

        self.output_size = num_attention_heads * 20

    def forward(self, text):

        if isinstance(text, dict):
            batch_size, num_sentence, seq_len = text['input_ids'].shape
        else:
            batch_size, num_sentence, seq_len = text.shape

        text_vectors = self.embedding(text)

        text_vectors = torch.reshape(text_vectors, (-1,  seq_len, self.word_embedding_dim))

        # shape = [batch_size*num_news, seq_length, word_embedding_size]
        multihead_text_vector = self.multihead_attention(
            text_vectors, text_vectors, text_vectors)

        multihead_text_vector = F.dropout(multihead_text_vector,
                                          p=self.dropout_rate,
                                          training=self.training)
        # batch_size, word_embedding_dim
        text_vector = self.additive_attention(multihead_text_vector)
        return text_vector

    def get_output_size(self):
        return self.output_size


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
        self.output_size = num_attention_heads * 20

    def forward(self, clicked_news_vecs):
        multi_clicked_vectors = self.multihead_attention(
            clicked_news_vecs, clicked_news_vecs, clicked_news_vecs
        )
        pos_user_vector = self.additive_attention(multi_clicked_vectors)

        user_vector = pos_user_vector
        return user_vector

    def get_output_size(self):
        return self.output_size


class basic_module(nn.Module):
    def __init__(self):
        super(basic_module, self).__init__()

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


class NRMS_backbone(basic_module):
    def __init__(self, args):
        super(NRMS_backbone, self).__init__()
        self._args = args
        self.text_encoder = TextEncoder(self._args)
        self.user_encoder = UserEncoder()

        self.req_grad_params = self.get_req_grad_params(debug=self._args.debug)

        self.output_size = self.text_encoder.get_output_size()

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

        return [user_vector, candidate_vector, score]

    def get_output_size(self):
        return self.output_size


class MACR(basic_module):
    def __init__(self, args):
        super(MACR, self).__init__()
        self._args = args
        self.backbone = NRMS_backbone(args)
        self.vec_size = self.backbone.get_output_size()

        self.user_side = nn.Sequential(
            nn.Linear(in_features=self.vec_size, out_features=1),
            nn.Sigmoid()
        )
        self.item_side = nn.Sequential(
            nn.Linear(in_features=self.vec_size, out_features=1),
            nn.Sigmoid()
        )

        self.req_grad_params = self.get_req_grad_params(debug=self._args.debug)

    def forward(self, input_line, is_train=True):
        user_vec, candidate_vecs, matching_score = self.backbone(input_line)

        batch_size, npratio, embed_size = candidate_vecs.shape
        user_vec_multiple = user_vec.unsqueeze(1).expand(-1, npratio, -1)

        item_score = self.item_side(candidate_vecs).squeeze(dim=-1)
        user_score = self.user_side(user_vec_multiple).squeeze(dim=-1)
        print('&&&&&&&&&&&& conformity_score=', torch.multiply(user_score, item_score))

        final_score = torch.multiply(matching_score, torch.multiply(user_score, item_score))

        return [final_score,  matching_score, user_score, item_score]


class PDA(basic_module):
    def __init__(self, args):
        super(PDA, self).__init__()
        self._args = args
        self.ctr_power = self._args.pda_power
        self.backbone = NRMS_backbone(args)
        self.vec_size = self.backbone.get_output_size()
        self.my_elu = MyELU()

        self.req_grad_params = self.get_req_grad_params(debug=self._args.debug)

    def forward(self, input_line, is_train=True):
        user_vec, candidate_vecs, matching_score = self.backbone(input_line)

        candidate_ctr = input_line['candidate_ctr']

        batch_size, npratio, embed_size = candidate_vecs.shape

        activated_matching = self.my_elu(matching_score)

        final_score = activated_matching * torch.pow(candidate_ctr, self.ctr_power)

        return [final_score, activated_matching, matching_score]


class TIDE(basic_module):
    def __init__(self, args, item_size):
        super(TIDE, self).__init__()
        self._args = args
        self._item_size = item_size
        self.backbone = NRMS_backbone(args)

        self.vec_size = self.backbone.get_output_size()

        self.quality_encoder = nn.Embedding(self._item_size, 1)
        self.conformity_encoder = ConformtyCalv2(args, emb_size=2)

        self.req_grad_params = self.get_req_grad_params(debug=self._args.debug)

    def forward(self, input_line, is_train=True):
        user_vec, candidate_vecs, matching_score = self.backbone(input_line)

        candidate_titles = input_line['candidate_index']

        item_quality = self.quality_encoder(candidate_titles).squeeze(dim=-1)

        candidate_ctr = input_line['candidate_ctr']
        item_conformity = self.conformity_encoder(candidate_ctr)

        if is_train:
            weight = F.tanh(item_quality + item_conformity)
        else:
            weight = F.tanh(item_quality)

        final_score = torch.multiply(weight, F.softplus(matching_score))

        return [final_score, matching_score, weight]


class DICE(basic_module):
    def __init__(self, args):
        super(DICE, self).__init__()
        self._args = args
        self.backbone_interest = NRMS_backbone(args)
        self.backbone_conformity = NRMS_backbone(args)

        self.vec_size = self.backbone_interest.get_output_size()
        self.req_grad_params = self.get_req_grad_params(debug=self._args.debug)

    def forward(self, input_line, is_train=True):
        u_interest_vec, c_interest_vecs, matching_interest_score = self.backbone_interest(input_line)
        u_conformity_vec, c_conformity_vecs, matching_conformity_score = self.backbone_conformity(input_line)

        final_u = torch.cat([u_interest_vec, u_conformity_vec], dim=-1)
        final_c = torch.cat([c_interest_vecs, c_conformity_vecs], dim=-1)

        final_score = torch.bmm(final_c, final_u.unsqueeze(-1)).squeeze(dim=-1)

        representations = [u_interest_vec, c_interest_vecs, u_conformity_vec, c_conformity_vecs]

        return [final_score, matching_interest_score, matching_conformity_score, representations]
