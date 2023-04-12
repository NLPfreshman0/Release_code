from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, accuracy_score, f1_score

import numpy as np
import os

import zipfile
import torch
from torch import nn

import torch.nn.functional as F
from torch.nn.parameter import Parameter

def mrr_score(y_true, y_score):
    """Computing mrr score metric.

    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_score (np.ndarray): Predicted labels.

    Returns:
        numpy.ndarray: mrr scores.
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def ndcg_score(y_true, y_score, k=10):
    """Computing ndcg score metric at k.

    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_score (np.ndarray): Predicted labels.

    Returns:
        numpy.ndarray: ndcg scores.
    """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def hit_score(y_true, y_score, k=10):
    """Computing hit score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: hit score.
    """
    ground_truth = np.where(y_true == 1)[0]
    argsort = np.argsort(y_score)[::-1][:k]
    for idx in argsort:
        if idx in ground_truth:
            return 1
    return 0


def dcg_score(y_true, y_score, k=10):
    """Computing dcg score metric at k.

    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_score (np.ndarray): Predicted labels.

    Returns:
        np.ndarray: dcg scores.
    """
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def cal_metric(labels, preds, metrics):
    """Calculate metrics.

    Available options are: `auc`, `rmse`, `logloss`, `acc` (accurary), `f1`, `mean_mrr`,
    `ndcg` (format like: ndcg@2;4;6;8), `hit` (format like: hit@2;4;6;8), `group_auc`.

    Args:
        labels (array-like): Labels.
        preds (array-like): Predictions.
        metrics (list): List of metric names.

    Return:
        dict: Metrics.

    Examples:
        cal_metric(labels, preds, ["ndcg@2;4;6", "group_auc"])
        {'ndcg@2': 0.4026, 'ndcg@4': 0.4953, 'ndcg@6': 0.5346, 'group_auc': 0.8096}

    """
    res = {}
    for metric in metrics:
        if metric == "auc":
            auc = roc_auc_score(np.asarray(labels), np.asarray(preds))
            res["auc"] = round(auc, 4)
        elif metric == "rmse":
            rmse = mean_squared_error(np.asarray(labels), np.asarray(preds))
            res["rmse"] = np.sqrt(round(rmse, 4))
        elif metric == "logloss":
            # avoid logloss nan
            preds = [max(min(p, 1.0 - 10e-12), 10e-12) for p in preds]
            logloss = log_loss(np.asarray(labels), np.asarray(preds))
            res["logloss"] = round(logloss, 4)
        elif metric == "acc":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            acc = accuracy_score(np.asarray(labels), pred)
            res["acc"] = round(acc, 4)
        elif metric == "f1":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            f1 = f1_score(np.asarray(labels), pred)
            res["f1"] = round(f1, 4)
        elif metric == "mean_mrr":
            mean_mrr = np.mean(
                [
                    mrr_score(each_labels, each_preds)
                    for each_labels, each_preds in zip(labels, preds)
                ]
            )
            res["mean_mrr"] = round(mean_mrr, 4)
        elif metric == "mrr":
            mean_mrr = mrr_score(labels, preds)
            res["mean_mrr"] = round(mean_mrr, 4)
        elif metric.startswith("ndcg"):  # format like:  ndcg@2;4;6;8
            ndcg_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                ndcg_list = [int(token) for token in ks[1].split(";")]
            for k in ndcg_list:
                ndcg_temp = ndcg_score(labels, preds, k)
                res["ndcg@{0}".format(k)] = round(ndcg_temp, 4)
        elif metric.startswith("hit"):  # format like:  hit@2;4;6;8
            hit_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                hit_list = [int(token) for token in ks[1].split(";")]
            for k in hit_list:
                hit_temp = hit_score(labels, preds, k)
                res["hit@{0}".format(k)] = round(hit_temp, 4)
        elif metric == "group_auc":
            group_auc = np.mean(
                [
                    roc_auc_score(each_labels, each_preds)
                    for each_labels, each_preds in zip(labels, preds)
                ]
            )
            res["group_auc"] = round(group_auc, 4)
        else:
            raise ValueError("Metric {0} not defined".format(metric))
    return res


class AdditiveAttention(nn.Module):
    ''' AttentionPooling used to weighted aggregate news vectors
    Arg:
        d_h: the last dimension of input
    '''

    def __init__(self, d_h, hidden_size=200):
        super(AdditiveAttention, self).__init__()
        self.att_fc1 = nn.Linear(d_h, hidden_size)
        self.att_fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: batch_size, candidate_size, candidate_vector_dim
            attn_mask: batch_size, candidate_size
        Returns:
            (shape) batch_size, candidate_vector_dim
        """
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)

        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)

        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))  # (bz, 400)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = torch.exp(scores)
        if attn_mask is not None:
            scores = scores * attn_mask
        attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)

        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model  # 300
        self.n_heads = n_heads  # 20
        self.d_k = d_k  # 20
        self.d_v = d_v  # 20

        self.W_Q = nn.Linear(d_model, d_k * n_heads)  # 300, 400
        self.W_K = nn.Linear(d_model, d_k * n_heads)  # 300, 400
        self.W_V = nn.Linear(d_model, d_v * n_heads)  # 300, 400

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, Q, K, V, attn_mask=None):
        """
        :param Q: [batch_size*number_sentence, seq_len, emb_size]
        :param K: [batch_size*number_sentence, seq_len, emb_size]
        :param V: [batch_size*number_sentence, seq_len, emb_size]
        :param attn_mask: [batch_size*number_sentence, seq_len, 1]
        :return: [batch_size*number_sentence, seq_len, n_heads*d_v]
        """
        residual = Q
        batch_size, seq_len = Q.shape[:2]

        # shape = [batch_size*num_sentence, seq_len, n_heads, d_k]
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        context, attn = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        return context


class ConformtyCal(nn.Module):
    def __init__(self, args, emb_size, hyper=None):
        self._args = args
        self._hyper = hyper
        self._emb_size = emb_size
        super(ConformtyCal, self).__init__()

        # for news
        self.candidate_conformity = nn.init.normal(Parameter(torch.Tensor(1,)))
        # for users
        self.his_conformity = nn.Sequential(
            nn.Linear(in_features=self._args.his_size, out_features=1),
            nn.Sigmoid(),
        )
        self.users_conformity = nn.Sequential(
            nn.Linear(in_features=self._emb_size, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, candidate_ctr=None, history_ctr=None, user_kl=None, hyper_alpha=1.0):
        """
        :param candidate_ctr: [batch_size, candidate_num]
        :param history_ctr: [batch_size, his_size]
        :param user_kl: [batch_size, 2] 或者 [batch_size, ]
        :return: []: shape:[batch_size, candidate_num]
        """
        if candidate_ctr is None:
            weight = self.users_conformity(user_kl)
            user_side = torch.mean(torch.multiply(weight, user_kl), dim=-1, keepdim=True)
            news_side = self.his_conformity(history_ctr)
            weighted_info = user_side + torch.multiply((1-weight), news_side)
            conformity = torch.div(news_side, 1 + weighted_info)
        elif user_kl is None:
            weight = torch.multiply(self.candidate_conformity, candidate_ctr)
            conformity = weight
        else:
        
            candidate_weight = torch.multiply(self.candidate_conformity, candidate_ctr)
            candidate_conformity = candidate_weight

            news_side = self.his_conformity(history_ctr)
            user_weight = self.users_conformity(user_kl)
            user_side = torch.mean(torch.multiply(user_weight, user_kl), dim=-1, keepdim=True)
            weighted_info = user_side + torch.multiply((1-user_weight), news_side)
            user_conformity = torch.div(news_side, 1+weighted_info)

            conformity = torch.add(candidate_conformity, hyper_alpha * user_conformity)

        return conformity


class ConformtyCalv2(nn.Module):
    def __init__(self, args, emb_size, hyper=None):
        self._args = args
        self._hyper = hyper
        self._emb_size = emb_size
        super(ConformtyCalv2, self).__init__()

        self.candidate_conformity = nn.init.normal(Parameter(torch.Tensor(1,)))
        self.his_conformity = nn.Sequential(
            nn.Linear(in_features=self._args.his_size, out_features=1),
            nn.Softplus(),
        )
        self.users_conformity = nn.Sequential(
            nn.Linear(in_features=self._emb_size, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, candidate_ctr=None, history_ctr=None, user_kl=None, fuse='add', hyper_alpha=1.0):
        """
        :param candidate_ctr: [batch_size, candidate_num]
        :param history_ctr: [batch_size, his_size]
        :param user_kl: [batch_size, 2] 或者 [batch_size, ]
        :return: []: shape:[batch_size, candidate_num]
        """
        def _fuse_func(news_side, user_side, weight):
            if fuse == 'add':
                w_users = torch.mean(torch.multiply(weight, user_side), dim=-1, keepdim=True)
                w_news = torch.multiply((1-weight), news_side)
                results = w_users + w_news
            elif fuse == 'div':
                w_users = torch.mean(torch.multiply(weight, user_side), dim=-1, keepdim=True)
                results = torch.div(news_side, 1 + w_users)
            else:
                raise ValueError(f'[Error!] wrong fusing selection, please try again')

            return results

        if candidate_ctr is None:
            weight = self.users_conformity(user_kl)
            news_side = self.his_conformity(history_ctr)
            conformity = _fuse_func(user_side=user_kl, news_side=news_side, weight=weight)
        elif user_kl is None:
            weight = torch.multiply(self.candidate_conformity, candidate_ctr)
            weight = F.sigmoid(weight)
            conformity = weight
        else:
            candidate_weight = torch.multiply(self.candidate_conformity, candidate_ctr)
            candidate_weight = F.sigmoid(candidate_weight)
            candidate_conformity = candidate_weight

            news_side = self.his_conformity(history_ctr)
            user_weight = self.users_conformity(user_kl)
            user_conformity = _fuse_func(news_side=news_side, user_side=user_kl, weight=user_weight)

            conformity = torch.add(candidate_conformity, hyper_alpha * user_conformity)
            conformity = [conformity, candidate_conformity]

        return conformity


class ConformtyCalv3(nn.Module):
    def __init__(self, args, emb_size, news_prob_size=1, hyper=None):
        self._args = args
        self._news_prob_size = news_prob_size
        self._hyper = hyper
        self._emb_size = emb_size
        super(ConformtyCalv3, self).__init__()

        self.candidate_conformity = nn.Sequential(
            nn.Linear(in_features=self._news_prob_size, out_features=1),
            nn.Sigmoid()
        )

        self.his_conformity = nn.Sequential(
            nn.Linear(in_features=self._news_prob_size, out_features=1),
            nn.Sigmoid(),
        )
        self.users_conformity = nn.Sequential(
            nn.Linear(in_features=self._emb_size, out_features=1),
            nn.Sigmoid(),
        )

        self.weight = nn.Sequential(
            nn.Linear(in_features=self._emb_size, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, candidate_ctr=None, history_ctr=None, user_kl=None, fuse='add', hyper_alpha=1.0):
        """
        :param candidate_ctr: [batch_size, candidate_num]
        :param history_ctr: [batch_size, his_size]
        :param user_kl: [batch_size, 2] 或者 [batch_size, ]
        :return: []: shape:[batch_size, candidate_num]
        """
        def _fuse_func(news_side, user_side, weight):
            w_users = torch.multiply(1-weight, user_side)
            w_news = torch.multiply(weight, news_side)
            if fuse == 'add':
                results = w_users + 1-w_news
            elif fuse == 'div':
                results = torch.div(w_news, 1 - w_users)
            else:
                raise ValueError(f'[Error!] wrong fusing selection, please try again')

            return results

        if candidate_ctr is None:
            weight = self.weight(user_kl)
            news_side = self.his_conformity(history_ctr)
            news_side = torch.mean(torch.squeeze(news_side), dim=-1, keepdim=True)
            user_side = self.users_conformity(user_kl)
            conformity = _fuse_func(user_side=user_side, news_side=news_side, weight=weight)
        elif user_kl is None:
            candidate_weight = self.candidate_conformity(candidate_ctr)
            conformity = candidate_weight
        else:
            candidate_weight = self.candidate_conformity(candidate_ctr)
            candidate_conformity = torch.squeeze(candidate_weight)
            news_side = self.his_conformity(history_ctr)
            news_side = torch.mean(torch.squeeze(news_side), dim=-1, keepdim=True)
            user_side = self.users_conformity(user_kl)
  
            user_weight = self.weight(user_kl)
            user_conformity = _fuse_func(news_side=news_side, user_side=user_side, weight=user_weight)

            conformity = torch.div(candidate_conformity, hyper_alpha * user_conformity)
            conformity = [conformity, candidate_conformity]

        return conformity


def write_file(file_path, file_name, content):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    with open(os.path.join(file_path, file_name), 'a', encoding='utf8') as w:
        if isinstance(content, list):
            for item in content:
                if isinstance(item, np.ndarray):
                    np.savetxt(w, item, fmt='%.4f', delimiter=', ')
                    w.write('\n')
                else:
                    w.write(item + '\n')
        elif isinstance(content, dict):
            for key, item in content.items():
                line = key + ':' + str(item)
                w.write(line + '\n')
        else:
            w.write(content + '\n')

    print(f'> content has been written into {file_path}')


def write_zip_file(file_path, file_name, content):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    with open(os.path.join(file_path, 'prediction.txt'), 'w', encoding='utf8') as w:
        if isinstance(content, list):
            for item in content:
                if isinstance(item, np.ndarray):
                    np.savetxt(w, item, fmt='%.4f', delimiter=', ')
                    w.write('\n')
                else:
                    w.write(item + '\n')
        elif isinstance(content, dict):
            for key, item in content.items():
                line = key + ':' + str(item)
                w.write(line + '\n')
        else:
            w.write(content + '\n')

    zip_f = zipfile.ZipFile(os.path.join(file_path, file_name), 'w', zipfile.ZIP_DEFLATED)
    zip_f.write(os.path.join(file_path, 'prediction.txt'), arcname='prediction.txt')
    zip_f.close()

    print(f'> content has been written into {file_path}')


def calculate_log_softmax(score, softmax_func='softmax'):
    shape = score.shape
    if softmax_func == 'softmax':
        func = nn.LogSoftmax(dim=1)
    elif softmax_func == 'softplus':
        func = nn.Softplus()
    else:
        raise ValueError(f'[Error] wrong softmax func selection {softmax_func}')
    if len(shape) == 2:
        value = func(score)
    else:
        raise ValueError(f'[Error] wrong input score.shape={shape}, please try again')

    return value


def elu(x: torch.Tensor):
    return torch.where(x.less(0), x.exp(), x.add(1))


class MyELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return elu(x)


class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, predictions, labels=None):
        positive_score = predictions[:, 0].unsqueeze(dim=1)
        negative_score = predictions[:, 1:]

        results = -torch.sum(torch.log(F.sigmoid(positive_score - negative_score)+1e-8), dim=1)
        results = torch.mean(results)

        return results


class DcorLoss(nn.Module):
    def __init__(self):
        super(DcorLoss, self).__init__()

    def forward(self, x, y):
        a = torch.norm(x[:, None] - x, p=2, dim=2)
        b = torch.norm(y[:, None] - y, p=2, dim=2)

        A = a - a.mean(dim=0)[None, :] - a.mean(dim=1)[:, None] + a.mean()
        B = b - b.mean(dim=0)[None, :] - b.mean(dim=1)[:, None] + b.mean()

        n = x.size(0)

        dcov2_xy = (A * B).sum() / float(n * n)
        dcov2_xx = (A * A).sum() / float(n * n)
        dcov2_yy = (B * B).sum() / float(n * n)
        dcor = -torch.sqrt(dcov2_xy) / torch.sqrt(torch.sqrt(dcov2_xx) * torch.sqrt(dcov2_yy))

        return dcor
