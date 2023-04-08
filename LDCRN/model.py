import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
import numpy as np

class MLPLayer(nn.Module):
    def __init__(self, args, config, use_con=False):
        super().__init__()
        if use_con:
            self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        config = AutoConfig.from_pretrained(args.bert_dir)
        self.bert = AutoModel.from_pretrained(args.bert_dir, config=config)
        self.pool_type = args.pool_type
        self.args = args
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.1)
        if args.confactual_train:
            self.ph_mlp = MLPLayer(args, config)
            self.ph_classifier = nn.Linear(config.hidden_size, 3)
            self.h_mlp = MLPLayer(args, config)
            self.h_classifier = nn.Linear(config.hidden_size, 3)
            if args.fusion == 'CON':
                self.main_mlp = MLPLayer(args, config, use_con=True)
                self.main_classifier = nn.Linear(config.hidden_size, 3)
            else:
                self.main_mlp = MLPLayer(args, config)
                self.main_classifier = nn.Linear(config.hidden_size, 3)
        else:
            self.mlp = MLPLayer(args, config)
            self.classifier = nn.Linear(config.hidden_size, 3)
    
    def forward(self, inputs, inference=False):
        if self.args.task_name == 'SNLI' or 'MultiNLI':
            labels = inputs['label']
            if not inference and (self.args.only_pre or self.args.only_hy):
                output = self.bert(inputs['input_ids'].reshape(-1, inputs['input_ids'].shape[-1]), inputs['attention_mask'].reshape(-1, inputs['attention_mask'].shape[-1]))
                return output.last_hidden_state[:, 0, :], labels
            if self.args.confactual_train:
                factual_input_ids = inputs['factual_text']['input_ids'].reshape(-1, inputs['factual_text']['input_ids'].shape[-1])
                factual_attention_mask = inputs['factual_text']['attention_mask'].reshape(-1, inputs['factual_text']['attention_mask'].shape[-1])
                confactual_input_ids = inputs['confactual_text']['input_ids'].reshape(-1, inputs['confactual_text']['input_ids'].shape[-1])
                confactual_attention_mask = inputs['confactual_text']['attention_mask'].reshape(-1, inputs['confactual_text']['attention_mask'].shape[-1])
                factual_output = self.bert(factual_input_ids, factual_attention_mask)
                if self.args.use_mean_pre:
                    if self.args.task_name == 'SNLI':
                        mean_pre = torch.Tensor(np.load('data/snli_all_mean.npy', allow_pickle=True).reshape(1, 1, -1)).cuda()
                        mean_en = torch.Tensor(np.load('data/bert_SNLI_entailment.npy', allow_pickle=True).reshape(1, 1, -1)).cuda()
                        mean_ne = torch.Tensor(np.load('data/bert_SNLI_neutral.npy', allow_pickle=True).reshape(1, 1, -1)).cuda()
                        mean_con = torch.Tensor(np.load('data/bert_SNLI_contradiction.npy', allow_pickle=True).reshape(1, 1, -1)).cuda()
                        mean_label = torch.cat([mean_en, mean_ne, mean_con], dim=1)
                    elif self.args.task_name == 'MultiNLI':
                        mean_pre = torch.Tensor(np.load('data/multinli_all_mean.npy', allow_pickle=True).reshape(1, 1, -1)).cuda()
                        mean_en = torch.Tensor(np.load('data/bert_MultiNLI_entailment.npy', allow_pickle=True).reshape(1, 1, -1)).cuda()
                        mean_ne = torch.Tensor(np.load('data/bert_MultiNLI_neutral.npy', allow_pickle=True).reshape(1, 1, -1)).cuda()
                        mean_con = torch.Tensor(np.load('data/bert_MultiNLI_contradiction.npy', allow_pickle=True).reshape(1, 1, -1)).cuda()
                        mean_label = torch.cat([mean_en, mean_ne, mean_con], dim=1)
                    confactual_emb = self.bert.embeddings(confactual_input_ids)
                    cls_emb = confactual_emb[:, 0:1, :]
                    hy_emb = confactual_emb[:, 1:, :]
                    label_emb = mean_label.repeat(cls_emb.shape[0], 1, 1)
                    confactual_emb = torch.cat([cls_emb, label_emb, hy_emb], dim=1)
                    confactual_attention_mask = torch.cat([torch.ones(cls_emb.shape[0], 3).cuda(), confactual_attention_mask], dim=1)
                    confactual_attention_mask = confactual_attention_mask[:, None, None, :]
                    attention_mask = (1.0 - confactual_attention_mask) * -10000.0
                    confactual_output = self.bert.encoder(confactual_emb, attention_mask)
                else:
                    confactual_output = self.bert(confactual_input_ids, confactual_attention_mask)
                if self.args.pool_type == 'cls':
                    Y_phm = factual_output.last_hidden_state[:, 0, :]
                    Y_hc = confactual_output.last_hidden_state[:, 0, :]
                elif self.args.pool_type == 'pooler':
                    factual_emb = factual_output.pooler_output
                    confactual_emb = confactual_output.pooler_output
                elif self.args.pool_type == 'mean':
                    factual_emb = ((factual_output.last_hidden_state * inputs['factual_text']['attention_mask'].unsqueeze(-1)).sum(1) / inputs['factual_text']['attention_mask'].sum(-1).unsqueeze(-1))
                    confactual_emb = ((confactual_output.last_hidden_state * inputs['confactual_text']['attention_mask'].unsqueeze(-1)).sum(1) / inputs['confactual_text']['attention_mask'].sum(-1).unsqueeze(-1))
                factual_emb = self.ph_mlp(Y_phm)
                confactual_emb = self.h_mlp(Y_hc)
                factual_logits = self.ph_classifier(factual_emb)
                confactual_logits = self.h_classifier(confactual_emb)
                if self.args.fusion == 'CON':
                    Y_phmc = torch.cat([Y_phm, Y_hc], dim=1)
                else:
                    Y_phmc = Y_phm + Y_hc 
                fusion_emb = self.main_mlp(Y_phmc)
                logits = self.main_classifier(fusion_emb)
                if inference:
                    con_emb = self.bert.embeddings(confactual_input_ids)
                    cls_emb = con_emb[:, 0:1, :]
                    hy_emb = con_emb[:, 1:, :]
                    pre_emb = mean_pre.repeat(cls_emb.shape[0], 1, 1)
                    con_emb = torch.cat([cls_emb, pre_emb, hy_emb], dim=1)
                    con_output = self.bert.encoder(con_emb, attention_mask)
                    con_Y_phm = con_output.last_hidden_state[:, 0, :]
                    if self.args.fusion == 'CON':
                        Y_phmc = torch.cat([Y_phm, Y_hc], dim=1)
                        con_Y_phmc = torch.cat([con_Y_phm, Y_hc], dim=1)
                    else:
                        Y_phmc = Y_phm + Y_hc 
                        con_Y_phmc = con_Y_phm + Y_hc 
                    logits = self.main_classifier(self.main_mlp(Y_phmc))
                    con_logits = self.main_classifier(self.main_mlp(con_Y_phmc))
                    return logits, con_logits, torch.argmax(logits, dim=1), labels
                main_loss = self.calc_loss(logits, labels)
                loss1 = self.calc_loss(factual_logits, labels)
                loss2 = self.calc_loss(confactual_logits, labels)
                a1 = 0.5
                a2 = 0.5
                loss =  main_loss + a1 * loss1 + a2 * loss2
                return loss

            else:
                input_ids = inputs['input_ids'].reshape(-1, inputs['input_ids'].shape[-1])
                attention_mask = inputs['attention_mask'].reshape(-1, inputs['attention_mask'].shape[-1])
                output = self.bert(input_ids, attention_mask)
                if self.args.pool_type == 'cls':
                    emb = output.last_hidden_state[:, 0, :]
                elif self.args.pool_type == 'pooler':
                    emb = output.pooler_output
                elif self.args.pool_type == 'mean':
                    emb = (output.last_hidden_state * attention_mask.squeeze(1).unsqueeze(-1)).sum(1) / attention_mask.squeeze(1).sum(-1).unsqueeze(-1)
                elif self.args.pool_type == 'avg_first_last':
                    first_hidden = output.hidden_states[0]
                    last_hidden = output.hidden_states[-1]
                    emb = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) /attention_mask.sum(-1).unsqueeze(-1)

            emb = self.mlp(emb)
            logits = self.classifier(emb)
            logits = F.softmax(logits, dim=-1)
            if inference:
                return logits, torch.argmax(logits, dim=1), labels
            return self.calc_loss(logits, labels)
    
    def calc_loss(self, logits, labels):
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).sum() / len(labels)
        return loss
        
    def get_emb(self, input_ids, attention_mask):
        return self.bert(input_ids, attention_mask)['last_hidden_state'][:, 0, :]
        
        




