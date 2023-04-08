import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
import numpy as np

class MLPLayer(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        if args.pair:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            self.dense = nn.Linear(config.hidden_size * 4, config.hidden_size)
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
        if args.task_name == 'SNLI-VE':
            self.dense = nn.Linear(2048, config.hidden_size)
        if args.confactual_train:
            self.ph_mlp = MLPLayer(args, config)
            self.ph_classifier = nn.Linear(config.hidden_size, 3)
            self.h_mlp = MLPLayer(args, config)
            self.h_classifier = nn.Linear(config.hidden_size, 3)
        else:
            self.mlp = MLPLayer(args, config)
            self.classifier = nn.Linear(config.hidden_size, 3)
        """if args.pair:
            self.classifier = nn.Linear(config.hidden_size, 3)
        else:
            self.classifier = nn.Linear(config.hidden_size * 4, 3)"""
    
    def forward(self, inputs, inference=False):
        if self.args.task_name == 'SNLI' or 'MultiNLI':
            labels = inputs['label']
            if not inference and (self.args.only_pre or self.args.only_hy):
                output = self.bert(inputs['input_ids'].reshape(-1, inputs['input_ids'].shape[-1]), inputs['attention_mask'].reshape(-1, inputs['attention_mask'].shape[-1]))
                return output.last_hidden_state[:, 0, :], labels
            if self.args.confactual_train:
                if self.args.pair:
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
                            mean_label = (mean_en + mean_ne + mean_con) / 3
                        elif self.args.task_name == 'MultiNLI':
                            mean_pre = torch.Tensor(np.load('data/multinli_all_mean.npy', allow_pickle=True).reshape(1, 1, -1)).cuda()
                            mean_en = torch.Tensor(np.load('data/bert_MultiNLI_entailment.npy', allow_pickle=True).reshape(1, 1, -1)).cuda()
                            mean_ne = torch.Tensor(np.load('data/bert_MultiNLI_neutral.npy', allow_pickle=True).reshape(1, 1, -1)).cuda()
                            mean_con = torch.Tensor(np.load('data/bert_MultiNLI_contradiction.npy', allow_pickle=True).reshape(1, 1, -1)).cuda()
                            mean_label = (mean_en + mean_ne + mean_con) / 3
                        confactual_emb = self.bert.embeddings(confactual_input_ids)
                        cls_emb = confactual_emb[:, 0:1, :]
                        hy_emb = confactual_emb[:, 1:, :]
                        pre_emb = mean_pre.repeat(cls_emb.shape[0], 1, 1)
                        label_emb = mean_label.repeat(cls_emb.shape[0], 1, 1)
                        confactual_emb = torch.cat([cls_emb, pre_emb, label_emb, hy_emb], dim=1)
                        confactual_attention_mask = torch.cat([torch.ones(cls_emb.shape[0], 2).cuda(), confactual_attention_mask], dim=1)
                        confactual_attention_mask = confactual_attention_mask[:, None, None, :]
                        attention_mask = (1.0 - confactual_attention_mask) * -10000.0
                        confactual_output = self.bert.encoder(confactual_emb, attention_mask)
                    else:
                        confactual_output = self.bert(confactual_input_ids, confactual_attention_mask)
                    if self.args.pool_type == 'cls':
                        factual_emb = factual_output.last_hidden_state[:, 0, :]
                        confactual_emb = confactual_output.last_hidden_state[:, 0, :]
                    elif self.args.pool_type == 'pooler':
                        factual_emb = factual_output.pooler_output
                        confactual_emb = confactual_output.pooler_output
                    elif self.args.pool_type == 'mean':
                        factual_emb = ((factual_output.last_hidden_state * inputs['factual_text']['attention_mask'].unsqueeze(-1)).sum(1) / inputs['factual_text']['attention_mask'].sum(-1).unsqueeze(-1))
                        confactual_emb = ((confactual_output.last_hidden_state * inputs['confactual_text']['attention_mask'].unsqueeze(-1)).sum(1) / inputs['confactual_text']['attention_mask'].sum(-1).unsqueeze(-1))
                    factual_emb = self.ph_mlp(factual_emb)
                    confactual_emb = self.h_mlp(confactual_emb)
                    factual_logits = self.ph_classifier(factual_emb)
                    confactual_logits = self.h_classifier(confactual_emb)
                    if inference:
                        return factual_logits, confactual_logits, torch.argmax(factual_logits, dim=1), labels
                    loss1 = self.calc_loss(factual_logits, labels)
                    loss2 = self.calc_loss(confactual_logits, labels)
                    """p_te = F.softmax(factual_logits, -1)
                    p_nde = F.softmax(confactual_logits, -1)
                    kl_loss = - p_te*p_nde.log()    
                    kl_loss = kl_loss.sum(1).mean()""" 
                    loss = loss1 + loss2
                    return loss
                    #return self.calc_confactual_loss(factual_logits, confactual_logits, labels)
                
            if self.args.pair:
                input_ids = inputs['input_ids'].reshape(-1, inputs['input_ids'].shape[-1])
                attention_mask = inputs['attention_mask'].reshape(-1, inputs['attention_mask'].shape[-1])
                output = self.bert(input_ids, attention_mask, output_hidden_states=True)
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
            else:
                output1 = self.bert(inputs['input_ids'][:, 0, :].squeeze(1), inputs['attention_mask'][:, 0, :].squeeze(1))
                output2 = self.bert(inputs['input_ids'][:, 1, :].squeeze(1), inputs['attention_mask'][:, 1, :].squeeze(1))
                if self.args.pool_type == 'cls':
                    emb1 = output1.last_hidden_state[:, 0, :]
                    emb2 = output2.last_hidden_state[:, 0, :]
                elif self.args.pool_type == 'pooler':
                    emb1 = output1.pooler_output
                    emb2 = output2.pooler_output
                elif self.args.pool_type == 'mean':
                    emb1 = ((output1.last_hidden_state * inputs['attention_mask'][:, 0, :].unsqueeze(-1)).sum(1) / inputs['attention_mask'][:, 0, :].sum(-1).unsqueeze(-1))
                    emb2 = ((output2.last_hidden_state * inputs['attention_mask'][:, 1, :].unsqueeze(-1)).sum(1) / inputs['attention_mask'][:, 1, :].sum(-1).unsqueeze(-1))
                emb = torch.cat([emb1, emb2, torch.abs(emb1-emb2), emb1*emb2], dim=1)
            
            #emb = self.dropout(emb)
            emb = self.mlp(emb)
            logits = self.classifier(emb)
            logits = F.softmax(logits, dim=-1)
            if inference:
                return logits, torch.argmax(logits, dim=1), labels
            return self.calc_loss(logits, labels)

        else:
            labels = inputs['label']
            image_embedding = inputs['image']
            image_embedding = self.dense(image_embedding)
            text_output = self.bert(inputs['text']['input_ids'].squeeze(1), inputs['text']['attention_mask'].squeeze(1))
            if self.args.pool_type == 'cls':
                text_embedding = text_output.last_hidden_state[:, 0, :]
            elif self.args.pool_type == 'pooler':
                text_embedding = text_output.pooler_output
            emb = torch.concat([image_embedding, text_embedding, torch.abs(image_embedding-text_embedding), image_embedding*text_embedding], dim=1)
            logits = self.classifier(emb)
            if inference:
                return torch.argmax(logits, dim=1), labels
            return self.calc_loss(logits, labels)
    
    def calc_loss(self, logits, labels):
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).sum() / len(labels)
        return loss
    
    def calc_confactual_loss(self, factual_logits, confactual_logits, labels):
        loss1 = F.cross_entropy(factual_logits, labels)
        p_te = F.softmax(factual_logits, -1)
        p_nde = F.softmax(confactual_logits, -1)
        kl_loss = - p_te*p_nde.log()    
        kl_loss = kl_loss.sum(1).mean() 
        loss = loss1 + kl_loss
        preds = torch.argmax(factual_logits, dim=1)
        accuracy = (preds == labels).sum().item() / len(labels)
        return loss
        
    def get_emb(self, input_ids, attention_mask):
        return self.bert(input_ids, attention_mask)['last_hidden_state'][:, 0, :]
        
        




