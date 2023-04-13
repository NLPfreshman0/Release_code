import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer
import numpy as np

class MLPLayer(nn.Module):
    def __init__(self, args, config, is_fusion=False):
        super().__init__()
        if args.task_name == 'GTE' and args.fusion == 'CON':
            if is_fusion:
                if args.confactual_train:
                    self.dense = nn.Linear(config.hidden_size * 3, config.hidden_size)
                else:
                    self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
            else:
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
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
        tokenizer = AutoTokenizer.from_pretrained(args.bert_dir, use_fast=True)
        self.bert = AutoModel.from_pretrained(args.bert_dir, config=config)
        self.pool_type = args.pool_type
        self.args = args
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.1)
        if args.confactual_train:
            self.VH_mlp = MLPLayer(args, config)
            self.VH_classifier = nn.Linear(config.hidden_size, 3)
            self.PH_mlp = MLPLayer(args, config)
            self.PH_classifier = nn.Linear(config.hidden_size, 3)
            self.H_mlp = MLPLayer(args, config)
            self.H_classifier = nn.Linear(config.hidden_size, 3)
            self.main_mlp = MLPLayer(args, config, is_fusion=True)
            self.main_classifier = nn.Linear(config.hidden_size, 3)
            self.confactual_text = torch.LongTensor([[tokenizer.mask_token_id for i in range(2 * args.bert_seq_length)]])
            self.confactual_image = torch.zeros(50, 768).unsqueeze(0)
        else:
            if args.train_mode == 'GTE':
                self.mlp = MLPLayer(args, config, is_fusion=True)
            else:
                self.mlp = MLPLayer(args, config)
            self.classifier = nn.Linear(config.hidden_size, 3)
    
    def forward(self, inputs, inference=False):
        if self.args.task_name == 'GTE'：
            if self.args.confactual_train:
                labels = inputs['label']
                image_embedding = inputs['image']
                hypothesis_embedding = self.bert.embeddings(inputs['text']['hypothesis']['input_ids'].squeeze(1))
                com_embedding = self.bert(inputs['text']['con_text']['input_ids'].squeeze(1), inputs['text']['con_text']['attention_mask'])
                image_embedding = self.bert.embeddings(inputs_embeds=image_embedding)
            
                cls_embedding = hypothesis_embedding[:, 0:1, :]
                image_embedding = image_embedding[:, 1:, :]
                hypothesis_embedding = hypothesis_embedding[:, 1:, :]
        
                VH_embedding = torch.cat([cls_embedding, image_embedding, hypothesis_embedding], dim=1)
                VH_attention_mask = torch.cat([torch.ones(cls_embedding.shape[0], 49).cuda(), inputs['text']['hypothesis']['attention_mask'].squeeze(1)], dim=1)
                attention_mask = VH_attention_mask[:, None, None, :]
                attention_mask = (1.0 - attention_mask) * -10000.0
                VH_output = self.bert.encoder(VH_embedding, attention_mask=attention_mask)["last_hidden_state"]
                PH_output = com_embedding["last_hidden_state"]
                H_output = self.bert(inputs['text']['hypothesis']['input_ids'].squeeze(1), inputs['text']['hypothesis']['attention_mask'].squeeze(1))["last_hidden_state"]
    
                if self.args.pool_type == 'cls':
                    VH_emb = VH_output[:, 0, :]
                    PH_emb = PH_output[:, 0, :]
                    H_emb = H_output[:, 0, :]
                if self.args.fusion == 'CON':
                    fusion_emb = torch.cat([VH_emb, PH_emb, H_emb], dim=1)
                else:
                    fusion_emb = VH_emb + PH_emb + H_emb
                    fusion_emb = torch.sigmoid(fusion_emb) 
                
                VH_emb = self.VH_mlp(VH_emb)
                PH_emb = self.PH_mlp(PH_emb)
                H_emb = self.H_mlp(H_emb)
                fusion_emb = self.main_mlp(fusion_emb)
                VH_logits = self.VH_classifier(VH_emb)
                VH_logits = F.softmax(VH_logits, dim=-1)
                PH_logits = self.PH_classifier(PH_emb)
                PH_logits = F.softmax(PH_logits, dim=-1)
                H_logits = self.H_classifier(H_emb)
                H_logits = F.softmax(H_logits, dim=-1)
                logits = self.main_classifier(fusion_emb)
                logits = F.softmax(logits, dim=-1)
                main_loss = self.calc_loss(logits, labels)
                VH_loss = self.calc_loss(VH_logits, labels)
                PH_loss = self.calc_loss(PH_logits, labels)
                H_loss = self.calc_loss(H_logits, labels)
                loss = main_loss + 0.5 * VH_loss + 0.5 * PH_loss + H_loss
                
                if inference:
                    if self.args.lack_visual:
                        PH_logits = self.PH_classifier(PH_emb)
                        PH_logits = F.softmax(PH_logits, dim=-1)
                        con_PH_emb = torch.zeros(cls_embedding.shape[0], 768).cuda()
                        con_PH_emb = self.PH_mlp(con_PH_emb)
                        con_PH_logits = self.PH_classifier(con_PH_emb)
                        con_PH_logits = F.softmax(con_PH_logits, dim=-1) 
                        logits = torch.sigmoid(PH_logits + H_logits)
                        con_logits = torch.sigmoid(con_PH_logits + H_logits)
                        return logits, con_logits, torch.argmax(logits, dim=1), labels
                    
                    elif self.args.lack_text:
                          VH_logits = self.VH_classifier(VH_emb)
                          VH_logits = F.softmax(VH_logits, dim=-1)
                          con_VH_emb = torch.zeros(cls_embedding.shape[0], 768).cuda()
                          con_VH_emb = self.VH_mlp(con_VH_emb)
                          con_VH_logits = self.VH_classifier(con_VH_emb)
                          con_VH_logits = F.softmax(con_VH_logits, dim=-1) 
                          logits = torch.sigmoid(VH_logits + H_logits)
                          con_logits = torch.sigmoid(con_VH_logits + H_logits)
                          return logits, con_logits, torch.argmax(logits, dim=1), labels
                        
                    if self.args.pool_type == 'cls':
                        con_VH_emb = torch.zeros(cls_embedding.shape[0], 768).cuda()
                        con_PH_emb = torch.zeros(cls_embedding.shape[0], 768).cuda()
                        #con_VH_emb = con_VH_output[:, 0, :]
                        #con_PH_emb = con_PH_output[:, 0, :]
                        H_emb = H_output[:, 0, :]

                    if self.args.fusion == 'CON':
                        con_fusion_emb = torch.cat([con_VH_emb, con_PH_emb, H_emb], dim=1)
                    else:
                        con_fusion_emb = con_VH_emb + con_PH_emb + H_emb
                        con_fusion_emb = torch.sigmoid(con_fusion_emb)
                    con_fusion_emb = self.main_mlp(con_fusion_emb) 
                    con_logits = self.main_classifier(con_fusion_emb)
                    con_logits = F.softmax(con_logits, dim=-1)
                    return logits, con_logits, torch.argmax(logits, dim=1), labels
                    
                if self.args.use_CL:
                    premise_emb = self.bert(inputs['text']['premise']['input_ids'].squeeze(1), inputs['text']['premise']['attention_mask'].squeeze(1))
                    premise_cls = premise_emb["last_hidden_state"][:, 0, :]
                    image_emb = self.bert(inputs_embeds=inputs['image'], attention_mask=torch.ones(cls_embedding.shape[0], 50).cuda())
                    image_cls = image_emb["last_hidden_state"][:, 0, :]
                    CL_loss = self.calc_itc_loss(premise_cls, image_cls)
                    return loss + CL_loss
                else:
                    return loss
                
            if self.args.train_mode == 'TE':
                labels = inputs['label']
                if self.args.only_hy:
                    hy_embedding = self.bert(inputs['text']['hypothesis']['input_ids'].squeeze(1), inputs['text']['hypothesis']['attention_mask'])
                    output = hy_embedding["last_hidden_state"]
                else:
                    con_embedding = self.bert(inputs['text']['con_text']['input_ids'].squeeze(1), inputs['text']['con_text']['attention_mask'])
                    output = con_embedding["last_hidden_state"]
                if self.args.pool_type == 'cls':
                    emb = output[:, 0, :]
                    
            elif self.args.train_mode == 'VE':
                labels = inputs['label']
                image_embedding = inputs['image']
                #image_embedding = self.dense(image_embedding).unsqueeze(1)
                hypothesis_embedding = self.bert.embeddings(inputs['text']['hypothesis']['input_ids'].squeeze(1))
                image_embedding = self.bert.embeddings(inputs_embeds=image_embedding)
            
                cls_embedding = hypothesis_embedding[:, 0:1, :]
                image_embedding = image_embedding[:, 1:, :]
                hypothesis_embedding = hypothesis_embedding[:, 1:, :]
        
                VH_embedding = torch.cat([cls_embedding, image_embedding, hypothesis_embedding], dim=1)
                VH_attention_mask = torch.cat([torch.ones(cls_embedding.shape[0], 49).cuda(), inputs['text']['hypothesis']['attention_mask'].squeeze(1)], dim=1)
                attention_mask = VH_attention_mask[:, None, None, :]
                attention_mask = (1.0 - attention_mask) * -10000.0
                VH_output = self.bert.encoder(VH_embedding, attention_mask=attention_mask)["last_hidden_state"]
                if self.args.pool_type == 'cls':
                    emb = VH_output[:, 0, :]
                elif self.args.pool_type == 'mean':
                    emb = (VH_output * VH_attention_mask.squeeze(1).unsqueeze(-1)).sum(1) / VH_attention_mask.squeeze(1).sum(-1).unsqueeze(-1)
                    
            else: 
                labels = inputs['label']
                image_embedding = inputs['image']
                #image_embedding = self.dense(image_embedding).unsqueeze(1)
                hypothesis_embedding = self.bert.embeddings(inputs['text']['hypothesis']['input_ids'].squeeze(1))
                con_embedding = self.bert(inputs['text']['con_text']['input_ids'].squeeze(1), inputs['text']['con_text']['attention_mask'])
                image_embedding = self.bert.embeddings(inputs_embeds=image_embedding)
            
                cls_embedding = hypothesis_embedding[:, 0:1, :]
                image_embedding = image_embedding[:, 1:, :]
                hypothesis_embedding = hypothesis_embedding[:, 1:, :]
        
                VH_embedding = torch.cat([cls_embedding, image_embedding, hypothesis_embedding], dim=1)
                VH_attention_mask = torch.cat([torch.ones(cls_embedding.shape[0], 49).cuda(), inputs['text']['hypothesis']['attention_mask'].squeeze(1)], dim=1)
                attention_mask = VH_attention_mask[:, None, None, :]
                attention_mask = (1.0 - attention_mask) * -10000.0
                VH_output = self.bert.encoder(VH_embedding, attention_mask=attention_mask)["last_hidden_state"]
                PH_output = con_embedding["last_hidden_state"]
        
    
                if self.args.pool_type == 'cls':
                    VH_emb = VH_output[:, 0, :]
                    PH_emb = PH_output[:, 0, :]
                if self.args.fusion == 'CON':
                    emb = torch.cat([VH_emb, PH_emb], dim=1)
                else:
                    emb = VH_emb + PH_emb
                    emb = torch.sigmoid(emb) 
            
            emb = self.mlp(emb)
            logits = self.classifier(emb)
            logits = F.softmax(logits, dim=-1)
            if inference:
                return logits, torch.argmax(logits, dim=1), labels
            if self.args.use_CL:
                premise_emb = self.bert(inputs['text']['premise']['input_ids'].squeeze(1), inputs['text']['premise']['attention_mask'].squeeze(1))
                premise_cls = premise_emb["last_hidden_state"][:, 0, :]
                image_emb = self.bert(inputs_embeds=inputs['image'], attention_mask=torch.ones(cls_embedding.shape[0], 50).cuda())
                image_cls = image_emb["last_hidden_state"][:, 0, :]
                CL_loss = self.calc_itc_loss(premise_cls, image_cls)
                return self.calc_loss(logits, labels) + CL_loss
            else:
                return self.calc_loss(logits, labels)
                
    
    def calc_loss(self, logits, labels):
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).sum() / len(labels)
        return loss
    
    def calc_itc_loss(self, premise_cls, image_cls, lamda=0.05):
        similarities = F.cosine_similarity(premise_cls.unsqueeze(1), image_cls.unsqueeze(0), dim=-1)
        similarities /= lamda
        labels = torch.arange(similarities.shape[0]).cuda()
        loss = F.cross_entropy(similarities, labels)
        return torch.mean(loss)
