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
        if args.pair:
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
        tokenizer = AutoTokenizer.from_pretrained(args.bert_dir, use_fast=True)
        self.bert = AutoModel.from_pretrained(args.bert_dir, config=config)
        self.pool_type = args.pool_type
        self.args = args
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.1)
        if args.task_name == 'SNLI-VE' or args.task_name == 'GTE':
            self.dense = nn.Linear(2048, config.hidden_size)
        if args.confactual_train:
            self.ph_mlp = MLPLayer(args, config)
            self.ph_classifier = nn.Linear(config.hidden_size, 3)
            self.h_mlp = MLPLayer(args, config)
            self.h_classifier = nn.Linear(config.hidden_size, 3)
            if args.task_name == 'SNLI-VE':
                self.v_mlp = MLPLayer(args, config)
                self.v_classifier = nn.Linear(config.hidden_size, 3)
            if args.task_name == 'GTE':
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
        """if args.pair:
            self.classifier = nn.Linear(config.hidden_size, 3)
        else:
            self.classifier = nn.Linear(config.hidden_size * 4, 3)"""
    
    def forward(self, inputs, inference=False):
        if self.args.task_name == 'GTE':
            if self.args.get_confactual_emb:
            
                labels = inputs['label']
                image_embedding = inputs['image']
                hypothesis_embedding = self.bert.embeddings(inputs['text']['hypothesis']['input_ids'].squeeze(1))
                con_embedding = self.bert(inputs['text']['con_text']['input_ids'].squeeze(1), inputs['text']['con_text']['attention_mask'].squeeze(1))
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
                #H_output = self.bert(inputs['text']['hypothesis']['input_ids'].squeeze(1), inputs['text']['hypothesis']['attention_mask'])
    
                if self.args.pool_type == 'cls':
                    VH_emb = VH_output[:, 0, :]
                    PH_emb = PH_output[:, 0, :]
                    #H_emb = H_output[:, 0, :]
                return VH_emb, PH_emb, labels
                
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
                #elif self.args.pool_type == 'mean':
                    #emb = (output * combine_attention_mask.squeeze(1).unsqueeze(-1)).sum(1) / combine_attention_mask.squeeze(1).sum(-1).unsqueeze(-1)
                #emb = torch.concat([image_embedding, text_embedding, torch.abs(image_embedding-text_embedding), image_embedding*text_embedding], dim=1)
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
                VH_logits = F.softmax(PH_logits, dim=-1)
                H_logits = self.H_classifier(H_emb)
                H_logits = F.softmax(H_logits, dim=-1)
                logits = self.main_classifier(fusion_emb)
                logits = F.softmax(logits, dim=-1)
                main_loss = self.calc_loss(logits, labels)
                VH_loss = self.calc_loss(VH_logits, labels)
                PH_loss = self.calc_loss(PH_logits, labels)
                H_loss = self.calc_loss(H_logits, labels)
                loss = main_loss + 0.5 * VH_loss + 0.5 * PH_loss
                
                if inference:
                    """
                    con_image_embedding = self.confactual_image.repeat(cls_embedding.shape[0], 1, 1).cuda()
                    con_text = self.confactual_text.repeat(cls_embedding.shape[0], 1).cuda()
                    con_hypothesis_embedding = self.bert.embeddings(con_text)
                    con_com_embedding = self.bert(con_text, inputs['text']['con_text']['attention_mask'].squeeze(1))
                    con_image_embedding = self.bert.embeddings(inputs_embeds=con_image_embedding)
                
                    con_cls_embedding = con_hypothesis_embedding[:, 0:1, :]
                    con_image_embedding = con_image_embedding[:, 1:, :]
                    con_hypothesis_embedding = con_hypothesis_embedding[:, 1:, :]
            
                    con_VH_embedding = torch.cat([con_cls_embedding, con_image_embedding, con_hypothesis_embedding], dim=1)
                    con_VH_output = self.bert.encoder(con_VH_embedding, attention_mask=attention_mask)["last_hidden_state"]
                    con_PH_output = con_com_embedding["last_hidden_state"]
                    """
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
                    if self.args.dynamic_debias:
                        return logits, con_logits, H_emb, labels
                        
                    '''con_PH_emb = self.PH_mlp(con_PH_emb)
                    con_PH_logits = self.PH_classifier(con_PH_emb)
                    con_PH_logits = F.softmax(con_PH_logits, dim=-1)
                    con_VH_emb = self.VH_mlp(con_VH_emb)
                    con_VH_logits = self.VH_classifier(con_VH_emb)
                    con_VH_logits = F.softmax(con_VH_logits, dim=-1) 
                    logits = torch.sigmoid(VH_logits + PH_logits + H_logits)
                    con_logits = torch.sigmoid(con_VH_logits + con_PH_logits + H_logits)'''
                    return logits, con_logits, torch.argmax(logits, dim=1), labels
                    
                if self.args.use_CL:
                    #method1
                    premise_emb = self.bert(inputs['text']['premise']['input_ids'].squeeze(1), inputs['text']['premise']['attention_mask'].squeeze(1))
                    premise_cls = premise_emb["last_hidden_state"][:, 0, :]
                    image_emb = self.bert(inputs_embeds=inputs['image'], attention_mask=torch.ones(cls_embedding.shape[0], 50).cuda())
                    image_cls = image_emb["last_hidden_state"][:, 0, :]
                    CL_loss = self.calc_itc_loss(premise_cls, image_cls)
                    #method2
                    """
                    image_emb = torch.mean(VH_output[:, 1:51, :], dim=1)
                    premise_mask = inputs['text']['premise']['attention_mask'].squeeze(1)[:, 1:]
                    premise_emb = (PH_output[:, 1:, :] * premise_mask.squeeze(1).unsqueeze(-1)).sum(1) / premise_mask.squeeze(1).sum(-1).unsqueeze(-1)
                    CL_loss = self.calc_itc_loss(premise_emb, image_emb)
                    """
                    return loss + CL_loss
                else:
                    return loss
                
            if self.args.train_mode == 'TE':
                labels = inputs['label']
                con_embedding = self.bert(inputs['text']['con_text']['input_ids'].squeeze(1), inputs['text']['con_text']['attention_mask'])
                PH_output = con_embedding["last_hidden_state"]
                if self.args.pool_type == 'cls':
                    emb = PH_output[:, 0, :]
                    
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
                #elif self.args.pool_type == 'mean':
                    #emb = (output * combine_attention_mask.squeeze(1).unsqueeze(-1)).sum(1) / combine_attention_mask.squeeze(1).sum(-1).unsqueeze(-1)
                #emb = torch.concat([image_embedding, text_embedding, torch.abs(image_embedding-text_embedding), image_embedding*text_embedding], dim=1)
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
                #method1
                """
                premise_emb = self.bert(inputs['text']['premise']['input_ids'].squeeze(1), inputs['text']['premise']['attention_mask'])
                premise_cls = premise_emb["last_hidden_state"][:, 0, :]
                image_emb = self.bert(inputs_embeds=inputs['image'], attention_mask=torch.ones(cls_embedding.shape[0], 50).cuda())
                image_cls = image_emb["last_hidden_state"][:, 0, :]
                CL_loss = self.calc_itc_loss(premise_cls, image_cls)
                """
                #method2
                image_emb = torch.mean(VH_output[:, 1:51, :], dim=1)
                premise_mask = inputs['text']['premise']['attention_mask'].squeeze(1)[:, 1:]
                premise_emb = (PH_output[:, 1:, :] * premise_mask.squeeze(1).unsqueeze(-1)).sum(1) / premise_mask.squeeze(1).sum(-1).unsqueeze(-1)
                CL_loss = self.calc_itc_loss(premise_emb, image_emb)
                return self.calc_loss(logits, labels) + CL_loss
            else:
                return self.calc_loss(logits, labels)
                
        if not inference and (self.args.only_pre or self.args.only_hy):
                labels = inputs['label']
                if self.args.task_name == 'SNLI-VE' and self.args.only_pre:
                    image_embedding = inputs['image']
                    image_embedding = self.dense(image_embedding).reshape(image_embedding.shape[0], 1, -1)
                    output = self.bert(inputs_embeds=image_embedding)["last_hidden_state"][:, 0, :]
                    return output, labels
                elif self.args.task_name == 'SNLI-VE' and self.args.only_hy:
                    output = self.bert(inputs['text']['input_ids'].squeeze(1), inputs['text']['attention_mask'].squeeze(1))["last_hidden_state"][:, 0, :]
                    return output, labels
                else:
                    output = self.bert(inputs['input_ids'].reshape(-1, inputs['input_ids'].shape[-1]), inputs['attention_mask'].reshape(-1, inputs['attention_mask'].shape[-1]))
                    return output.last_hidden_state[:, 0, :], labels
                    

        else:
            labels = inputs['label']
            if self.args.only_img:
                image_embedding = inputs['image']
                image_embedding = self.dense(image_embedding).reshape(image_embedding.shape[0], 1, -1)
                image_embedding = self.bert.embeddings(inputs_embeds=image_embedding)
                #output = self.bert(inputs_embeds=image_embedding).last_hidden_state
                attention_mask = torch.ones(image_embedding.shape[0], 1).cuda()
                attention_mask = attention_mask[:, None, None, :]
                attention_mask = (1.0 - attention_mask) * -10000.0
        
                output = self.bert.encoder(image_embedding, attention_mask=attention_mask)["last_hidden_state"]
    
                if self.args.pool_type == 'cls':
                    emb = output[:, 0, :]
                elif self.args.pool_type == 'mean':
                    emb = (output * combine_attention_mask.squeeze(1).unsqueeze(-1)).sum(1) / combine_attention_mask.squeeze(1).sum(-1).unsqueeze(-1)
                #emb = torch.concat([image_embedding, text_embedding, torch.abs(image_embedding-text_embedding), image_embedding*text_embedding], dim=1)
                emb = self.mlp(emb)
                logits = self.classifier(emb)
                logits = F.softmax(logits, dim=-1)
                if inference:
                    return logits, torch.argmax(logits, dim=1), labels
                return self.calc_loss(logits, labels)
            if self.args.snli_ve_only_hy:
                emb = self.bert(inputs['text']['input_ids'].squeeze(1), inputs['text']['attention_mask'].squeeze(1))["last_hidden_state"][:, 0, :]
                emb = self.mlp(emb)
                logits = self.classifier(emb)
                logits = F.softmax(logits, dim=-1)
                if inference:
                    return logits, torch.argmax(logits, dim=1), labels
                return self.calc_loss(logits, labels)
                
            if self.args.confactual_train:
                mean_pre = torch.Tensor(np.load('data/snli-ve_all_mean.npy', allow_pickle=True).reshape(1, 1, -1)).cuda()
                #mean_en = torch.Tensor(np.load('data/bert_SNLI-VE_entailment.npy', allow_pickle=True).reshape(1, 1, -1)).cuda()
                #mean_ne = torch.Tensor(np.load('data/bert_SNLI-VE_neutral.npy', allow_pickle=True).reshape(1, 1, -1)).cuda()
                #mean_con = torch.Tensor(np.load('data/bert_SNLI-VE_contradiction.npy', allow_pickle=True).reshape(1, 1, -1)).cuda()
                #mean_label = (mean_en + mean_ne + mean_con) / 3
                
                image_embedding = inputs['image']
                image_embedding = self.dense(image_embedding).reshape(image_embedding.shape[0], 1, -1)
                text_embedding = self.bert.embeddings(inputs['text']['input_ids'].squeeze(1))
                image_embedding = self.bert.embeddings(inputs_embeds=image_embedding)
                
                cls_embedding = text_embedding[:, 0:1, :]
                text_embedding = text_embedding[:, 1:, :]
                
                VT_embedding = torch.cat([cls_embedding, image_embedding, text_embedding], dim=1)
                combine_attention_mask = torch.cat([torch.ones(cls_embedding.shape[0], 1).cuda(), inputs['text']['attention_mask'].squeeze(1)], dim=1)
                attention_mask = combine_attention_mask[:, None, None, :]
                attention_mask = (1.0 - attention_mask) * -10000.0
                
                VT_output = self.bert.encoder(VT_embedding, attention_mask=attention_mask)
                T_output = self.bert(inputs['text']['input_ids'].squeeze(1), inputs['text']['attention_mask'].squeeze(1))
                Y_vhm = VT_output.last_hidden_state[:, 0, :]
                Y_v = VT_output.last_hidden_state[:, 1, :]
                Y_b = T_output.last_hidden_state[:, 0, :]
        
                vhm_emb = self.ph_mlp(Y_vhm)
                b_emb = self.h_mlp(Y_b)
                v_emb = self.v_mlp(Y_v)
                vhm_logits = self.ph_classifier(vhm_emb)
                b_logits = self.h_classifier(b_emb)
                v_logits = self.v_classifier(v_emb)
                
                if self.args.fusion == 'CON':
                    Y_vhmb = torch.cat([Y_vhm, Y_v, Y_b], dim=1)
                else:
                    Y_vhmb = Y_vhm + Y_v + Y_b
                    Y_vhmb = torch.sigmoid(Y_vhmb)
                    
                fusion_emb = self.main_mlp(Y_vhmb)
                logits = self.main_classifier(fusion_emb)
                if inference:
                    pre_emb = mean_pre.repeat(cls_embedding.shape[0], 1, 1)
                    con_emb = torch.cat([cls_embedding, pre_emb, text_embedding], dim=1)
                    con_output = self.bert.encoder(con_emb, attention_mask)
                    con_Y_vhm = con_output.last_hidden_state[:, 0, :]
                    if self.args.fusion == 'CON':
                        Y_vhmb = torch.cat([Y_vhm, Y_v, Y_b], dim=1)
                        con_Y_vhmb = torch.cat([con_Y_vhm, Y_v, Y_b], dim=1)
                    else:
                        Y_vhmb = Y_vhm + Y_v + Y_b
                        Y_vhmb = torch.sigmoid(Y_vhmb)
                        con_Y_vhmb = con_Y_vhm + Y_v + Y_b
                        con_Y_vhmb = torch.sigmoid(con_Y_vhmb)
                    logits = self.main_classifier(self.main_mlp(Y_vhmb))
                    con_logits = self.main_classifier(self.main_mlp(con_Y_vhmb))
                    return logits, con_logits, torch.argmax(logits, dim=1), labels
                main_loss = self.calc_loss(logits, labels)
                loss1 = self.calc_loss(vhm_logits, labels)
                loss2 = self.calc_loss(b_logits, labels)
                loss3 = self.calc_loss(v_logits, labels)
                a1 = 0.3
                a2 = 0.3
                a3 = 0.3
                loss = main_loss + a1 * loss1 + a2 * loss2 + a3 * loss3
                return loss
                
            else:
                image_embedding = inputs['image']
                image_embedding = self.dense(image_embedding).reshape(image_embedding.shape[0], 1, -1)
                text_embedding = self.bert.embeddings(inputs['text']['input_ids'].squeeze(1))
                image_embedding = self.bert.embeddings(inputs_embeds=image_embedding)
            
                cls_embedding = text_embedding[:, 0:1, :]
                text_embedding = text_embedding[:, 1:, :]
        
                combine_embedding = torch.cat([cls_embedding, image_embedding, text_embedding], dim=1)
                combine_attention_mask = torch.cat([torch.ones(cls_embedding.shape[0], 1).cuda(), inputs['text']['attention_mask'].squeeze(1)], dim=1)
                attention_mask = combine_attention_mask[:, None, None, :]
                attention_mask = (1.0 - attention_mask) * -10000.0
        
                output = self.bert.encoder(combine_embedding, attention_mask=attention_mask)["last_hidden_state"]
    
                if self.args.pool_type == 'cls':
                    #emb = output[:, 0, :]
                    emb = output[:, 0, :]
                elif self.args.pool_type == 'mean':
                    emb = (output * combine_attention_mask.squeeze(1).unsqueeze(-1)).sum(1) / combine_attention_mask.squeeze(1).sum(-1).unsqueeze(-1)
                #emb = torch.concat([image_embedding, text_embedding, torch.abs(image_embedding-text_embedding), image_embedding*text_embedding], dim=1)
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
    
    def calc_itc_loss(self, premise_cls, image_cls, lamda=0.05):
        similarities = F.cosine_similarity(premise_cls.unsqueeze(1), image_cls.unsqueeze(0), dim=-1)
        similarities /= lamda
        labels = torch.arange(similarities.shape[0]).cuda()
        loss = F.cross_entropy(similarities, labels)
        return torch.mean(loss)
     
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
        
        




