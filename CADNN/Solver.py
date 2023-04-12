import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
import torch.nn.functional as F

import os
import pickle as pkl

from transformers import AdamW
import time, datetime
from tqdm import tqdm
import gc


from data.dataset import MINDdata
from NRMS_model import NRMS, NRMS_CI
from baselines import PDA, TIDE, MACR, DICE
from class_utils import write_file, cal_metric, write_zip_file, BPRLoss, DcorLoss


class CISolver:
    def __init__(self, args):
        self._args = args
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._num_workers = max([4*torch.cuda.device_count(), 4])
        torch.manual_seed(48) 

        dataset = MINDdata(args=self._args)

        self.train_loader = dataset.get_loader(
            batch_size=self._args.batch_size,
            type='train',
            shuffle=True,
            num_worker=self._num_workers,
            pin_memory=True,
            drop_last=True,
        )

        self.dev_loader = dataset.get_loader(
            batch_size=1,
            type='dev',
            shuffle=False,
            num_worker=self._num_workers,
            pin_memory=True,
            drop_last=False
        )

        self.test_loader = dataset.get_loader(
            batch_size=1,
            type='test',
            shuffle=False,
            num_worker=self._num_workers,
            pin_memory=True,
            drop_last=False
        )

        print(f'#examples:'
              f'\n#train {len(self.train_loader.dataset)}'
              f'\n#dev {len(self.dev_loader.dataset)}'
              f'\n#test {len(self.test_loader.dataset)}'
              )

        model_set = {
            'nrms': NRMS,
            'nrms_ci': NRMS_CI,
        }

        try:
            model_select = self._args.net.lower()
            self.model = model_set[model_select](self._args)
        except:
            raise ValueError(f'[Error] Unconstructed models {self._args.net.lower()}, please try again!')

        device_count = 0
        if self._device == 'cuda':
            device_count = torch.cuda.device_count()
            if device_count > 1:
                self.model = nn.DataParallel(self.model)
            torch.backends.cudnn.benchmark = True
            print(f'> {device_count} GPUs have been used')

        self.model.to(self._device)

        params = self.model.module.req_grad_params if device_count > 1 else self.model.req_grad_params
        optimizer = optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))

        lr_decay = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, 
            mode='max',
            factor=0.5,
            patience=3,
            eps=10-7
            )

        if self._args.use_bert and self._args.train_bert:
            params_bert = list(self.model.module.bert.named_parameters() if device_count > 1 else self.model.bert.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_group_params = [
                {
                    'params': [p for n, p in params_bert if not any(nd in n for nd in no_decay)],
                    'weight_decay_rate': 0.01
                },
                {
                    'params': [p for n, p in params_bert if any(nd in n for nd in no_decay)],
                    'weight_decay_rate': 0.0
                }
            ]
            optimizer_bert = AdamW(optimizer_group_params, lr=3e-5)
            self._optimizer_bert = optimizer_bert

        if self._args.npratio == 0:
            loss = nn.BCELoss()
        else:
            if self._args.use_bpr:
                loss = BPRLoss()
            else:
                loss = nn.CrossEntropyLoss()

        self._dev_loss = nn.BCELoss()

        ckpt_path = os.path.join('checkpoint', '{}'.format(args.name))
        if not os.path.exists(ckpt_path+'.pth'):
            print(f'>> Checkpoint Path has not been found {ckpt_path} !')

        self._optimizer = optimizer
        self._loss = loss
        self._ckpt_path = ckpt_path
        self._lr_decay = lr_decay

        self.batch_idx = []
        self.train_logs = []
        self.test_log = []

    def train(self):
        print(f'> start training model {self._args.name}')

        best_auc = 0.
        self.best_epoch = 0

        for epoch in range(1, self._args.epochs + 1):
            print('=='*20 + f'Epoch: {epoch}' + '=='*20)

            if epoch != 1:
                write_file(self._args.log_path, f'test_log_{self._args.name}.log', self.test_log)
                self.test_log.clear()

            train_loss = self.train_epoch(epoch)

            evaluation_metrics, predict_rank = self.evaluation_epoch('dev')

            if evaluation_metrics['auc'] > best_auc:
                best_auc = evaluation_metrics['auc']
                self.save_model('dev')
                self.best_epoch = epoch
            
            self._lr_decay.step(evaluation_metrics['auc'])

            single_log = f'='*30 + f'{datetime.datetime.now()}' + f'='*30 + '\n'
            single_log += f'{self._args.name}\ttrain_loss:{train_loss}\t'
            for key, value in evaluation_metrics.items():
                single_log += f'{key}:{value:.4f}\t'

            self.test_log.append(single_log)

            print(single_log.replace('\t', '\n'))

            write_file(self._args.log_path, f'train_log_{self._args.name}.log', self.train_logs)
            self.train_logs.clear()

        print('> modeling training finished!')
        self.save_model('last')
        self.test()

    def train_epoch(self, epoch_idx):
        self.model.train()
        train_loss = 0.
        correct = 0
        train_example_count = 0

        final_score = []
        log_score = []
        original_score = []
        conformity_score = []

        total_iteration = int(len(self.train_loader.dataset)//self._args.batch_size)

        with autograd.detect_anomaly():
            for batch_idx, contents in enumerate(self.train_loader):
                labels = contents[0].to(self._device)
                user_idx = contents[1].to(self._device)
                candidate_ctr = contents[3].to(self._device)
                history_ctr = contents[5].to(self._device)
                user_category_kl = contents[6].to(self._device)
                user_subcategory_kl = contents[7].to(self._device)
                click_title_prob = contents[-5].to(self._device)
                candidate_title_prob = contents[-4].to(self._device)
                all_candidate_ctr = contents[-2].to(self._device)

                if self._args.use_bert:
                    candidate_titles = {key: value.to(self._device) for key, value in contents[2].items()}
                    history_titles = {key: value.to(self._device) for key, value in contents[4].items()}
                else:
                    candidate_titles = contents[2].to(self._device)
                    history_titles = contents[4].to(self._device)

                input_line = {
                    'user_idx': user_idx,
                    'candidate_titles': candidate_titles,
                    'history_titles': history_titles,
                    'candidate_ctr': candidate_ctr,
                    'history_ctr': history_ctr,
                    'user_category_kl': user_category_kl,
                    'user_subcategory_kl': user_subcategory_kl,
                    'click_title_prob': click_title_prob,
                    'candidate_title_prob': candidate_title_prob,
                    'all_candidate_ctr': all_candidate_ctr
                }

                predict_info = self.model(input_line)

                if isinstance(predict_info, list):
                    if len(predict_info) == 1:
                        predictions = predict_info[0]
                    elif len(predict_info) == 3:
                        predictions = predict_info[0]
                        final_score.append(predict_info[0].cpu().detach().numpy())
                        original_score.append(predict_info[1].cpu().detach().numpy())
                        conformity_score.append(predict_info[2].cpu().detach().numpy())
                    else:
                        raise ValueError('wrong return value number, please try again!')
                else:
                    raise TypeError('wrong return value type, please try again!')

                torch.autograd.set_detect_anomaly(True)
                loss = self._loss(predictions, labels)

                if math.isnan(loss):
                    print(f'[Error!] loss is Nan-> {loss}')
                    exit(-1)

                if loss < 0:
                    print(f'[Error!] loss is 0-> {loss}')
                    exit(-1)

                self._optimizer.zero_grad()
                if self._args.use_bert and self._args.train_bert:
                    self._optimizer_bert.zero_grad()

                loss.backward()

                if self._args.grad_max_norm > 0:
                    torch.nn.utils.clip_grad_norm(self.model.req_grad_params, self._args.grad_max_norm)

                self._optimizer.step()
                if self._args.use_bert and self._args.train_bert:
                    self._optimizer_bert.step()

                current_time = str(datetime.datetime.now()).split('.')[0]
                log_screen = f'{current_time}\tEpoch:{epoch_idx}-{batch_idx+1}\t{self._args.name}\tloss: {loss:.4f}\t'

                self.train_logs.append(log_screen)
                if batch_idx == 0 or (batch_idx + 1) % self._args.display_step == 0 or batch_idx == total_iteration:
                    print(log_screen.replace('\t', ','))

                if (batch_idx + 1) % (10 * self._args.display_step) == 0 and len(final_score) != 0:
                    write_file('logs', 'final_score.txt', final_score)
                    write_file('logs', 'original_score.txt', original_score)
                    write_file('logs', 'conformity_score.txt', conformity_score)
                    final_score.clear()
                    original_score.clear()
                    conformity_score.clear()

                train_loss += loss
                train_example_count += self._args.batch_size

        return train_loss / (train_example_count * 1.0)

    def test(self):
        test_logs = []

        if self._args.test:
            eval_sets = ['test']
            self.best_epoch = -1
        else:
            eval_sets = ['valid']

        for name in ['dev', 'test', 'last']:
            try:
                self.load_model(name)
            except FileNotFoundError:
                print(f'> checkpoint is not found: {name}')
                continue
            for eval_set in eval_sets:
                print('=='*20 + f'{eval_set} results at {name}' + '=='*20)
                evaluation_metrics, predict_rank = self.evaluation_epoch(eval_set)
                log = f'{eval_set}: best epoch:{self.best_epoch}\t'
                log_value = f'all results: \t'
                for key, value in evaluation_metrics.items():
                    log += f'{key}:{value:.4f}\t'
                    log_value += f'{value:.4f}\t'
                print(log.replace('\t', '\n'))
                print(log_value)
                test_logs.append(log)

                if self._args.test:
                    writed_line = []
                    for idx, values in tqdm(predict_rank.items()):
                        writed_line.append(str(idx) + ' ' + values)
                    write_zip_file(self._args.log_path, f'predict_rank_{self._args.name}_{name}.zip', writed_line)

        write_file(self._args.log_path, f'test_log_{self._args.name}.log', test_logs)

    def evaluation_epoch(self, eval_set):
        self.model.eval()
        if eval_set.lower() == 'dev':
            loader = self.dev_loader
        else:
            loader = self.test_loader

        eval_loss = 0
        eval_count = 0
        ground_targets = []
        preds = []
        predict_rank = {}
        with torch.no_grad():
            for batch_idx, contents in tqdm(enumerate(loader)):
                labels = contents[0].float().to(self._device)
                user_idx = contents[1].to(self._device)
                candidate_ctr = contents[3].to(self._device)
                history_ctr = contents[5].to(self._device)
                user_category_kl = contents[6].to(self._device)
                user_subcategory_kl = contents[7].to(self._device)
                impressId = contents[-1].to(self._device)

                click_title_prob = contents[-5].to(self._device)
                candidate_title_prob = contents[-4].to(self._device)
                all_candidate_ctr = contents[-2].to(self._device)

                if self._args.use_bert:
                    candidate_titles = {key: value.to(self._device) for key, value in contents[2].items()}
                    history_titles = {key: value.to(self._device) for key, value in contents[4].items()}
                else:
                    candidate_titles = contents[2].to(self._device)
                    history_titles = contents[4].to(self._device)

                input_line = {
                    'user_idx': user_idx,
                    'candidate_titles': candidate_titles,
                    'history_titles': history_titles,
                    'candidate_ctr': candidate_ctr,
                    'history_ctr': history_ctr,
                    'user_category_kl': user_category_kl,
                    'user_subcategory_kl': user_subcategory_kl,
                    'click_title_prob': click_title_prob,
                    'candidate_title_prob': candidate_title_prob,
                    'all_candidate_ctr': all_candidate_ctr
                 
                }

                predict_info = self.model(input_line)

                if isinstance(predict_info, list):
                    if len(predict_info) == 1:
                        predictions = predict_info[0]
                    elif len(predict_info) == 3:
                        predictions = predict_info[1]
                    else:
                        raise ValueError('wrong return value number, please try again!')
                else:
                    raise TypeError('wrong return value type, please try again!')

                current_labels = labels.squeeze().cpu().numpy()
                current_predicts = predictions.squeeze().cpu().numpy()
                ground_targets.append(current_labels)
                preds.append(current_predicts)
                current_rank = np.argsort(current_predicts)
                current_rank = current_rank[::-1] + 1
                recoreded_rank = '[' + ','.join([str(i) for i in current_rank.tolist()]) + ']'
                predict_rank[impressId.squeeze().cpu().item()] = recoreded_rank

        
            evaluations = []
            for y_true, y_pred in tqdm(zip(ground_targets, preds)):
                try:
                    evals = cal_metric(
                        labels=y_true,
                        preds=y_pred,
                        metrics=self._args.metrics
                    )
                    evaluations.append(evals)
                except:
                    print(y_true)
                    print(y_pred)
                    exit(-1)

            dict_keys = evaluations[0].keys()
            final_evaluation = {}
            for key in dict_keys:
                final_evaluation[key] = np.mean([item[key] for item in evaluations])
        
        return final_evaluation, predict_rank

    def save_model(self, name='dev'):
        model_dict = {}
        model_dict['state_dict'] = self.model.state_dict()
        model_dict['m_config'] = self._args
        model_dict['optimizer'] = self._optimizer.state_dict()
        if name is None:
            ckpt_path = self._ckpt_path + '.pth'
        else:
            ckpt_path = self._ckpt_path + '_' + name + '.pth'

        if not os.path.exists(os.path.dirname(ckpt_path)):
            os.makedirs(os.path.dirname(ckpt_path))
        torch.save(model_dict, ckpt_path)
        print(f'> model have been saved at {ckpt_path}')

    def load_model(self, name='dev'):
        ckpt_path = self._ckpt_path + '_' + name + '.pth'
        print(f'load checkpoint {ckpt_path}')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self._device)
        else:
            raise FileNotFoundError(f'checkpoint has not been found in {ckpt_path}')
        try:
            self.model.load_state_dict(checkpoint['state_dict'],False)
        except:
            self.model = nn.DataParallel(self.model)
            self.model.load_state_dict(checkpoint['state_dict'],False)

        print(f'> model have been load at {ckpt_path}')


class BaselineSolver:
    def __init__(self, args):
        self._args = args
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._num_workers = 0
        torch.manual_seed(48)  

        dataset = MINDdata(args=self._args)

        self.train_loader = dataset.get_loader(
            batch_size=self._args.batch_size,
            type='train',
            shuffle=True,
            num_worker=self._num_workers,
            pin_memory=True,
            drop_last=True,
        )

        if self._args.net.lower() == 'dice':
            self.negative_train_loader = dataset.get_loader(
                batch_size=self._args.batch_size,
                type='positive_train',
                shuffle=True,
                num_worker=self._num_workers,
                pin_memory=True,
                drop_last=True,

            )

        self.dev_loader = dataset.get_loader(
            batch_size=1,
            type='dev',
            shuffle=False,
            num_worker=self._num_workers,
            pin_memory=True,
            drop_last=False
        )

        self.test_loader = dataset.get_loader(
            batch_size=1,
            type='test',
            shuffle=False,
            num_worker=self._num_workers,
            pin_memory=True,
            drop_last=False
        )

        print(f'#examples:'
              f'\n#train {len(self.train_loader.dataset)}'
              f'\n#dev {len(self.dev_loader.dataset)}'
              f'\n#test {len(self.test_loader.dataset)}'
              )

        model_set = {
            'dice': DICE,
            'pda': PDA,
            'macr': MACR,
            'tide': TIDE,
        }

        try:
            model_select = self._args.net.lower()
            if self._args.net.lower() == 'tide':
                self.model = model_set[model_select](self._args, item_size=self.train_loader.dataset._get_news_count())
            else:
                self.model = model_set[model_select](self._args)
        except:
            raise ValueError(f'[Error] Unconstructed models {self._args.net.lower()}, please try again!')

        checkpoint_path = 'checkpoint/naml_ci_mind_dev.pth'
        checkpoint = torch.load(checkpoint_path)
        self.load_model()
        device_count = 0
        if self._device == 'cuda':
            device_count = torch.cuda.device_count()
            if device_count > 1:
                self.model = nn.DataParallel(self.model)
            torch.backends.cudnn.benchmark = True
            print(f'> {device_count} GPUs have been used')

        self.model.to(self._device)

        params = self.model.module.req_grad_params if device_count > 1 else self.model.req_grad_params
        optimizer = optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))
        optimizer.load_state_dict(checkpoint['optimizer'])

        lr_decay = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            eps=10 - 7
        )

        if self._args.use_bert and self._args.train_bert:
            params_bert = list(
                self.model.module.bert.named_parameters() if device_count > 1 else self.model.bert.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_group_params = [
                {
                    'params': [p for n, p in params_bert if not any(nd in n for nd in no_decay)],
                    'weight_decay_rate': 0.01
                },
                {
                    'params': [p for n, p in params_bert if any(nd in n for nd in no_decay)],
                    'weight_decay_rate': 0.0
                }
            ]
            optimizer_bert = AdamW(optimizer_group_params, lr=3e-5)
            self._optimizer_bert = optimizer_bert

        if self._args.net.lower() == 'macr':
            loss = nn.CrossEntropyLoss()
        elif self._args.net.lower() in ['pda', 'tide', 'dice']:
            loss = BPRLoss()
        else:
            raise ValueError(f'[Error!] wrong net selection, please try again')

        self._disc_loss = DcorLoss()

        self._dev_loss = nn.BCELoss()

        ckpt_path = os.path.join('checkpoint', '{}'.format(args.name))
        if not os.path.exists(ckpt_path + '.pth'):
            print(f'>> Checkpoint Path has not been found {ckpt_path} !')

        self._optimizer = optimizer
        self._loss = loss
        self._ckpt_path = ckpt_path
        self._lr_decay = lr_decay

        self.batch_idx = []
        self.train_logs = []
        self.test_log = []

    def train(self):
        print(f'> start training model {self._args.name}')

        best_auc = 0.
        self.best_epoch = 0

        for epoch in range(1, self._args.epochs + 1):
            print('==' * 20 + f'Epoch: {epoch}' + '==' * 20)

            if epoch != 1:
                write_file(self._args.log_path, f'test_log_{self._args.name}.log', self.test_log)
                self.test_log.clear()

            if self._args.net.lower() == 'dice':
                train_loss = 0
                for idx, current_loader in enumerate([self.train_loader, self.negative_train_loader]):
                    train_loss += self.train_epoch(epoch, current_loader, idx)
                train_loss = train_loss / 2
            else:
                train_loss = self.train_epoch(epoch, self.train_loader)
                
            evaluation_metrics, predict_rank = self.evaluation_epoch('dev')

            if evaluation_metrics['auc'] > best_auc:
                best_auc = evaluation_metrics['auc']
                self.save_model('dev')
                self.best_epoch = epoch

            self._lr_decay.step(evaluation_metrics['auc'])

            single_log = f'=' * 30 + f'{datetime.datetime.now()}' + f'=' * 30 + '\n'
            single_log += f'{self._args.name}\ttrain_loss:{train_loss}\t'
            for key, value in evaluation_metrics.items():
                single_log += f'{key}:{value:.4f}\t'

            self.test_log.append(single_log)

            print(single_log.replace('\t', '\n'))

            write_file(self._args.log_path, f'train_log_{self._args.name}.log', self.train_logs)
            self.train_logs.clear()

        print('> modeling training finished!')
        self.save_model('last')
        self.test()

    def train_epoch(self, epoch_idx, current_loader, loader_sign=None):
        self.model.train()
        train_loss = 0.
        correct = 0
        train_example_count = 0

        final_score = []
        log_score = []
        original_score = []
        conformity_score = []

        total_iteration = int(len(current_loader.dataset) // self._args.batch_size)

        with autograd.detect_anomaly():
            for batch_idx, contents in enumerate(current_loader):
                labels = contents[0].to(self._device)
                user_idx = contents[1].to(self._device)
                candidate_ctr = contents[3].to(self._device)
                history_ctr = contents[5].to(self._device)
                user_category_kl = contents[6].to(self._device)
                user_subcategory_kl = contents[7].to(self._device)

                candidate_index = contents[-3].to(self._device)

                if self._args.use_bert:
                    candidate_titles = {key: value.to(self._device) for key, value in contents[2].items()}
                    history_titles = {key: value.to(self._device) for key, value in contents[4].items()}
                else:
                    candidate_titles = contents[2].to(self._device)
                    history_titles = contents[4].to(self._device)

                input_line = {
                    'user_idx': user_idx,
                    'candidate_index': candidate_index,
                    'candidate_titles': candidate_titles,
                    'history_titles': history_titles,
                    'candidate_ctr': candidate_ctr,
                    'history_ctr': history_ctr,
                    'user_category_kl': user_category_kl,
                    'user_subcategory_kl': user_subcategory_kl
                }
            
                predict_info = self.model(input_line, True)

                if self._args.net.lower() == 'macr':
                    final_score, matching_score, user_score, item_score = predict_info
                    l1 = self._loss(final_score, labels)
                    l2 = self._loss(item_score, labels)
                    l3 = self._loss(user_score, labels)
                    loss = l1 + self._args.macr_alpha * l2 + self._args.macr_beta * l3
                elif self._args.net.lower() == 'pda':
                    final_score, acticated_matching, matching_score = predict_info
                    loss = self._loss(final_score)
                elif self._args.net.lower() == 'tide':
                    final_score, matching_score, weight = predict_info
                    loss = self._loss(final_score)
                elif self._args.net.lower() == 'dice':
                    final_score, matching_interest, matching_conformity, representations = predict_info
                    if loader_sign == 0:   
                        l_conf = self._loss(matching_conformity)
                        l_click = self._loss(final_score)
                        l_intere = 0
                    else:               
                        l_conf = -1 * self._loss(matching_conformity)
                        l_click = self._loss(final_score)
                        l_intere = self._loss(matching_interest)
                    loss = l_click +  0.25*l_intere + self._args.dice_alpha*(l_conf)
              
                else:
                    raise ValueError('[Error!]wrong net selection, please try again!')

                self._optimizer.zero_grad()
                if self._args.use_bert and self._args.train_bert:
                    self._optimizer_bert.zero_grad()

                try:
                    loss.backward()
                except RuntimeError:
                    print(final_score.shape)

                if self._args.grad_max_norm > 0:
                    torch.nn.utils.clip_grad_norm(self.model.req_grad_params, self._args.grad_max_norm)

                self._optimizer.step()
                if self._args.use_bert and self._args.train_bert:
                    self._optimizer_bert.step()

                current_time = str(datetime.datetime.now()).split('.')[0]
                log_screen = f'{current_time}\tEpoch:{epoch_idx}-{batch_idx + 1}\t{self._args.name}\tloss: {loss:.4f}\t'

                self.train_logs.append(log_screen)
                if batch_idx == 0 or (batch_idx + 1) % self._args.display_step == 0 or batch_idx == total_iteration:
                    print(log_screen.replace('\t', ','))

                train_loss += loss
                train_example_count += self._args.batch_size

        return train_loss / (train_example_count * 1.0)

    def test(self):
        test_logs = []

        if self._args.test:
            eval_sets = ['test']
            self.best_epoch = -1
        else:
            eval_sets = ['valid']

        for name in ['dev', 'test', 'last']:
            try:
                self.load_model(name)
            except FileNotFoundError:
                print(f'> checkpoint is not found: {name}')
                continue
            for eval_set in eval_sets:
                print('==' * 20 + f'{eval_set} results at {name}' + '==' * 20)
                evaluation_metrics, predict_rank = self.evaluation_epoch(eval_set)
                log = f'{eval_set}: best epoch:{self.best_epoch}\t'
                for key, value in evaluation_metrics.items():
                    log += f'{key}:{value:.4f}\t'
                print(log.replace('\t', '\n'))
                test_logs.append(log)

                if eval_set == 'test':
                    writed_line = []
                    for idx, values in tqdm(predict_rank.items()):
                        writed_line.append(str(idx) + ' ' + values)
                    write_zip_file(self._args.log_path, f'predict_rank_{self._args.name}_{name}.zip', writed_line)

        write_file(self._args.log_path, f'test_log_{self._args.name}.log', test_logs)

    def evaluation_epoch(self, eval_set):
        self.model.eval()
        if eval_set.lower() == 'dev':
            loader = self.dev_loader
        else:
            loader = self.test_loader

        eval_loss = 0
        eval_count = 0
        ground_targets = []
        preds = []
        predict_rank = {}
        with torch.no_grad():
            for batch_idx, contents in tqdm(enumerate(loader)):
                labels = contents[0].float().to(self._device)
                user_idx = contents[1].to(self._device)
                candidate_ctr = contents[3].to(self._device)
                history_ctr = contents[5].to(self._device)
                user_category_kl = contents[6].to(self._device)
                user_subcategory_kl = contents[7].to(self._device)
                impressId = contents[-1].to(self._device)

                candidate_index = contents[-3].to(self._device)

                if self._args.use_bert:
                    candidate_titles = {key: value.to(self._device) for key, value in contents[2].items()}
                    history_titles = {key: value.to(self._device) for key, value in contents[4].items()}
                else:
                    candidate_titles = contents[2].to(self._device)
                    history_titles = contents[4].to(self._device)

                input_line = {
                    'user_idx': user_idx,
                    'candidate_index': candidate_index,
                    'candidate_titles': candidate_titles,
                    'history_titles': history_titles,
                    'candidate_ctr': candidate_ctr,
                    'history_ctr': history_ctr,
                    'user_category_kl': user_category_kl,
                    'user_subcategory_kl': user_subcategory_kl
                }
                predict_info = self.model(input_line, is_train=False)

                if self._args.net.lower() == 'macr':
                    final_score, matching_score, user_score, item_score = predict_info
                    predictions = torch.multiply(
                        torch.subtract(matching_score, self._args.macr_c),
                        torch.multiply(F.sigmoid(user_score), F.sigmoid(item_score))
                    )
                elif self._args.net.lower() == 'pda':
                    final_score, activated_matching, matching_score = predict_info
                    predictions = activated_matching
                elif self._args.net.lower() == 'tide':
                    final_score, matching_score, weight = predict_info
                    predictions = final_score
                elif self._args.net.lower() == 'dice':
                    final_score, matching_interest, matching_conformity, representations = predict_info
                    predictions = matching_interest
                else:
                    raise ValueError('[Error!]wrong net selection, please try again!')

                current_labels = labels.squeeze().cpu().numpy()

                current_predicts = predictions.squeeze().cpu().numpy()
                ground_targets.append(current_labels)
                preds.append(current_predicts)
                current_rank = np.argsort(current_predicts)
                current_rank = current_rank[::-1] + 1
                recoreded_rank = '[' + ','.join([str(i) for i in current_rank.tolist()]) + ']'
                predict_rank[impressId.squeeze().cpu().item()] = recoreded_rank
            
            evaluations = []
            for y_true, y_pred in tqdm(zip(ground_targets, preds)):
                try:
                    evals = cal_metric(
                        labels=y_true,
                        preds=y_pred,
                        metrics=self._args.metrics
                    )
                    evaluations.append(evals)
                except:
                    print(y_true)
                    print(y_pred)
                    exit(-1)

            dict_keys = evaluations[0].keys()
            final_evaluation = {}
            for key in dict_keys:
                final_evaluation[key] = np.mean([item[key] for item in evaluations])
    
            if not self._args.test:
                for y_true, y_pred in tqdm(zip(ground_targets, preds)):
                    try:
                        evals = cal_metric(
                            labels=y_true,
                            preds=y_pred,
                            metrics=self._args.metrics
                        )
                        evaluations.append(evals)
                    except:
                        print(y_true)
                        print(y_pred)
                        exit(-1)

                dict_keys = evaluations[0].keys()
                final_evaluation = {}
                for key in dict_keys:
                    final_evaluation[key] = np.mean([item[key] for item in evaluations])
            else:
                final_evaluation = {}

        write_file(self._args.log_path, f'predict_rank_{self._args.name}.txt', predict_rank)

        return final_evaluation, predict_rank

    def save_model(self, name='dev'):
        model_dict = {}
        model_dict['state_dict'] = self.model.state_dict()
        model_dict['m_config'] = self._args
        model_dict['optimizer'] = self._optimizer.state_dict()
        if name is None:
            ckpt_path = self._ckpt_path + '.pth'
        else:
            ckpt_path = self._ckpt_path + '_' + name + '.pth'

        if not os.path.exists(os.path.dirname(ckpt_path)):
            os.makedirs(os.path.dirname(ckpt_path))
        torch.save(model_dict, ckpt_path)
        print(f'> model have been saved at {ckpt_path}')

    def load_model(self, name='dev'):
        ckpt_path = self._ckpt_path + '_' + name + '.pth'
        print(f'load checkpoint {ckpt_path}')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self._device)
        else:
            raise FileNotFoundError(f'checkpoint has not been found in {ckpt_path}')
        try:
            self.model.load_state_dict(checkpoint['state_dict'])
        except:
            self.model = nn.DataParallel(self.model)
            self.model.load_state_dict(checkpoint['state_dict'])

        print(f'> model have been load at {ckpt_path}')
