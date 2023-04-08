from config import parse_args
from model import Model
from load_datasets import create_dataloaders, create_hard_dataloaders, create_MultiNLI_hard_dataloaders, create_snli_ve_hard_dataloaders, create_GTE_hard
import logging
import os
import time
import torch
from tqdm import tqdm
from optimizer import build_optimizer
import scipy.stats
import random
import numpy as np
import torch.nn.functional as F

def setup_device(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()
    
def setup_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

args = parse_args()
setup_device(args)
setup_seed(args)
_, val_dataloader, test_dataloader = create_dataloaders(args)
hard_dataloader = create_hard_dataloaders(args)

masked_confactual_hard_dataloader, _ = create_MultiNLI_hard_dataloaders(args, confactual=1)
onlyhy_confactual_hard_dataloader, _ = create_MultiNLI_hard_dataloaders(args, confactual=2)
factual_hard_dataloader, _ = create_MultiNLI_hard_dataloaders(args)
match_hard, mismatch_hard = create_MultiNLI_hard_dataloaders(args)
snli_ve_hard = create_snli_ve_hard_dataloaders(args, confactual=args.confactual)
GTE_hard = create_GTE_hard(args)

model_path = "/data/zhangdacao/save/GTE_confactual/clip-CON-roberta-base_withitc/model.bin"
#model_path = "/data/zhangdacao/save/lambda_snli_baseline_fusion/3lambda_bert-base-uncased_pair/model.bin"
#model_path = "/data/zhangdacao/save/confactual_multinli_baseline_fusion/roberta-base_pair/model.bin"
model = Model(args)

model.load_state_dict(torch.load(model_path, map_location='cpu'))

if args.device == 'cuda':
    model = torch.nn.parallel.DataParallel(model.to(args.device))

model.eval()


def confactual_prediction(dataloader, a):
    overall = [0, 0, 0]
    per_class = [0, 0, 0]
    preds = []
    labels = []
    with torch.no_grad():
        loop = tqdm(dataloader, total = len(dataloader))
        for batch in loop:
            factual_logits, confactual_logits, _, label = model(batch, inference=True)
            logits = factual_logits - a * confactual_logits
            #logits = confactual_logits
            logits = F.softmax(logits, dim=-1)
            #logits = confactual_logits
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().numpy())
            labels.extend(label.cpu().numpy())

        for i in range(len(labels)):
            overall[labels[i]] += 1
            if labels[i] == preds[i]:
                per_class[labels[i]] += 1

        #print(overall)
        accuracy = sum(per_class) / sum(overall)
        e_accuracy = per_class[0] / overall[0]
        n_accuracy = per_class[1] / overall[1]
        c_accuracy = per_class[2] / overall[2]
        print('overall_accuracy:', accuracy)
        print('e_accuracy:', e_accuracy, 'n_accuracy:', n_accuracy, 'c_accuracy:', c_accuracy)
    return accuracy
#confactual_prediction(val_dataloader, 1)
#confactual_prediction(test_dataloader, 1)

def grid_search(val_dataloader):
    grid_map = {}
    best_x, best_acc = 0, 0
    for i in np.arange(-2, 2.1, 0.1):
        print(i)
        cur_acc = confactual_prediction(val_dataloader, i)
        grid_map[str(i)] = cur_acc
        if cur_acc > best_acc:
            best_acc = cur_acc
            best_x = i
    print(best_x, best_acc)
    np.save('data/snli_confactual_hard_grid_map.npy', grid_map)
    return best_x, best_acc

a = [1]
best_rate, best_score = 0, 0 
#a = [1.1, 1.2, 1.3, 1.4, 1.5]
for rate in a:
    print(rate)
    confactual_prediction(val_dataloader, rate)
    confactual_prediction(test_dataloader, rate)
    acc = confactual_prediction(GTE_hard, rate)
    if acc > best_score:
        best_rate = rate
        best_score = acc
print(best_rate, best_score)
#confactual_prediction(val_dataloader, args.debias_rate)
#confactual_prediction(test_dataloader, args.debias_rate)
#confactual_prediction(hard_dataloader, args.debias_rate)
#confactual_prediction(match_hard, 0)
#confactual_prediction(mismatch_hard, 0)
    
