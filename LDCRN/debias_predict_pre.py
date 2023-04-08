from config import parse_args
from model import Model
from load_datasets import create_dataloaders, create_hard_dataloaders, create_MultiNLI_hard_dataloaders
import logging
import os
import time
import torch
from tqdm import tqdm
from optimizer import build_optimizer
import scipy.stats
import random
import numpy as np

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
_, masked_confactual_val_dataloader, masked_confactual_test_dataloader = create_dataloaders(args, confactual=1)
_, onlyhy_confactual_val_dataloader, onlyhy_confactual_test_dataloader = create_dataloaders(args, confactual=2)
_, factual_val_dataloader, factual_test_dataloader = create_dataloaders(args)

masked_confactual_hard_dataloader, _ = create_MultiNLI_hard_dataloaders(args, confactual=1)
onlyhy_confactual_hard_dataloader, _ = create_MultiNLI_hard_dataloaders(args, confactual=2)
factual_hard_dataloader, _ = create_MultiNLI_hard_dataloaders(args)
#masked_confactual_hard_dataloader = create_hard_dataloaders(args, confactual=1)
#onlyhy_confactual_hard_dataloader = create_hard_dataloaders(args, confactual=2)
#factual_hard_dataloader = create_hard_dataloaders(args)

#masked_model_path = "/data/zhangdacao/save/maskpre_snli_baseline/bert-base-uncased_pair/model.bin"
#onlyhy_model_path = "/data/zhangdacao/save/onlyhy_snli_baseline/bert-base-uncased_pair/model.bin"
#factual_model_path = "/data/zhangdacao/save/snli_baseline/bert-base-uncased_pair/model.bin"
masked_model_path = "/data/zhangdacao/save/maskpre_multinli_baseline/bert-base-uncased_pair/model.bin"
onlyhy_model_path = "/data/zhangdacao/save/onlyhy_multinli_baseline/bert-base-uncased_pair/model.bin"
factual_model_path = "/data/zhangdacao/save/multinli_baseline/bert-base-uncased_pair/model.bin"
masked_model = Model(args)
onlyhy_model = Model(args)
factual_model = Model(args)
masked_model.load_state_dict(torch.load(masked_model_path, map_location='cpu'))
onlyhy_model.load_state_dict(torch.load(onlyhy_model_path, map_location='cpu'))
factual_model.load_state_dict(torch.load(factual_model_path, map_location='cpu'))

if args.device == 'cuda':
    masked_model = torch.nn.parallel.DataParallel(masked_model.to(args.device))
    onlyhy_model = torch.nn.parallel.DataParallel(onlyhy_model.to(args.device))
    factual_model = torch.nn.parallel.DataParallel(factual_model.to(args.device))
masked_model.eval()
onlyhy_model.eval()
factual_model.eval()

def confactual_prediction(masked_dataloader, onlyhy_dataloader, factual_dataloader, a1, a2):
    overall = [0, 0, 0]
    per_class = [0, 0, 0]
    preds = []
    labels = []
    with torch.no_grad():
        loop = tqdm(zip(masked_dataloader, onlyhy_dataloader, factual_dataloader), total = len(factual_dataloader))
        for (batch1, batch2, batch3) in loop:
            masked_logits, _, label = masked_model(batch1, inference=True)
            onlyhy_logits, _, _ = onlyhy_model(batch2, inference=True)
            factual_logits, _, _ = factual_model(batch3, inference=True)
            logits = factual_logits - a1 * masked_logits - a2 * onlyhy_logits
            #logits = onlyhy_logits
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
confactual_prediction(masked_confactual_val_dataloader, onlyhy_confactual_val_dataloader, factual_val_dataloader, 0.1, 0.9)
confactual_prediction(masked_confactual_test_dataloader, onlyhy_confactual_test_dataloader, factual_test_dataloader, 0.1, 0.9)
confactual_prediction(masked_confactual_hard_dataloader, onlyhy_confactual_hard_dataloader, factual_hard_dataloader, 0.1, 0.9)

#confactual_prediction(masked_confactual_val_dataloader, onlyhy_confactual_val_dataloader, factual_val_dataloader, 0.1, 0.9)
#confactual_prediction(masked_confactual_test_dataloader, onlyhy_confactual_test_dataloader, factual_test_dataloader, 0.1, 0.9)
#confactual_prediction(masked_confactual_hard_dataloader, onlyhy_confactual_hard_dataloader, factual_hard_dataloader, 0.1, 0.9)

#confactual_prediction(masked_confactual_val_dataloader, onlyhy_confactual_val_dataloader, factual_val_dataloader, 0.3, -0.3)
#confactual_prediction(masked_confactual_test_dataloader, onlyhy_confactual_test_dataloader, factual_test_dataloader, 0.3, -0.3)
#confactual_prediction(masked_confactual_hard_dataloader, onlyhy_confactual_hard_dataloader, factual_hard_dataloader, 0.3, -0.3)
#confactual_prediction(masked_confactual_test_dataloader, onlyhy_confactual_test_dataloader, factual_test_dataloader, -0.2, 0.1)

def grid_search(masked_dataloader, onlyhy_dataloader, factual_dataloader):
    grid_map = {}
    best_x, best_y, best_acc = 0, 0, 0
    for i in np.arange(-1, 1.1, 0.1):
        for j in np.arange(-1, 1.1, 0.1):
            print(i, j)
            cur_acc = confactual_prediction(masked_dataloader, onlyhy_dataloader, factual_dataloader, i, j)
            grid_map[str(i)+'_'+str(j)] = cur_acc
            if cur_acc > best_acc:
                best_acc = cur_acc
                best_x = i
                best_y = j
    print(best_x, best_y, best_acc)
    np.save('data/multinli_hard_grid_map.npy', grid_map)
    return best_x, best_y, best_acc

#grid_search(masked_confactual_val_dataloader, onlyhy_confactual_val_dataloader, factual_val_dataloader)
grid_search(masked_confactual_hard_dataloader, onlyhy_confactual_hard_dataloader, factual_hard_dataloader)
    
