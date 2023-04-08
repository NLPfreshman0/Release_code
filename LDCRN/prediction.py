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

train_dataloader, val_dataloader, test_dataloader = create_dataloaders(args)
snli_hard_dataloader = create_hard_dataloaders(args)

#model_path = "/data/zhangdacao/save/snli_baseline/bert-base-uncased_pair/model.bin"
#model_path = "/data/zhangdacao/save/snli_baseline/unsup_simcse_roberta_pair/model.bin"
model_path = "/data/zhangdacao/save/snli_baseline/sncse_roberta_pair/model.bin"
model = Model(args)

model.load_state_dict(torch.load(model_path, map_location='cpu'))

if args.device == 'cuda':
    model = torch.nn.parallel.DataParallel(model.to(args.device))

model.eval()


def predict(dataloader):
    overall = [0, 0, 0]
    per_class = [0, 0, 0]
    preds = []
    labels = []
    with torch.no_grad():
        loop = tqdm(dataloader, total = len(dataloader))
        for batch in loop:
            logits, pred, label = model(batch, inference=True)
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

predict(val_dataloader)
predict(test_dataloader)
predict(snli_hard_dataloader)