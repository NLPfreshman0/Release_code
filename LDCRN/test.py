import torch
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
from tqdm import tqdm

def setup_device(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()
    
def setup_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
def test(args, model, test_dataloader):
    model.load_state_dict(torch.load(args.ckpt_file, map_location='cpu'))
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))
    model.eval()
    overall = [0, 0, 0]
    per_class = [0, 0, 0]
    preds = []
    labels = []
    with torch.no_grad():
        loop = tqdm(test_dataloader, total = len(test_dataloader))
        for batch in loop:
            try:
                _, pred, label = model(batch, inference=True)
                preds.extend(pred.cpu().numpy())
                labels.extend(label.cpu().numpy())
            except:
                print(batch)

    for i in range(len(labels)):
        overall[labels[i]] += 1
        if labels[i] == preds[i]:
            per_class[labels[i]] += 1
    accuracy = sum(per_class) / sum(overall)
    e_accuracy = per_class[0] / overall[0]
    n_accuracy = per_class[1] / overall[1]
    c_accuracy = per_class[2] / overall[2]
    print('overall_accuracy:', accuracy)
    print('e_accuracy:', e_accuracy, 'n_accuracy:', n_accuracy, 'c_accuracy:', c_accuracy)
        
        
if __name__ == '__main__':
    args = parse_args()
    setup_device(args)
    setup_seed(args)
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(args, confactual=args.confactual)
    hard_dataloader = create_hard_dataloaders(args, confactual=args.confactual)
    matched_dataloader, mismatched_dataloader = create_MultiNLI_hard_dataloaders(args, confactual=args.confactual)
    model = Model(args)
    print('val_dataset')
    test(args, model, val_dataloader)
    print('test_dataset')
    test(args, model, test_dataloader)
    if args.task_name == 'SNLI':
        print('hard_dataset')
        test(args, model, hard_dataloader)
    else:
        print('matched_dataset')
        test(args, model, matched_dataloader)
        print('mismatched_dataset')
        test(args, model, mismatched_dataloader)
