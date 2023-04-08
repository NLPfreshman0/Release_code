from config import parse_args
from model import Model
from load_datasets import create_dataloaders
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

def validate(model, val_dataloader):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in val_dataloader:
            if args.confactual_train:
                _, _, pred, label = model(batch, inference=True)
            else:
                _, pred, label = model(batch, inference=True)
            preds.extend(pred.cpu().numpy())
            labels.extend(label.cpu().numpy())
    accuracy = np.sum(np.array(preds) == np.array(labels)) / len(labels)
    model.train()
    return accuracy

def train(args):
     # 1. load data
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(args, confactual=args.confactual)

    # 2. build model and optimizers
    model = Model(args)
    optimizer, scheduler = build_optimizer(args, model)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))

    # 3. training
    step = 0
    best_score = args.best_score
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs
    model.train()
    
    for epoch in range(args.max_epochs):
        loop = tqdm(train_dataloader, total = len(train_dataloader), ncols=100)
        for batch in loop:
            loss = model(batch)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            loop.set_description(f'Epoch [{epoch}]')
            loop.set_postfix(loss=loss.item(), valid=best_score)

            step += 1
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, valid {best_score:.3f}, lr {optimizer.param_groups[0]['lr']:.3e}")

            # 4. validation
            if step == 20 or step % args.valid_steps == 0:
                valid = validate(model, val_dataloader)
                logging.info(f"Epoch {epoch} step {step}: valid {valid:.3f}")

                # 5. save checkpoint
                if valid > best_score:
                    best_score = valid
                    torch.save(model.module.state_dict(), f'{args.savedmodel_path}/model_epoch_{epoch}_valid_{valid}.bin')
                    
    valid = validate(model, val_dataloader)
    logging.info(f"Epoch {epoch} step {step}: valid {valid:.3f}")
    # 5. save checkpoint
    if valid > best_score:
        best_score = valid
        torch.save(model.module.state_dict(), f'{args.savedmodel_path}/model_epoch_{epoch}_valid_{valid}.bin')
        
    torch.save(model.module.state_dict(), f'{args.savedmodel_path}/model_final.bin')

if __name__ == '__main__':
    args = parse_args()
    setup_device(args)
    setup_seed(args)
    os.makedirs(args.savedmodel_path, exist_ok=True)
    os.makedirs('train_log', exist_ok=True)
    localTime = time.strftime("%Y%m%d%H%M%S", time.localtime())
    log_name = 'train_log/' + localTime + ' train.log'
    logging.basicConfig(filename=log_name, filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.INFO)
    argsDict = args.__dict__
    logging.info('------------------ start ------------------' + '\n')
    for eachArg, value in argsDict.items():
        logging.info(eachArg + ' : ' + str(value))
    logging.info('------------------- end -------------------')
    train(args)
    # test = validate(model, test_dataloader)
    # logging.info(f"Test {test:.3f}")