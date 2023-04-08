from datasets import load_dataset
import numpy as np
import jsonlines
from tqdm import tqdm

#process MultiNLI dataset
dataset = load_dataset("multi_nli")
splits = ['train', 'validation_matched', 'validation_mismatched']
for split in splits:
    data = []
    for i in range(len(dataset[split])):
        if dataset[split][i]['label'] in [0, 1, 2]:
            data.append({'premise':dataset[split][i]['premise'], 'hypothesis':dataset[split][i]['hypothesis'], 'label':dataset[split][i]['label']})
    np.save('dataset/multi_nli/'+split+'.npy', data)
    
#process snli dataset
dataset = load_dataset('snli')
splits = ['train', 'validation', 'test']
for split in splits:
    data = []
    for i in range(len(dataset[split])):
        if dataset[split][i]['label'] in [0, 1, 2]:
            data.append({'premise':dataset[split][i]['premise'], 'hypothesis':dataset[split][i]['hypothesis'], 'label':dataset[split][i]['label']})
    np.save('dataset/snli/'+split+'.npy', data)
