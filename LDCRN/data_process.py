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
        if dataset['train'][i]['label'] in [0, 1, 2]:
            data.append({'premise':dataset[split][i]['premise'], 'hypothesis':dataset[split][i]['hypothesis'], 'label':dataset[split][i]['label']})
    np.save('/data/zhangdacao/dataset/multi_nli/'+split+'.npy', data)
"""#process snli dataset
dataset = load_dataset('snli')
splits = ['train', 'validation', 'test']
for split in splits:
    data = []
    for i in range(len(dataset[split])):
        if dataset['train'][i]['label'] in [0, 1, 2]:
            data.append({'premise':dataset[split][i]['premise'], 'hypothesis':dataset[split][i]['hypothesis'], 'label':dataset[split][i]['label']})
    np.save('/data/zhangdacao/dataset/snli/'+split+'.npy', data)

#process snli-ve dataset
v_dataset = load_dataset('carlosejimenez/flickr30k_images_SimCLRv2')
image_dict = {}
label_dict = {'entailment':0, 'neutral':1, 'contradiction':2}
loop = tqdm(v_dataset['train'], total = len(v_dataset['train']))
for image in loop:
    idx = image['image_id'].split('/')[-1][:-4]
    emb = image['embedding']
    image_dict[idx] = emb
np.save('/data/zhangdacao/dataset/snli_ve/image.npy', image_dict)


image_dict = np.load('/data/zhangdacao/dataset/snli_ve/image.npy', allow_pickle='TRUE').item()
print('start_split')
splits = ['train', 'dev', 'test']
for split in splits:
    ve_data = []
    with open('/data/zhangdacao/dataset/snli_ve/data/snli_ve_'+split+'.jsonl') as jsonl_file:
        for line in jsonlines.Reader(jsonl_file):
            Flikr30kID = str(line['Flickr30K_ID'])
            gold_label = str(line['gold_label'])
            hypothesis = str(line['sentence2'])
            ve_data.append({'premise':Flikr30kID, 'hypothesis':hypothesis, 'label':label_dict[gold_label]})
    if split == 'dev':
        split = 'validation'
    np.save('/data/zhangdacao/dataset/snli_ve/'+split+'.npy', ve_data)"""

            