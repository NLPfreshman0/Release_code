from datasets import load_dataset
import numpy as np
import jsonlines
from tqdm import tqdm
    
#process GTE dataset
splits = ['train', 'dev', 'test']
label_dict = {'entailment':0, 'neutral':1, 'contradiction':2}
for split in splits:
    GTE_data = []
    with open('/data/zhangdacao/dataset/snli_ve/data/snli_ve_'+split+'.jsonl') as jsonl_file:
        for line in jsonlines.Reader(jsonl_file):
            Flickr30kID = str(line['Flickr30K_ID'])
            gold_label = str(line['gold_label'])
            premise = str(line['sentence1'])
            hypothesis = str(line['sentence2'])
            GTE_data.append({'premise':premise, 'pre_img': Flickr30kID, 'hypothesis':hypothesis, 'label':label_dict[gold_label]})
    if split == 'dev':
        split = 'validation'
    np.save('/data/zhangdacao/dataset/GTE/'+split+'.npy', GTE_data)


            
