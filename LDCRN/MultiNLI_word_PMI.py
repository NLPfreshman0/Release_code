from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import jsonlines
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
import pandas as pd

#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

top_K = 50
use_stop_words = True
full_datasets = np.load('/data/zhangdacao/dataset/multi_nli/validation_matched.npy', allow_pickle='TRUE')
hard_datasets = np.load('/data/zhangdacao/dataset/multi_nli/matched_hard.npy', allow_pickle='TRUE')
    
full_pre_word_counts_e = defaultdict(int)
full_pre_word_counts_n = defaultdict(int)
full_pre_word_counts_c = defaultdict(int)
full_hy_word_counts_e = defaultdict(int)
full_hy_word_counts_n = defaultdict(int)
full_hy_word_counts_c = defaultdict(int)
all_num, en_num, ne_num, con_num = 0, 0, 0, 0
for sample in full_datasets:
    pre = sample['premise'].lower()
    pre = pre.translate(str.maketrans('', '', string.punctuation))
    if use_stop_words:
        pre_words = [w for w in pre.split() if not w in stop_words]
    else:
        pre_words = pre.split()
    hy = sample['hypothesis'].lower()
    hy = hy.translate(str.maketrans('', '', string.punctuation))
    if use_stop_words:
        hy_words = [w for w in hy.split() if not w in stop_words]
    else:
        hy_words = hy.split()
    label = sample['label']
    all_num += 1
    if label == 0:
        en_num += 1
        for word in pre_words:
            full_pre_word_counts_e[word] += 1
        for word in hy_words: 
            full_hy_word_counts_e[word] += 1
    elif label == 1:
        ne_num += 1
        for word in pre_words:
            full_pre_word_counts_n[word] += 1
        for word in hy_words:
            full_hy_word_counts_n[word] += 1
    elif label == 2:
        con_num += 1
        for word in pre_words:
            full_pre_word_counts_c[word] += 1
        for word in hy_words:
            full_hy_word_counts_c[word] += 1
            
full_pre_PMI_e = defaultdict(int)
full_pre_PMI_n = defaultdict(int)
full_pre_PMI_c = defaultdict(int)
full_hy_PMI_e = defaultdict(int)
full_hy_PMI_n = defaultdict(int)
full_hy_PMI_c = defaultdict(int)

for word in full_pre_word_counts_e:
    if full_pre_word_counts_e[word] < 10:
        continue
    p_word_class = full_pre_word_counts_e[word] / all_num
    p_word = (full_pre_word_counts_e[word] + full_pre_word_counts_n[word] + full_pre_word_counts_c[word]) / all_num
    p_class = en_num / all_num
    PMI = np.log(p_word_class / (p_word * p_class))
    full_pre_PMI_e[word] = PMI

for word in full_pre_word_counts_n:
    if full_pre_word_counts_n[word] < 10:
        continue
    p_word_class = full_pre_word_counts_n[word] / all_num
    p_word = (full_pre_word_counts_e[word] + full_pre_word_counts_n[word] + full_pre_word_counts_c[word]) / all_num
    p_class = ne_num / all_num
    PMI = np.log(p_word_class / (p_word * p_class))
    full_pre_PMI_n[word] = PMI
 
for word in full_pre_word_counts_c:
    if full_pre_word_counts_c[word] < 10:
        continue
    p_word_class = full_pre_word_counts_c[word] / all_num
    p_word = (full_pre_word_counts_e[word] + full_pre_word_counts_n[word] + full_pre_word_counts_c[word]) / all_num
    p_class = con_num / all_num
    PMI = np.log(p_word_class / (p_word * p_class))
    full_pre_PMI_c[word] = PMI  
    
for word in full_hy_word_counts_e:
    if full_hy_word_counts_e[word] < 10:
        continue
    p_word_class = full_hy_word_counts_e[word] / all_num
    p_word = (full_hy_word_counts_e[word] + full_hy_word_counts_n[word] + full_hy_word_counts_c[word]) / all_num
    p_class = en_num / all_num
    PMI = np.log(p_word_class / (p_word * p_class))
    full_hy_PMI_e[word] = PMI
    
for word in full_hy_word_counts_n:
    if full_hy_word_counts_n[word] < 10:
        continue
    p_word_class = full_hy_word_counts_n[word] / all_num
    p_word = (full_hy_word_counts_e[word] + full_hy_word_counts_n[word] + full_hy_word_counts_c[word]) / all_num
    p_class = ne_num / all_num
    PMI = np.log(p_word_class / (p_word * p_class))
    full_hy_PMI_n[word] = PMI

for word in full_hy_word_counts_c:
    if full_hy_word_counts_c[word] < 10:
        continue
    p_word_class = full_hy_word_counts_c[word] / all_num
    p_word = (full_hy_word_counts_e[word] + full_hy_word_counts_n[word] + full_hy_word_counts_c[word]) / all_num
    p_class = con_num / all_num
    PMI = np.log(p_word_class / (p_word * p_class))
    full_hy_PMI_c[word] = PMI
    
hard_pre_word_counts_e = defaultdict(int)
hard_pre_word_counts_n = defaultdict(int)
hard_pre_word_counts_c = defaultdict(int)
hard_hy_word_counts_e = defaultdict(int)
hard_hy_word_counts_n = defaultdict(int)
hard_hy_word_counts_c = defaultdict(int)
all_num, en_num, ne_num, con_num = 0, 0, 0, 0
for sample in hard_datasets:
    pre = sample['premise'].lower()
    pre = pre.translate(str.maketrans('', '', string.punctuation))
    if use_stop_words:
        pre_words = [w for w in pre.split() if not w in stop_words]
    else:
        pre_words = pre.split()
    hy = sample['hypothesis'].lower()
    hy = hy.translate(str.maketrans('', '', string.punctuation))
    if use_stop_words:
        hy_words = [w for w in hy.split() if not w in stop_words]
    else:
        hy_words = hy.split()
    label = sample['label']
    all_num += 1
    if label == 0:
        en_num += 1
        for word in pre_words:
            hard_pre_word_counts_e[word] += 1
        for word in hy_words: 
            hard_hy_word_counts_e[word] += 1
    elif label == 1:
        ne_num += 1
        for word in pre_words:
            hard_pre_word_counts_n[word] += 1
        for word in hy_words:
            hard_hy_word_counts_n[word] += 1
    elif label == 2:
        con_num += 1
        for word in pre_words:
            hard_pre_word_counts_c[word] += 1
        for word in hy_words:
            hard_hy_word_counts_c[word] += 1

hard_pre_PMI_e = defaultdict(int)
hard_pre_PMI_n = defaultdict(int)
hard_pre_PMI_c = defaultdict(int)
hard_hy_PMI_e = defaultdict(int)
hard_hy_PMI_n = defaultdict(int)
hard_hy_PMI_c = defaultdict(int)
for word in hard_pre_word_counts_e:
    if hard_pre_word_counts_e[word] < 10:
        continue
    p_word_class =hard_pre_word_counts_e[word] / all_num
    p_word = (hard_pre_word_counts_e[word] + hard_pre_word_counts_n[word] + hard_pre_word_counts_c[word]) / all_num
    p_class = en_num / all_num
    PMI = np.log(p_word_class / (p_word * p_class))
    hard_pre_PMI_e[word] = PMI

for word in hard_pre_word_counts_n:
    if hard_pre_word_counts_n[word] < 10:
        continue
    p_word_class = hard_pre_word_counts_n[word] / all_num
    p_word = (hard_pre_word_counts_e[word] + hard_pre_word_counts_n[word] + hard_pre_word_counts_c[word]) / all_num
    p_class = ne_num / all_num
    PMI = np.log(p_word_class / (p_word * p_class))
    hard_pre_PMI_n[word] = PMI
 
for word in hard_pre_word_counts_c:
    if hard_pre_word_counts_c[word] < 10:
        continue
    p_word_class =hard_pre_word_counts_c[word] / all_num
    p_word = (hard_pre_word_counts_e[word] + hard_pre_word_counts_n[word] + hard_pre_word_counts_c[word]) / all_num
    p_class = con_num / all_num
    PMI = np.log(p_word_class / (p_word * p_class))
    hard_pre_PMI_c[word] = PMI  
    
for word in hard_hy_word_counts_e:
    if hard_hy_word_counts_e[word] < 10:
        continue
    p_word_class = hard_hy_word_counts_e[word] / all_num
    p_word = (hard_hy_word_counts_e[word] + hard_hy_word_counts_n[word] + hard_hy_word_counts_c[word]) / all_num
    p_class = en_num / all_num
    PMI = np.log(p_word_class / (p_word * p_class))
    hard_hy_PMI_e[word] = PMI
    
for word in hard_hy_word_counts_n:
    if hard_hy_word_counts_n[word] < 10:
        continue
    p_word_class = hard_hy_word_counts_n[word] / all_num
    p_word = (hard_hy_word_counts_e[word] + hard_hy_word_counts_n[word] + hard_hy_word_counts_c[word]) / all_num
    p_class = ne_num / all_num
    PMI = np.log(p_word_class / (p_word * p_class))
    hard_hy_PMI_n[word] = PMI

for word in hard_hy_word_counts_c:
    if hard_hy_word_counts_c[word] < 5:
        continue
    p_word_class = hard_hy_word_counts_c[word] / all_num
    p_word = (hard_hy_word_counts_e[word] + hard_hy_word_counts_n[word] + hard_hy_word_counts_c[word]) / all_num
    p_class = con_num / all_num
    PMI = np.log(p_word_class / (p_word * p_class))
    hard_hy_PMI_c[word] = PMI

print(len(hard_hy_word_counts_c), all_num, en_num, ne_num, con_num)
full_pre_e_sorted_words = sorted(full_pre_PMI_e.items(), key=lambda x: x[1], reverse=True)
full_pre_n_sorted_words = sorted(full_pre_PMI_n.items(), key=lambda x: x[1], reverse=True)
full_pre_c_sorted_words = sorted(full_pre_PMI_c.items(), key=lambda x: x[1], reverse=True)
full_hy_e_sorted_words = sorted(full_hy_PMI_e.items(), key=lambda x: x[1], reverse=True)
full_hy_n_sorted_words = sorted(full_hy_PMI_n.items(), key=lambda x: x[1], reverse=True)
full_hy_c_sorted_words = sorted(full_hy_PMI_c.items(), key=lambda x: x[1], reverse=True)

hard_pre_e_sorted_words = sorted(hard_pre_PMI_e.items(), key=lambda x: x[1], reverse=True)
hard_pre_n_sorted_words = sorted(hard_pre_PMI_n.items(), key=lambda x: x[1], reverse=True)
hard_pre_c_sorted_words = sorted(hard_pre_PMI_c.items(), key=lambda x: x[1], reverse=True)
hard_hy_e_sorted_words = sorted(hard_hy_PMI_e.items(), key=lambda x: x[1], reverse=True)
hard_hy_n_sorted_words = sorted(hard_hy_PMI_n.items(), key=lambda x: x[1], reverse=True)
hard_hy_c_sorted_words = sorted(hard_hy_PMI_c.items(), key=lambda x: x[1], reverse=True)
"""
print(full_pre_e_sorted_words[:top_K])
print(full_pre_n_sorted_words[:top_K])
print(full_pre_c_sorted_words[:top_K])
print(full_hy_e_sorted_words[:top_K])
print(full_hy_n_sorted_words[:top_K])
print(full_hy_c_sorted_words[:top_K])

print(hard_pre_e_sorted_words[:top_K])
print(hard_pre_n_sorted_words[:top_K])
print(hard_pre_c_sorted_words[:top_K])
print(hard_hy_e_sorted_words[:top_K])
print(hard_hy_n_sorted_words[:top_K])
print(hard_hy_c_sorted_words[:top_K])
"""
df = pd.DataFrame()
df['full_pre_entailment'] = full_pre_e_sorted_words[:top_K]
df['full_hy_entailment'] = full_hy_e_sorted_words[:top_K]
df['full_pre_neutral'] = full_pre_n_sorted_words[:top_K]
df['full_hy_neutral'] = full_hy_n_sorted_words[:top_K]
df['full_pre_contradiction'] = full_pre_c_sorted_words[:top_K]
df['full_hy_contradiction'] = full_hy_c_sorted_words[:top_K]

df['hard_pre_entailment'] = hard_pre_e_sorted_words[:top_K]
df['hard_hy_entailment'] = hard_hy_e_sorted_words[:top_K]
df['hard_pre_neutral'] = hard_pre_n_sorted_words[:top_K]
df['hard_hy_neutral'] = hard_hy_n_sorted_words[:top_K]
df['hard_pre_contradiction'] = hard_pre_c_sorted_words[:top_K]
df['hard_hy_contradiction'] = hard_hy_c_sorted_words[:top_K]

df.to_csv('data/MultiNLI_PMI_without_stopwords.csv', index=False)

