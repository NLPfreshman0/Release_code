from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import jsonlines
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
import pandas as pd

#停用词
#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

top_K = 50
use_stop_words = True
full_datasets = np.load('/data/zhangdacao/dataset/multi_nli/validation_matched.npy', allow_pickle='TRUE')
hard_datasets = np.load('/data/zhangdacao/dataset/multi_nli/matched_hard.npy', allow_pickle='TRUE')
    
# 定义一个默认字典，用于统计词语的词频
full_pre_word_counts_e = defaultdict(int)
full_pre_word_counts_n = defaultdict(int)
full_pre_word_counts_c = defaultdict(int)
full_hy_word_counts_e = defaultdict(int)
full_hy_word_counts_n = defaultdict(int)
full_hy_word_counts_c = defaultdict(int)
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
    
    if label == 0:
        for word in pre_words:
            full_pre_word_counts_e[word] += 1
        for word in hy_words:
            full_hy_word_counts_e[word] += 1
    elif label == 1:
        for word in pre_words:
            full_pre_word_counts_n[word] += 1
        for word in hy_words:
            full_hy_word_counts_n[word] += 1
    elif label == 2:
        for word in pre_words:
            full_pre_word_counts_c[word] += 1
        for word in hy_words:
            full_hy_word_counts_c[word] += 1

hard_pre_word_counts_e = defaultdict(int)
hard_pre_word_counts_n = defaultdict(int)
hard_pre_word_counts_c = defaultdict(int)
hard_hy_word_counts_e = defaultdict(int)
hard_hy_word_counts_n = defaultdict(int)
hard_hy_word_counts_c = defaultdict(int)
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
    if label == 0:
        for word in pre_words:
            hard_pre_word_counts_e[word] += 1
        for word in hy_words:
            hard_hy_word_counts_e[word] += 1
    elif label == 1:
        for word in pre_words:
            hard_pre_word_counts_n[word] += 1
        for word in hy_words:
            hard_hy_word_counts_n[word] += 1
    elif label == 2:
        for word in pre_words:
            hard_pre_word_counts_c[word] += 1
        for word in hy_words:
            hard_hy_word_counts_c[word] += 1


# 按字典的值排序
full_pre_e_sorted_words = sorted(full_pre_word_counts_e.items(), key=lambda x: x[1], reverse=True)
full_pre_n_sorted_words = sorted(full_pre_word_counts_n.items(), key=lambda x: x[1], reverse=True)
full_pre_c_sorted_words = sorted(full_pre_word_counts_c.items(), key=lambda x: x[1], reverse=True)
full_hy_e_sorted_words = sorted(full_hy_word_counts_e.items(), key=lambda x: x[1], reverse=True)
full_hy_n_sorted_words = sorted(full_hy_word_counts_n.items(), key=lambda x: x[1], reverse=True)
full_hy_c_sorted_words = sorted(full_hy_word_counts_c.items(), key=lambda x: x[1], reverse=True)

hard_pre_e_sorted_words = sorted(hard_pre_word_counts_e.items(), key=lambda x: x[1], reverse=True)
hard_pre_n_sorted_words = sorted(hard_pre_word_counts_n.items(), key=lambda x: x[1], reverse=True)
hard_pre_c_sorted_words = sorted(hard_pre_word_counts_c.items(), key=lambda x: x[1], reverse=True)
hard_hy_e_sorted_words = sorted(hard_hy_word_counts_e.items(), key=lambda x: x[1], reverse=True)
hard_hy_n_sorted_words = sorted(hard_hy_word_counts_n.items(), key=lambda x: x[1], reverse=True)
hard_hy_c_sorted_words = sorted(hard_hy_word_counts_c.items(), key=lambda x: x[1], reverse=True)


# 打印排序后的词语
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

# 使用to_csv函数保存DataFrame到CSV文件
#df.to_csv('data/MultiNLI_word_counts_with_stopwords.csv', index=False)
df.to_csv('data/MultiNLI_word_counts_without_stopwords.csv', index=False)

