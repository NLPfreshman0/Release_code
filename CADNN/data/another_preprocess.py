from pathlib import Path
import time
import datetime
from tqdm import tqdm
from collections import defaultdict, Counter
import copy
import random
import re
import numpy as np
import os
import pickle

data_path = '../dataset'
data_choose = 'small' 
glove_path = '../dataset/glove_embedding'

npratio = 4
max_his_len = 50
min_word_cnt = 5
max_title_len = 30

news_info = {"<unk>": ""}
nid2index = {"<unk>": 0}
word_cnt = Counter()


def word_tokenize(sent):
    pat = re.compile(r"[\w]+|[.,!?;|]")
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []

for l in tqdm(open(os.path.join(data_path, data_choose, 'train', 'news.tsv'), "r", encoding='utf8')):
    nid, vert, subvert, title, abst, url, ten, aen = l.strip("\n").split("\t")
    if nid in nid2index:
        continue
    title = word_tokenize(title)[:max_title_len]
    nid2index[nid] = len(nid2index)
    news_info[nid] = title
    word_cnt.update(title)

for l in tqdm(open(os.path.join(data_path, data_choose, 'valid', 'news.tsv'), "r", encoding='utf-8')):
    nid, vert, subvert, title, abst, url, ten, aen = l.strip("\n").split("\t")
    if nid in nid2index:
        continue
    title = word_tokenize(title)[:max_title_len]
    nid2index[nid] = len(nid2index)
    news_info[nid] = title
    word_cnt.update(title)

with open(os.path.join(data_path, data_choose, "nid2idx.pkl"), "wb") as f:
    pickle.dump(nid2index, f)

with open(os.path.join(data_path, data_choose, "news_info.pkl"), "wb") as f:
    pickle.dump(news_info, f)

if os.path.exists(os.path.join(data_path, data_choose, "test") ):
    test_news_info = {"<unk>": ""}
    test_nid2index = {"<unk>": 0}
    for l in tqdm(open(os.path.join(data_path, data_choose, "test", "news.tsv"), "r", encoding='utf-8')):
        nid, vert, subvert, title, abst, url, ten, aen = l.strip("\n").split("\t")
        if nid in test_nid2index:
            continue
        title = word_tokenize(title)[:max_title_len]
        test_nid2index[nid] = len(test_nid2index)
        test_news_info[nid] = title

    with open(os.path.join(data_path, data_choose, "test_nid2idx.pkl"), "wb") as f:
        pickle.dump(test_nid2index, f)

    with open(os.path.join(data_path, data_choose, "test_news_info.pkl"), "wb") as f:
        pickle.dump(test_news_info, f)

vocab_dict = {"<unk>": 0}

for w, c in tqdm(word_cnt.items()):
    if c >= min_word_cnt:
        vocab_dict[w] = len(vocab_dict)

with open(os.path.join(data_path, data_choose, "word2idx.pkl"), "wb") as f:
    pickle.dump(vocab_dict, f)

news_index = np.zeros((len(news_info) + 1, max_title_len), dtype="float32")

for nid in tqdm(nid2index):
    news_index[nid2index[nid]] = [
        vocab_dict[w] if w in vocab_dict else 0 for w in news_info[nid]
    ] + [0] * (max_title_len - len(news_info[nid]))

np.save(os.path.join(data_path, data_choose, "news_index"), news_index)

if os.path.exists(os.path.join(data_path, data_choose, "test")):
    test_news_index = np.zeros((len(test_news_info) + 1, max_title_len), dtype="float32")

    for nid in tqdm(test_nid2index):
        test_news_index[test_nid2index[nid]] = [
            vocab_dict[w] if w in vocab_dict else 0 for w in test_news_info[nid]
        ] + [0] * (max_title_len - len(test_news_info[nid]))

    np.save(os.path.join(data_path, data_choose, "test_news_index"), test_news_index)


def load_matrix(glove_path, word_dict):
    embedding_matrix = np.zeros((len(word_dict) + 1, 300))
    exist_word = []
    with open(glove_path, "rb") as f:
        for l in tqdm(f):
            l = l.split()
            word = l[0].decode()
            if len(word) != 0 and word in word_dict:
                wordvec = [float(x) for x in l[1:]]
                index = word_dict[word]
                embedding_matrix[index] = np.array(wordvec)
                exist_word.append(word)
    return embedding_matrix, exist_word


embedding_matrix, exist_word = load_matrix(os.path.join(glove_path, 'glove.840B.300d.txt'), vocab_dict)

print(embedding_matrix.shape[0], len(exist_word))

np.save(os.path.join(data_path, data_choose, "word2vec.npy"), embedding_matrix)
