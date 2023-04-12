import os
import re

import random
import pickle
import numpy as np

from tqdm import tqdm
from scipy import stats


def write_file(file_path, file_name, content, writing_type='a'):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    with open(os.path.join(file_path, file_name), writing_type, encoding='utf8') as w:
        if isinstance(content, list):
            for item in content:
                w.write(item + '\n')
        elif isinstance(content, dict):
            for key, item in content.items():
                line = key + ':' + str(item)
                w.write(line + '\n')
        else:
            w.write(content + '\n')

    print(f'> content has been written into {file_path}')


def read_file(file_path, file_name=None, delimer=':'):
    if file_name is not None:
        name = os.path.join(file_path, file_name)
    else:
        name = file_path

    results = {}
    with open(name, 'r', encoding='utf8') as read:
        lines = read.readlines()

        for line in lines:
            tokens = line.strip().split(delimer)
            key = tokens[0]
            prob = float(tokens[-1])
            if key not in results.keys():
                results[key] = prob

    return results


def normalized_vec(vec, type='norm', original_value=None):
    if type == 'norm':
        if original_value is None:
            total = np.sum(vec)
        else:
            total = original_value
        result = vec / (total + 1e-8)
    elif type == 'regular':
        minval = 1e10
        maxval = 0

        for val in vec:
            if val < minval:
                minval = val
            if val > maxval:
                maxval = val

        result = (vec - minval) / (maxval - minval)
    else:
        raise ValueError('unseen type, please try again')
    return result


def calculate_kl(p_vector, q_vector):
    result = stats.entropy(p_vector, q_vector)

    return result


def myTanh(x):
    s1 = np.exp(x) - np.exp(-x)
    s2 = np.exp(x) + np.exp(-x)
    s = s1 / s2
    return s

def word_tokenize(sent):
    """Split sentence into word list using regex.
    Args:
        sent (str): Input sentence

    Return:
        list: word list
    """
    pat = re.compile(r"[\w]+|[.,!?;|]")
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []


def newsample(news, ratio):
    """Sample ratio samples from news list.
    If length of news is less than ratio, pad zeros.

    Args:
        news (list): input news list
        ratio (int): sample number

    Returns:
        list: output of sample list.
    """
    if ratio > len(news):
        return news + [0] * (ratio - len(news))
    else:
        return random.sample(news, ratio)


def conditionsample(news, news_ctr, threshold, ratio, condition):
    """Sample ratio samples from news list.
    If length of news is less than ratio, pad zeros.

    Args:
        news (list): input news list
        ratio (int): sample number

    Returns:
        list: output of sample list.
    """
    targets = []
    for item, ctr in zip(news, news_ctr):
        if condition == 1:
            if ctr >= threshold:
               targets.append(item)
        else:
            if ctr < threshold:
                targets.append(item)

    if ratio > len(targets):
        return targets + [0] * (ratio - len(targets))
    else:
        return random.sample(targets, ratio)

def load_dict(file_path):
    """load pickle file

    Args:
        file path (str): file path

    Returns:
        object: pickle loaded object
    """
    # print(file_path)
    with open(file_path, "rb") as f:
        return pickle.load(f)


def load_word_dict(file_path, save_path):
    with open(file_path, 'r', encoding='utf8') as read:
        lines = read.readlines()

        word2idx = {' ': 0}
        embedding_matrix = []
        for line in tqdm(lines):
            tokens = line.split()
            word = tokens[0]
            vec = [float(x) for x in tokens[1:]]
            if word not in word2idx.keys():
                word2idx[word] = len(word2idx)
                embedding_matrix.append(vec)

    with open(os.path.join(save_path, 'word2idx.pkl'), 'wb') as w:
        pickle.dump(word2idx, w)

    embedding_matrix = np.asarray(embedding_matrix, dtype=float)
    np.save(os.path.join(save_path, 'word2vec.npy'), embedding_matrix)
    # with open(os.path.join(save_path, 'embedding.pkl'), 'wb') as w:
    #     pickle.dump(embedding_matrix, w)

    print(f'> finishing processing word2idx and embedding matrix info')


def generate_id2idx(file_path, save_path):
    with open(file_path, 'r', encoding='utf8') as read:
        lines = read.readlines()
        uid2idx = {}
        for line in lines:
            tokens = line.strip().split('\t')
            usrId = tokens[1]
            if usrId not in uid2idx.keys():
                uid2idx[usrId] = len(uid2idx)

    with open(os.path.join(save_path, 'uid2idx.pkl'), 'wb') as w:
        pickle.dump(uid2idx, w)

    print(f'> finishing processing userid to index info')
