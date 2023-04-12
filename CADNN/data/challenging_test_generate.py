import math
import time
from data_utils import *

base_path = '../dataset'
data_choose = '/small'  # two different selection: large, small


conformity_type = 'perturb'


def probCalculat(news_list, news_count):
    news_prob = []
    for item in news_list:
        news, sign = item.split ('-')
        count = news_count[news]
        if count == 1:
            prob = 0.8
        else:
            prob = 1.0 / count

        news_prob.append (prob)

    norm_prob = []
    value = sum (news_prob)
    for item in news_prob:
        norm_prob.append (item / value)

    return norm_prob


def positive_extract(content):
    p_list = []
    for item in content:
        news, sign = item.split ('-')
        if sign == '1':
            p_list.append (item)

    return p_list


def weighted_newsampling(content, news_count, test_type='remove'):
    if len (content) < 6:
        sampled_news = content
    else:
        prob = probCalculat (content, news_count)

        current_news = np.random.choice (content, size=int (round (len (content) * 0.7)), replace=False, p=prob)
        current_news = list (current_news)

        if test_type == 'remove':
            sampled_news = None
        else:
            p_list = positive_extract (content)
            sample_value = random.choice ([1, 2, 3, 4])
            if sample_value >= len (p_list):
                results = p_list
            else:
                results = list (np.random.choice (p_list, size=sample_value, replace=False))
            for p in results:
                current_news.append (p)
            sampled_news = current_news

    return sampled_news


def process_rec_list(contents, split='train'):
    positive_list = []
    negative_list = []
    if isinstance (contents, str):
        contents = contents.strip ().split (' ')
    for item in contents:
        news, sign = item.split ('-')
        sign = int (sign)
        if sign == 1:
            positive_list.append (news)
        else:
            negative_list.append (news)

    return positive_list, negative_list



def dict_count(dic, content):
    for item in content:
        if item in dic.keys ():
            dic[item] += 1
        else:
            dic[item] = 1

    return dic


def generate_fair_user_file(file_name):
    split = 'test_split'
    name = split + '/behaviors.tsv'
    candidate_count = {}
    results = []
    invalid_record_count = 0
    with open (os.path.join (file_name, data_choose, name), 'r', encoding='utf8') as read:
        lines = read.readlines ()

        for line in tqdm (lines):
            tokens = line.strip ().split ('\t')
            candidates = tokens[4]
            positive_list, negative_list = process_rec_list (candidates, 'valid')
            
            candidate_count = dict_count (candidate_count, positive_list)
            candidate_count = dict_count (candidate_count, negative_list)
            

        for line in tqdm (lines):
            tokens = line.strip ().split ('\t')
            candidates = tokens[4].split (' ')
    
            sampled_candidates = weighted_newsampling (candidates, candidate_count, 'random')
            if sampled_candidates is None:
                continue
            tokens[4] = ' '.join (sampled_candidates)
            modified_line = '\t'.join (tokens)
            results.append (modified_line)

    with open (os.path.join (file_name, data_choose, 'test/behaviors.tsv'), 'w', encoding='utf8') as w:
        for item in tqdm (results):
            w.write (item + '\n')


if __name__ == '__main__':
    print (f'generating fair data with weighted sampling method')
    generate_fair_user_file (base_path)
