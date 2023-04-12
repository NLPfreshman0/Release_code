import os
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

basic_path = '../dataset/small'
cache_dir = '../dataset/pretrained'
embedding_path = '../dataset/glove_embedding'


def draw_line(info):
    with plt.style.context(['science', 'ieee']):
        x_info = list(info.keys())
        x_info.sort()
        print(x_info)
        y_info = [info[item] for item in x_info]

        plt.figure(figsize=(18, 8))
        plt.plot(x_info, y_info, 'o-', color='g', label='count')
        plt.xticks(rotation=60)
        plt.legend(loc='best')
        plt.show()


def count_news_info(file_path):
    news_info = {}

    with open(os.path.join(file_path, 'news.tsv'), 'r', encoding='utf8') as r:
        lines = r.readlines()

        for line in tqdm(lines):
            tokens = line.strip().split('\t')
            newsId = tokens[0]
            category = tokens[1]
            sub_category = tokens[2]

            news_info[newsId] = [category, sub_category]

    return news_info


def topK(content, K):
    results = {}
    sort_info = sorted(content.items(), key=lambda item: item[1], reverse=True)
    for idx, info in enumerate(sort_info):
        if idx < K:
            results[info[0]] = info[1]
        else:
            break

    return results


def count_candidate_news(file_path, news_info):
    news_count = {}
    category_count = {}
    subcategory_count = {}

    def _add_op(item, count):
        if item in count.keys():
            count[item] += 1
        else:
            count[item] = 1

    with open(os.path.join(file_path, 'behaviors.tsv'), 'r', encoding='utf8') as r:
        lines = r.readlines()

        for line in tqdm(lines):
            tokens = line.strip().split('\t')
            candidate_news = tokens[4].split(' ')
            for item in candidate_news:
                newsId, sign = item.split('-')
                if sign == '1':
                    category, sub_category = news_info[newsId]

                    _add_op(newsId, news_count)
                    _add_op(category, category_count)
                    _add_op(sub_category, subcategory_count)
    

    return news_count, category_count, subcategory_count


if __name__ == '__main__':
    path = os.path.join(basic_path, 'train')
    news_info = count_news_info(path)
    news_count, category_count, subcategory_count = count_candidate_news(path, news_info)

    subcategory_count = topK(subcategory_count, 50)
    news_count = topK(news_count, 50)
    
    print('-'*20)
    draw_line(category_count)
    draw_line(subcategory_count)
    draw_line(news_count)
    print('-'*20)

