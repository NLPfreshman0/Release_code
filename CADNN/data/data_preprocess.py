import math
import time
from data_utils import *

base_path = '../dataset'
data_choose = 'small' 

conformity_type = 'perturb'

def generate_index(is_mix=False, is_read=False):
    """
    process files to generate id to index data
    :param is_mix:
    :return:
    """
    if is_read:
        uid2idx = pickle.load(open(os.path.join(base_path, data_choose, 'uid2idx.pkl'), 'rb'))
        nid2idx = pickle.load(open(os.path.join(base_path, data_choose, 'nid2idx.pkl'), 'rb'))
    else:
        if is_mix:
            user_set = ['train/behaviors.tsv', 'valid/behaviors.tsv']
            news_set = ['train/news.tsv', 'valid/news.tsv']
        else:
            user_set = ['train/behaviors.tsv']
            news_set = ['train/news.tsv']

        def _process_file(file_name, id2index, index=0, delema='\t'):
            with open(file_name, 'r', encoding='utf8') as read:
                lines = read.readlines()
                for line in tqdm(lines):
                    ids = line.strip().split(delema)[index]
                    if ids not in id2index.keys():
                        id2index[ids] = len(id2index)

            return id2index

        uid2idx = {'u0': 0}
        for name in user_set:
            uid2idx = _process_file(os.path.join(base_path, data_choose, name), uid2idx, index=1)

        nid2idx = {'n0': 0}
        for name in news_set:
            nid2idx = _process_file(os.path.join(base_path, data_choose, name), nid2idx, index=0)

        for save_name, info in zip(['uid2idx.pkl', 'nid2idx.pkl'], [uid2idx, nid2idx]):
            print(f'{save_name} information processed, total user count is {len(info)}')
            with open(os.path.join(base_path, data_choose, save_name), 'wb') as w:
                pickle.dump(info, w)

    return uid2idx, nid2idx


def conver(time_string):
    tokens = time_string.strip().split(' ')
    date = tokens[0]
    current_time = tokens[1]
    if len(tokens) == 2:
        date_info = date + ' ' + current_time
    elif len(tokens) == 3:
        sign = tokens[2]
        if sign.lower() == 'am':
            date_info = date+' '+current_time
        else:
            hour, min, sec = current_time.split(':')
            if hour == '12':
                date_info = date+' '+current_time
            else:
                hour = int(hour) + 12
                hour = str(hour)
                new_time = ':'.join([hour, min, sec])
                date_info = date+' '+new_time
    else:
        raise ValueError('unexpected value error for the data, please check the data info')

    info_data = time.strptime(date_info, '%m/%d/%Y %H:%M:%S')
    return date_info, info_data


def conver_user_line(id, content, content_keys, is_user=True):
    if not is_user:
        result = [id]
        for key in content_keys:
            if isinstance(content[key], list):
                value = ' '.join(content[key])
            elif isinstance(content[key], float):
                value = str(content[key])
            elif isinstance(content[key], int):
                value = str(content[key])
            else:
                value = content[key]
            result.append(value)
        result = '\t'.join(result)

        return result
    else:
        if isinstance(content, dict):
            results = []
            for key in content_keys:
                if key == 'impressId':
                    results.insert(1, id)
                elif key == 'impression':
                    results.append(content[key])
                elif key == 'click_history':
                    results.append(' '.join(content[key]))
                elif key == 'time':
                    results.append(content[key])
                else:
                    results.append(str(content[key]))

            results = '\t'.join(results)
            return results
        else:
            raise ValueError('[Error!] wrong data type, please try again')


def write_results(file_path, contents, is_user=True, split='train'):
    news_keys = ['category', 'sub_category', 'titile', 'content', 'ctr', 'history_click', 'prob', 'ctprob', 'sctprob']
    user_keys = ['userId', 'impressId', 'time', 'click_history', 'impression', 'category_kl', 'subcategory_kl', 'candidate_ctr']
    results = []

    for idx in tqdm(contents.keys()):
        content = contents[idx]
        if is_user:
            result = conver_user_line(idx, content, user_keys, is_user)
        else:
            result = conver_user_line(idx, content, news_keys, is_user)

        results.append(result)

    if is_user:
        write_file(os.path.join(file_path, data_choose), f'convert_{split}_behavior.tsv', results, writing_type='w')
    else:
        write_file(os.path.join(file_path, data_choose), f'convert_{split}_news.tsv', results, writing_type='w')


def process_rec_list(contents, split='train'):
    positive_list = []
    negative_list = []
    if isinstance(contents, str):
        contents = contents.strip().split(' ')
    for item in contents:
        news, sign = item.split('-')
        sign = int(sign)
        if sign == 1:
            positive_list.append(news)
        else:
            negative_list.append(news)

    return positive_list, negative_list



def collect_topic(news_info, save_path=None):
    category = {}
    sub_category = {}

    category_file = os.path.join(base_path, data_choose, 'news_cagetory.txt')
    subcategory_file = os.path.join(base_path, data_choose, 'news_sub_cagetory.txt')

    if os.path.exists(category_file) and os.path.exists(subcategory_file):
        category = read_file(category_file, delimer=':')
        sub_category = read_file(subcategory_file, delimer=':')
        return category, sub_category

    def _add_in(category, topic):
        if topic not in category.keys():
            category[topic] = 1
        else:
            category[topic] += 1

    def _cal_prob(category):
        total_count = 0
        category_prob = {'other': 0.0}
        filter_category = {}
        for key, value in category.items():
            total_count += value

        other_count = 0
        for key, value in category.items():
            if value < 10:
                other_count += value
                continue

            prob = round(value * 1.0 / (total_count * 1.0), 4)
            val = str(value) + ':' + str(prob)
            filter_category[key] = val
            category_prob[key] = prob

        prob = round(other_count * 1.0 / (total_count * 1.0), 4)
        val = str(other_count) + ':' + str(prob)
        filter_category['other'] = val
        category_prob['other'] = prob

        return filter_category, category_prob

    for ids in news_info.keys():
        info = news_info[ids]
        if isinstance(info, dict):
            categ = info['category']
            sub_categ = info['sub_category']
            _add_in(category, categ)
            _add_in(sub_category, sub_categ)
        elif isinstance(info, list):
            for item in info:
                categ = item['category']
                sub_categ = item['sub_category']
                _add_in(category, categ)
                _add_in(sub_category, sub_categ)

    category, category_prob = _cal_prob(category)
    sub_category, subcategory_prob = _cal_prob(sub_category)

    if save_path is not None:
        write_file(os.path.join(save_path, data_choose), 'news_cagetory.txt', category)
        write_file(os.path.join(save_path, data_choose), 'news_sub_cagetory.txt', sub_category)

    return category_prob, subcategory_prob


def generate_global_prob(category):
    """
    :param category: dict {category: probability}
    :return:
    """
    category_index = {}
    prob_vec = np.zeros([len(category)], dtype=float)

    for key, value in category.items():
        if key not in category_index.keys():
            category_index[key] = len(category_index)

    for key, index in category_index.items():
        value = category[key]
        prob_vec[index] = value

    prob_vec = normalized_vec(prob_vec, type='norm')

    return category_index, prob_vec


def process_user_file(file_name, split='train'):
    # process user information for information count
    name = split + '/behaviors.tsv'
    user_info = {}
    invalid_record_count = 0
    with open(os.path.join(file_name, data_choose, name), encoding='utf8') as read:
        lines = read.readlines()

        for line in tqdm(lines):
            tokens = line.strip().split('\t')
            impressionId = tokens[0]
            usrId = tokens[1]
            time = conver(tokens[2])[0]
            click_history = tokens[3]
            if click_history == '':
                invalid_record_count += 1
                click_history = []
            else:
                click_history = tokens[3].strip().split(' ')

            positive_list, negative_list = process_rec_list(tokens[4], split)

            info = {'userId': usrId,
                    'time': time,
                    'click_history': click_history,
                    'positive_list': positive_list,
                    'negative_list': negative_list,
                    'impression': tokens[4]
                    }

            user_info[impressionId] = info

        print(f'invalid record count in {data_choose}-{name} user_info {invalid_record_count}')
        print(f'user information processed, total user count is {len(user_info)}')

    return user_info


def process_news_file(file_name, split='train'):
    # process news information for information count
    name = split + '/news.tsv'
    news_info = {}
    with open(os.path.join(file_name, data_choose, name), encoding='utf8') as read:
        lines = read.readlines()

        for line in tqdm(lines):
            tokens = line.strip().split('\t')
            newsId = tokens[0]
            category = tokens[1]
            sub_category = tokens[2]
            title = word_tokenize(tokens[3].strip())
            content = word_tokenize(tokens[4].strip())

            info = {'category': category, 'sub_category': sub_category, 'titile': title, 'content': content}
            news_info[newsId] = info

        print(f'news information processed, total news count is {len(news_info)}')
        return news_info


def news_ctr_calculation(news_info, user_info, category_prob, subcategory_prob, split='train'):
    """
    click_news = {newsid: count}
    :param news_info:
    :param users_file:
    :return:
    """
    
    click_news = {}
    recommend_news = {}
    history_count = {}
    click_count = 0
    recommend_count = 0
    pos_count = 0
    neg_count = 0

    def _add_operation(item, item_dict, p):
        if item in item_dict.keys():
            item_dict[item] += 1 * p
        else:
            item_dict[item] = 1 * p
    for impressId, content in tqdm(user_info.items()):
        click_history = content['click_history']

        for item in click_history:
            _add_operation(item, history_count, 1)

        if split != 'test':
            positive_list = content['positive_list']
            negative_list = content['negative_list']
            for item in positive_list:
                pos_count +=len(positive_list)
                _add_operation(item, click_news, 1)
                click_count += 1
                _add_operation(item, recommend_news, 1)
                recommend_count += 1
            for item in negative_list:
                neg_count += len(negative_list)
                _add_operation(item, recommend_news, 1)
                recommend_count += 1

    if split != 'test':
        average_ctr = click_count / (recommend_count * 1.0)
    
   
    total_count = sum([value for key, value in recommend_news.items()])
    total_history_count = sum([value for key, value in history_count.items()])
    total_click_count = sum([value for key, value in click_news.items()])

    for newId in tqdm(news_info.keys()):
        category = news_info[newId]['category']
        subcategory = news_info[newId]['sub_category']

        if split != 'test':
            if newId in recommend_news.keys():
                if newId in click_news.keys():
                    ctr = click_news[newId] / (recommend_news[newId] * 1.0)
                    prob = (recommend_news[newId] * 1.0) / (total_count * 1.0)
                else:
                    ctr = 0
                    prob = 0
            elif newId in history_count.keys():
                ctr = 0
                prob = (history_count[newId] * 1.0) / (total_history_count * 1.0)
            else:
                ctr = average_ctr
                prob = 0
        else:
            ctr = 0
            prob = 0

        if category not in category_prob.keys():
            ctprob = category_prob['other']
            news_info[newId].update({'category': 'other'})
        else:
            ctprob = category_prob[news_info[newId]['category']]
        if subcategory not in subcategory_prob.keys():
            sctprob = subcategory_prob['other']
            news_info[newId].update({'sub_category': 'other'})
        else:
            sctprob = subcategory_prob[news_info[newId]['sub_category']]

        if newId in history_count.keys():
            history_click = history_count[newId]
        else:
            history_click = 0
        news_info[newId].update({'ctr': float(ctr), 'history_click': history_click, 'prob': prob, 'ctprob': ctprob, 'sctprob': sctprob})

    return news_info


def caculate_user_click_probV2(category_index, click_history, category_prob=None):
    # click distribution of historical click of users
    prob_vec = np.zeros([len(category_index)], dtype=float)
    results = {}
    for item in click_history:
        if item not in category_prob.keys():
            item = 'other'

        if item in results.keys():
            results[item] += 1
        else:
            results[item] = 1

    origin_total = 0

    for key, value in results.items():
        if key not in category_prob.keys():
            key = 'other'
        idx = category_index[key]
        origin_total += value
        if conformity_type == 'weighted':
            prob_vec[idx] = value * category_prob[key]
        elif conformity_type == 'perturb':
            prob_vec[idx] = value + myTanh(1 - category_prob[key])
        else:
            prob_vec[idx] = value

    prob_vec = normalized_vec(prob_vec, type='norm', original_value=None)

    return prob_vec


def user_prob_calculationV2(user_info, news_info, category_index, category_probs, topic_prob=None):
    """
    :param user_info:
    :param news_info:
    :param category_index: index and category
    :param category_probs: distribution of category
    :param topic_prob: the probability of each topic
    :return:
    """
    if topic_prob is not None:
        topic_p = topic_prob[0]
        topic_sp = topic_prob[1]
    else:
        topic_p = topic_prob
        topic_sp = topic_prob

    category_prob = category_probs[0]
    subcategory_prob = category_probs[1]

    if not isinstance(category_index, list):
        raise ValueError('both category and subcategory should be provided')
    else:
        p_index = category_index[0]
        sp_index = category_index[1]

    for impressId, content in tqdm(user_info.items()):
        history_category = []
        history_subcategory = []
        click_history = content['click_history']
        for item in content['click_history']:
            history_category.append(news_info[item]['category'])
            history_subcategory.append(news_info[item]['sub_category'])

        p_prob = caculate_user_click_probV2(p_index, history_category, topic_p)
        sp_prob = caculate_user_click_probV2(sp_index, history_subcategory, topic_sp)

        p_kl = calculate_kl(p_prob, category_prob)
        sp_kl = calculate_kl(sp_prob, subcategory_prob)

        if math.isnan(p_kl) or math.isnan(sp_kl):
            p_kl = 10.
            sp_kl = 50.
        p_ctr = 0.
        a_ctr = 0.
        for item in content['positive_list']:
            p_ctr += news_info[item]['ctr']
            a_ctr += news_info[item]['ctr']
        for item in content['negative_list']:
            a_ctr += news_info[item]['ctr']

        avg_ctr = p_ctr / (a_ctr + 1e-8)

        user_info[impressId].update({'category_kl': p_kl, 'subcategory_kl': sp_kl, 'candidate_ctr': avg_ctr})

    return user_info


def generated_data(split='train'):
    print(f'> genrating index file')
    if split == 'train':
        uid2idx, nid2idx = generate_index(is_mix=True, is_read=False)
    else:
        uid2idx, nid2idx = generate_index(is_mix=True, is_read=True)

    # read all the data
    print(f'> starting read raw data')
    user_info = process_user_file(base_path, split)
    news_info = process_news_file(base_path, split)
    print(f'> raw data read finished, obtain \nuser_info: {len(user_info)}, \nnew_info: {len(news_info)}')

    # obtain category information
    print(f'> starting calculate the category probability')
    category, sub_category = collect_topic(news_info, base_path)

    # obatin the global distribution of topic information
    category_index, category_prob = generate_global_prob(category)
    subcategory_index, subcategory_prob = generate_global_prob(sub_category)


    # calculate ctr, topic click prob, etc
    print(f'> starting calculate various probability of users and news')
    updated_news_info = news_ctr_calculation(news_info=news_info, user_info=user_info, split=split, category_prob=category, subcategory_prob=sub_category)

    updated_user_info = user_prob_calculationV2(
        user_info=user_info,
        news_info=updated_news_info,
        category_index=[category_index, subcategory_index],
        category_probs=[category_prob, subcategory_prob],
        topic_prob=[category, sub_category]
    )
    print(f'> data process finished, obtain \nuser_info: {len(updated_user_info)}, \nnew_info: {len(updated_news_info)}')

    print(f'> starting store to file line by line')
    write_results(
        file_path=base_path,
        contents=updated_news_info,
        split=split,
        is_user=False
    )
    write_results(
        file_path=base_path,
        contents=updated_user_info,
        split=split,
        is_user=True
    )


if __name__ == '__main__':
    for split in ['train', 'valid']:
        print(f'> start processing {split} data')
        generated_data(split=split)
