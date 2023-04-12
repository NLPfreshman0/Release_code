import argparse
import pprint
import os


def train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='nrms_ci')
    parser.add_argument('--name', default='nrms_ci')
    basic_path = '/root/autodl-tmp/newsRec/dataset/small'
    cache_dir = '/root/autodl-tmp/newsRec/dataset/pretrained'
    embedding_path = '/root/autodl-tmp/newsRec/dataset/glove_embedding'


    parser.add_argument('--basic_path', type=str, default=basic_path)
    parser.add_argument('--embedding_path', type=str, default=embedding_path)
    parser.add_argument('--log_path', type=str, default='nrms_ci/train/')
    parser.add_argument('--cache_dir', type=str, default=cache_dir)

    parser.add_argument('--fusing', type=str, default='add')
    parser.add_argument('--use_bpr', action='store_true', default=True)

    parser.add_argument('--title_size', type=int, default=30)
    parser.add_argument('--his_size', type=int, default=50)
    parser.add_argument('--npratio', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--display_step', type=int, default=500)

    parser.add_argument('--user_weight', type=float, default=1.0)

    parser.add_argument('--metrics', type=list, default=['auc', 'mrr', 'ndcg@5;10;15;20'])

    parser.add_argument('--kl_usage', type=str, default='subcategory')
    parser.add_argument('--prob_usage', type=int, default=4)

    parser.add_argument('--pre_trained_model', type=str, default='albert-base-v2')
    parser.add_argument('--gpu', type=str, default='0')

    parser.add_argument('--use_bert', action='store_true', default=False)
    parser.add_argument('--train_bert', action='store_true', default=False)
    parser.add_argument('--embedding_train', action='store_true', default=True)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_decay', type=float, default=0.)
    parser.add_argument('--grad_max_norm', type=float, default=0.)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=0.005)

    parser.add_argument('--macr_alpha', type=float, default=1e-4) # range [1e-5, 1e-4, 1e-3, 1e-2]
    parser.add_argument('--macr_beta', type=float, default=1e-5) # range [1e-5, 1e-4, 1e-3, 1e-2]
    parser.add_argument('--macr_c', type=float, default=30)  # range [20, 40]

    parser.add_argument('--pda_power', type=float, default=0.18) # range [0.02, 0.25]
    parser.add_argument('--dice_alpha', type=float, default=0.02)  # range [0.02, 0.25]
    parser.add_argument('--dice_beta', type=float, default=0.01)  # range [0.02, 0.25]

    args = parser.parse_args()

    return args

def test_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='nrms_ci')
    parser.add_argument('--name', default='nrms_ci')
    basic_path = '/root/autodl-tmp/newsRec/dataset/small'
    cache_dir = '/root/autodl-tmp/newsRec/dataset/pretrained'
    embedding_path = '/root/autodl-tmp/newsRec/dataset/glove_embedding'

    parser.add_argument('--basic_path', type=str, default=basic_path)
    parser.add_argument('--embedding_path', type=str, default=embedding_path)
    parser.add_argument('--log_path', type=str, default='nrms_ci/test/')
    parser.add_argument('--cache_dir', type=str, default=cache_dir)

    parser.add_argument('--fusing', type=str, default='add')
    parser.add_argument('--use_bpr', action='store_true', default=True)

    parser.add_argument('--title_size', type=int, default=30)
    parser.add_argument('--his_size', type=int, default=50)
    parser.add_argument('--npratio', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--display_step', type=int, default=500)

    parser.add_argument('--user_weight', type=float, default=1.0)

    parser.add_argument('--metrics', type=list, default=['auc', 'mrr', 'ndcg@5;10;15;20'])

    parser.add_argument('--kl_usage', type=str, default='subcategory')
    parser.add_argument('--prob_usage', type=int, default=4)

    parser.add_argument('--pre_trained_model', type=str, default='albert-base-v2')
    parser.add_argument('--gpu', type=str, default='0')

    parser.add_argument('--use_bert', action='store_true', default=False)
    parser.add_argument('--train_bert', action='store_true', default=False)
    parser.add_argument('--embedding_train', action='store_true', default=True)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=True)

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--grad_max_norm', type=float, default=0.)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=0.005)

    parser.add_argument('--macr_alpha', type=float, default=1e-4) # range [1e-5, 1e-4, 1e-3, 1e-2]
    parser.add_argument('--macr_beta', type=float, default=1e-5) # range [1e-5, 1e-4, 1e-3, 1e-2]
    parser.add_argument('--macr_c', type=float, default=30)  # range [20, 40]

    parser.add_argument('--pda_power', type=float, default=0.18) # range [0.02, 0.25]
    parser.add_argument('--dice_alpha', type=float, default=0.02)  # range [0.02, 0.25]
    parser.add_argument('--dice_beta', type=float, default=0.01)  # range [0.02, 0.25]

    args = parser.parse_args()

    return args
