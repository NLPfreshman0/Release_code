import argparse


def parse_args():
    
    parser = argparse.ArgumentParser(description="simcse")
    parser.add_argument("--seed", type=int, default=2023, help="random seed.")
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout ratio')
    parser.add_argument("--task_name", type=str, default='SNLI', help="task name")
    parser.add_argument("--pool_type", type=str, default='cls', help="pool type")
    parser.add_argument("--train_mode", type=str, default='GTE', help="train mode")
    parser.add_argument("--pair", action='store_true', help="whether to use pair")
    parser.add_argument("--confactual", type=int, default=0, help="confactual type")
    parser.add_argument("--confactual_train", action='store_true', help="whether to train by confactual method")
    parser.add_argument("--only_pre", action='store_true', help="only get premise")
    parser.add_argument("--only_hy", action='store_true', help="only use hypothesis")
    parser.add_argument("--only_img", action='store_true', help="only use image")
    parser.add_argument("--snli_ve_only_hy", action='store_true', help="only get snli-ve hypothesis")
    parser.add_argument("--valid_steps", type=int, default=1000, help="valid steps")
    parser.add_argument("--use_mean_pre", action='store_true', help="use mean premise embedding")
    parser.add_argument("--fusion", type=str, default=None, help="fusion type")
    parser.add_argument("--debias_rate", type=float, default=0.0, help="fusion type")
    parser.add_argument("--use_CL", action='store_true', help="whether to use itc loss")
    parser.add_argument("--get_confactual_emb", action='store_true', help="get confactual embeddings")
    parser.add_argument("--dynamic_debias", action='store_true', help="whether to use dynamic debias")
    parser.add_argument("--lack_visual", action='store_true', help="whether to use dynamic debias")
    parser.add_argument("--lack_text", action='store_true', help="whether to use dynamic debias")
    # ========================= Data Configs ==========================
    parser.add_argument('--val_ratio', default=0.05, type=float, help='split 10 percentages of training data as validation')
    parser.add_argument('--batch_size', default=64, type=int, help="use for training duration per worker")
    parser.add_argument('--val_batch_size', default=256, type=int, help="use for validation duration per worker")
    parser.add_argument('--test_batch_size', default=256, type=int, help="use for testing duration per worker")
    parser.add_argument('--prefetch', default=16, type=int, help="use for training duration per worker")
    parser.add_argument('--num_workers', default=14, type=int, help="num_workers for dataloaders")

    # ======================== SavedModel Configs =========================
    parser.add_argument('--savedmodel_path', type=str, default='/data/zhangdacao/save/snli_baseline/bert-base-uncased')
    parser.add_argument('--ckpt_file', type=str, default='/data/zhangdacao/save/baseline/bert-base-uncased/model.bin')
    parser.add_argument('--best_score', default=0.8, type=float, help='save checkpoint if mean_f1 > best_score')

    # ========================= Learning Configs ==========================
    parser.add_argument('--max_epochs', type=int, default=4, help='How many epochs')
    parser.add_argument('--max_steps', default=40000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--print_steps', type=int, default=20, help="Number of steps to log training metrics.")
    parser.add_argument('--warmup_steps', default=1000, type=int, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--minimum_lr', default=0., type=float, help='minimum learning rate')
    parser.add_argument('--learning_rate', default=3e-5, type=float, help='initial learning rate')
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")

    # ========================== Title BERT =============================
    parser.add_argument('--bert_dir', type=str, default='/data/zhangdacao/opensource-model/bert-base-uncased')
    parser.add_argument('--bert_seq_length', type=int, default=128)
    return parser.parse_args()
