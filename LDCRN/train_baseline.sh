#python train.py --bert_dir '/data/zhangdacao/opensource-model/bert-base-uncased' --savedmodel_path '/data/zhangdacao/save/baseline/bert-base-uncased_pair' --pair

#python train.py --bert_dir '/data/zhangdacao/opensource-model/roberta-base' --savedmodel_path '/data/zhangdacao/save/baseline/roberta-base_pair' --pair

#python train.py --bert_dir '/data/zhangdacao/opensource-model/bert-base-uncased' --savedmodel_path '/data/zhangdacao/save/baseline/bert-base-uncased' --best_score 0.6

#python train.py --bert_dir '/data/zhangdacao/opensource-model/roberta-base' --savedmodel_path '/data/zhangdacao/save/baseline/roberta-base' --best_score 0.6

#python train.py --task_name 'SNLI' --bert_dir '/data/zhangdacao/opensource-model/bert-base-uncased' --savedmodel_path '/data/zhangdacao/save/onlyhy_snli_baseline/bert-base-uncased_pair' --pair --confactual 2 --best_score 0.5

#python train.py --task_name 'SNLI' --bert_dir '/data/zhangdacao/opensource-model/bert-base-uncased' --savedmodel_path '/data/zhangdacao/save/maskpre_snli_baseline/bert-base-uncased_pair' --pair --confactual 1 --best_score 0.5


python train.py --task_name 'SNLI' --bert_dir '/data/zhangdacao/opensource-model/bert-base-uncased' --savedmodel_path '/data/zhangdacao/save/confactual_snli_baseline/bert-base-uncased_pair' --pair --confactual_train


python train.py --bert_dir '/data/zhangdacao/opensource-model/unsup-simcse-bert-base-uncased' --savedmodel_path '/data/zhangdacao/save/snli_baseline/unsup_simcse_pair' --pair

python train.py --bert_dir '/data/zhangdacao/opensource-model/unsup-simcse-roberta-base' --savedmodel_path '/data/zhangdacao/save/snli_baseline/unsup_simcse_roberta_pair' --pair















#python train.py --task_name 'SNLI' --bert_dir '/data/zhangdacao/opensource-model/roberta-base' --savedmodel_path '/data/zhangdacao/save/onlyhy_snli_baseline/roberta-base_pair' --pair --confactual 2 --best_score 0.5

#python train.py --task_name 'SNLI' --bert_dir '/data/zhangdacao/opensource-model/roberta-base' --savedmodel_path '/data/zhangdacao/save/maskpre_snli_baseline/roberta-base_pair' --pair --confactual 1 --best_score 0.5

python train.py --task_name 'MultiNLI' --bert_dir '/data/zhangdacao/opensource-model/bert-base-uncased' --savedmodel_path '/data/zhangdacao/save/onlyhy_multinli_baseline/bert-base-uncased_pair' --pair --confactual 2 --best_score 0.5















#python train.py --bert_dir '/root/autodl-tmp/opensource-model/princeton-nlp/unsup-simcse-bert-base-uncased' --savedmodel_path '/root/autodl-tmp/save/baseline/unsup-simcse_pair' --pair

#python train.py --bert_dir '/root/autodl-tmp/opensource-model/princeton-nlp/sup-simcse-bert-base-uncased' --savedmodel_path '/root/autodl-tmp/save/baseline/sup-simcse_pair' --pair

# python train.py --bert_dir '/root/autodl-tmp/opensource-model/bert-base-uncased' --savedmodel_path '/root/autodl-tmp/save/baseline/bert-base-uncased'

#python train.py --bert_dir '/root/autodl-tmp/opensource-model/roberta-base' --savedmodel_path '/root/autodl-tmp/save/baseline/roberta-base'

#python train.py --bert_dir '/root/autodl-tmp/opensource-model/princeton-nlp/unsup-simcse-bert-base-uncased' --savedmodel_path '/root/autodl-tmp/save/baseline/unsup-simcse'

#python train.py --bert_dir '/root/autodl-tmp/opensource-model/princeton-nlp/sup-simcse-bert-base-uncased' --savedmodel_path '/root/autodl-tmp/save/baseline/sup-simcse'

#python train.py --task_name 'MultiNLI' --bert_dir '/data/zhangdacao/opensource-model/bert-base-uncased' --savedmodel_path '/data/zhangdacao/save/multinli_baseline/bert-base-uncased_pair' --pair

#python train.py --task_name 'MultiNLI' --bert_dir '/data/zhangdacao/opensource-model/bert-base-uncased' --savedmodel_path '/data/zhangdacao/save/onlyhy_multinli_baseline/bert-base-uncased_pair' --pair --confactual 2 --best_score 0.5

#python gen_hard_MultiNLI.py --task_name 'MultiNLI' --bert_dir '/data/zhangdacao/opensource-model/bert-base-uncased' --ckpt_file '/data/zhangdacao/save/onlyhy_multinli_baseline/bert-base-uncased_pair/model.bin' --pair 