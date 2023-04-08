# python test.py --bert_dir '/data/zhangdacao/opensource-model/bert-base-uncased' --ckpt_file '/data/zhangdacao/save/baseline/bert-base-uncased_pair/model.bin' --pair
# python test.py --bert_dir '/root/autodl-tmp/opensource-model/roberta-base' --ckpt_file '/root/autodl-tmp/save/baseline/roberta-base_pair/model.bin' --pair
# python test.py --bert_dir '/root/autodl-tmp/opensource-model/princeton-nlp/unsup-simcse-bert-base-uncased' --ckpt_file '/root/autodl-tmp/save/baseline/unsup-simcse_pair/model.bin' --pair
# python test.py --bert_dir '/root/autodl-tmp/opensource-model/princeton-nlp/sup-simcse-bert-base-uncased' --ckpt_file '/root/autodl-tmp/save/baseline/sup-simcse_pair/model.bin' --pair

python test.py --bert_dir '/root/autodl-tmp/opensource-model/bert-base-uncased' --ckpt_file '/root/autodl-tmp/save/baseline/bert-base-uncased/model.bin'
python test.py --bert_dir '/root/autodl-tmp/opensource-model/roberta-base' --ckpt_file '/root/autodl-tmp/save/baseline/roberta-base/model.bin'
python test.py --bert_dir '/root/autodl-tmp/opensource-model/princeton-nlp/unsup-simcse-bert-base-uncased' --ckpt_file '/root/autodl-tmp/save/baseline/unsup-simcse/model.bin'
python test.py --bert_dir '/root/autodl-tmp/opensource-model/princeton-nlp/sup-simcse-bert-base-uncased' --ckpt_file '/root/autodl-tmp/save/baseline/sup-simcse/model.bin'

#python test.py --task_name 'SNLI-VE' --bert_dir '/root/autodl-tmp/opensource-model/bert-base-uncased' --ckpt_file '/root/autodl-tmp/save/baseline/bert-base-uncased_ve/model.bin'

python test.py --task_name 'MultiNLI' --bert_dir '/data/zhangdacao/opensource-model/bert-base-uncased' --ckpt_file '/data/zhangdacao/save/multinli_baseline/bert-base-uncased_pair/model.bin' --pair

python test.py --task_name 'SNLI' --bert_dir '/data/zhangdacao/opensource-model/bert-base-uncased' --ckpt_file '/data/zhangdacao/save/onlyhy_snli_baseline/bert-base-uncased_pair/model.bin' --pair --confactual 2
