# 论文标题：Label-aware Debiased Causal Reasoning for Natural Language Inference
## 1 预训练模型与数据处理
### 1.1 数据集下载
使用download_model.py下载预训练模型BERT和RoBERTa,使用data_process.py下载并处理SNLI数据集和MultiNLI数据集,从网上下载SNLI-hard数据集
```
SNLI-hard:https://nlp.stanford.edu/projects/snli/snli_1.0_test_hard.jsonl
```
### 1.2 生成MultiNLI-hard数据集
首先使用BERT模型仅使用MultiNLI训练集中的假设句进行微调:
```
python train.py --task_name 'MultiNLI'
                --only_hy
                --bert_dir 'opensource-model/bert-base-uncased' 
                --saved_model_path 'save_path'
```
得到微调的模型之后使用gen_hard_MultiNLI.py生成MultiNLI数据集的挑战集，matched_hard.npy和mismatched_hard.npy
### 1.3 使用SNLI_word_PMI.py和MultiNLI_word_PMI.py生成每个类别的top-K词
### 1.4 使用calc_mean_emb.py为SNLI和MultiNLI数据集生成训练集前提句的平均表征，并生成每个类别的top-K词的平均表征
## 2 模型训练
### 2.1 SNLI数据集
```
python train.py --task_name 'SNLI'  
                --bert_dir 'opensource-model/bert-base-uncased' 
                --saved_model_path 'save_path'
```
### 2.2 MultiNLI数据集
```
python train.py --task_name 'MultiNLI' 
                --bert_dir 'opensource-model/bert-base-uncased' 
                --saved_model_path 'save_path'
```
### 2.3 LDCRN-SNLI
```
python train.py --task_name 'SNLI' 
                --confactual_train 
                --bert_dir 'opensource-model/bert-base-uncased' 
                --saved_model_path 'save_path'
                --fusion 'CON'
                --use_mean_pre
```
### 2.4 LDCRN-MultiNLI
```
python train.py --task_name 'MultiNLI' 
                --bert_dir 'opensource-model/bert-base-uncased' 
                --saved_model_path 'save_path' 
                --confactual_train 
                --fusion 'CON'
                --use_mean_pre
```
## 3 测试
### 3.1 有偏预测
```
python test.py --bert_dir 'opensource-model/bert-base-uncased' 
                --ckpt_flie 'save_path' 
```
### 3.2 去偏预测（LDCRN）
```
python debias_predict.py --bert_dir 'opensource-model/bert-base-uncased' 
                         --ckpt_flie 'save_path' 
                         --confactual_train
                         --fusion 'CON'
                         --use_mean_pre
```
