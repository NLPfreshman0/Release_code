# 论文标题：针对情境感知的自然语言推理任务的因果去偏方法研究
## 1 预训练模型与数据处理
### 1.1 数据集下载
使用download_model.py下载预训练模型BERT和RoBERTa,从官网上下载SNLI-VE数据集和Flickr30k数据集
```
SNLI-VE:https://github.com/necla-ml/SNLI-VE
Flickr30k:http://shannon.cs.illinois.edu/DenotationGraph/
```
### 1.2 生成GTE数据集
下载完数据集之后，使用data_process.py对数据进行处理，基于SNLI-VE数据集生成GTE数据集train.npy,validation.npy,test.npy
运行extract_images.py，使用CLIP模型的Visual部分对Flickr30k数据集的图片进行特征抽取,生成clip_image.npy
### 1.3 生成GTE-hard数据集
首先使用BERT模型仅使用训练集中的假设据进行微调:
```
python train.py --task_name 'GTE'
                --train_mode 'TE'
                --only_hy
                --bert_dir 'opensource-model/bert-base-uncased' 
                --saved_model_path 'save_path'
```
得到微调的模型之后使用gen_hard_GTE.py生成GTE数据集的挑战集，hard.npy
## 2 模型训练
### 2.1 VE模型
```
python train.py --task_name 'GTE' 
                --train_mode 'VE' 
                --bert_dir 'opensource-model/bert-base-uncased' 
                --saved_model_path 'save_path'
```
### 2.2 TE 模型
```
python train.py --task_name 'GTE' 
                --train_mode 'TE' 
                --bert_dir 'opensource-model/bert-base-uncased' 
                --saved_model_path 'save_path'
```
### 2.3 GTE 模型
```
python train.py --task_name 'GTE' 
                --train_mode 'GTE' 
                --bert_dir 'opensource-model/bert-base-uncased' 
                --saved_model_path 'save_path' --fusion 'CON'
```
### 2.4 CBDRF 模型
```
python train.py --task_name 'GTE' 
                --bert_dir 'opensource-model/bert-base-uncased' 
                --saved_model_path 'save_path' 
                --fusion 'CON' 
                --confactual_train 
                --use_CL
                --batch_size 32
                --max_steps 80000
                --valid_steps 4000
```
## 3 测试
### 3.1 有偏预测
```
python test.py --bert_dir 'opensource-model/bert-base-uncased' 
                --ckpt_flie 'save_path' 
                --train_mode '' 
                --fusion 'CON'(GTE需要)
```
### 3.2 去偏预测（CBDRF）
```
python debias_predict.py --bert_dir 'opensource-model/bert-base-uncased' 
                         --ckpt_flie 'save_path' 
                         --confactual_train
                         --fusion 'CON'
```
