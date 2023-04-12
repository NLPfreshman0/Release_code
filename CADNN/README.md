# 论文标题：融合从众性建模的去偏新闻推荐研究

## 1. 数据处理
### 1.1 数据集下载
使用data_download.py下载MIND数据集和glove预训练文件
MIND数据集:https://msnews.github.io/
glove预训练文件:https://nlp.stanford.edu/projects/glove/
## 1.2 数据集处理
### 1.2.1 使用data_preprocess.py对MIND数据集进行处理，生成处理后的新闻和用户文件
### 1.2.2 使用another_preprocess.py生成词表征文件word2vec.npy
### 1.3 去偏测试集生成
```
    python challenging_test_generate.py
```
# 2.模型训练
## 2.1 baselines
### 2.1.1 MACR模型
```
python train.py --net'macr'
                --name'macr'
                --saved_model_path'checkpoint/macr.pth'
```
### 2.1.2 PDA模型
```
python train.py --net'pda'
                --name'pda'
                --saved_model_path'checkpoint/pda.pth'
```
### 2.1.3 DICE模型
```
python train.py --net'dice'
                --name'dice'
                --saved_model_path'checkpoint/dice.pth'
```
### 2.1.4 TIDE模型
```
python train.py --net'tide'
                --name'tide'
                --saved_model_path'checkpoint/tide.pth'
```
## 2.2 backbone
### 2.2.1 NRMS模型
```
python train.py --net'nrms'
                --name'nrms'
                --saved_model_path'checkpoint/nrms.pth'
```
### 2.2.1 NAML模型
```
python train.py --net'naml'
                --name'naml'
                --saved_model_path'checkpoint/naml.pth'
```
## 2.3 CADNN模型
### 2.3.1 以NRMS为backbone：NRMS_CI
```
python train.py --net'nrms_ci'
                --name'nrms_ci'
                --kl_usage'subcategory'
                --prob_usage 4
                --saved_model_path'checkpoint/nrms_ci.pth'
```
### 2.3.2 以NAML为backbone：NAML_CI
```
python train.py --net'naml_ci'
                --name'naml_ci'
                --kl_usage'all'
                --prob_usage 4
                --saved_model_path'checkpoint/naml_ci.pth'
```
# 3. 模型测试
```
python test.py  --net '' 
                --name ''
                --ckpt_flie 'checkpoint/save_path' 
```
