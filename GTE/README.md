# 论文标题：针对情境感知的自然语言推理任务的因果去偏方法研究
## 1.预训练模型与数据处理
使用download_model.py下载预训练模型BERT和RoBERTa,从官网上下载SNLI-VE数据集和Flickr30k数据集
```
SNLI-VE:https://github.com/necla-ml/SNLI-VE
Flickr30k:http://shannon.cs.illinois.edu/DenotationGraph/
```
下载完数据集之后，使用data_process.py对数据进行处理，生成train.npy,validation.npy,test.npy。
