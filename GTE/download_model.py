import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pathlib import Path
import pathlib
import os

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    AutoModelForPreTraining,
    TFAutoModel
)

NEW_DIR = "/data/zhangdacao/opensource-model/"
print('Transformers version',transformers.__version__) 

def transformers_model_dowloader(pretrained_model_name_list = ['bert-base-uncased'], is_tf = True, model_class=AutoModelForPreTraining):
    if is_tf:
        model_class = TFAutoModel

    for i, pretrained_model_name in enumerate(pretrained_model_name_list):
        print(i+1,'/',len(pretrained_model_name_list))
        print("Download model and tokenizer", pretrained_model_name)
        transformer_model = model_class.from_pretrained(pretrained_model_name)
        transformer_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        print("Save model and tokenizer", pretrained_model_name, 'in directory', NEW_DIR + pretrained_model_name.split('/')[-1])
        transformer_model.save_pretrained(NEW_DIR + pretrained_model_name.split('/')[-1])
        transformer_tokenizer.save_pretrained(NEW_DIR + pretrained_model_name.split('/')[-1])

pathlib.Path(NEW_DIR).mkdir(parents=True, exist_ok=True)

pretrained_model_name_list = ['princeton-nlp/unsup-simcse-bert-base-uncased', 'princeton-nlp/unsup-simcse-roberta-base']
transformers_model_dowloader(pretrained_model_name_list, is_tf = False)
print("Download finish")
