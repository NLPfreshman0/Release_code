from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModel
import os
from tqdm import tqdm
import torch
import numpy as np

class ImageDataset(Dataset):
    def __init__(self):
        self.path = "/data/zhangdacao/dataset/Flickr30k/flickr30k-images"
        self.files= os.listdir(self.path) 
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_id = self.files[idx][:-4]
        file = self.files[idx]
        image = Image.open(self.path+"/"+file)
        inputs = self.processor(images=image, return_tensors="pt")['pixel_values']
        return {'idx':img_id, 'inputs':inputs}

dataset = ImageDataset()
dataloader = DataLoader(dataset, batch_size=256, num_workers=8, shuffle=False, pin_memory=True, drop_last=False)

model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
model = torch.nn.parallel.DataParallel(model.cuda())

model.eval()
clip_image = dict()
loop = tqdm(dataloader, total = len(dataloader))
with torch.no_grad():
    for data in loop:
        idx = data['idx']
        inputs = data['inputs'].squeeze(1)
        outputs = model(inputs)
        last_hidden_state = outputs.last_hidden_state
        emb = last_hidden_state.cpu().numpy()
        for i, e in zip(idx, emb):
            clip_image[i] = e

np.save('/data/zhangdacao/dataset/GTE/clip_image.npy', clip_image)