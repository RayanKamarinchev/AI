import torch
from PIL import Image
from torchvision import models
from tqdm import tqdm
import os
import sys

class Dataset(torch.utils.data.Dataset):
    def __init__(self, split, preprocess):
        self.split = split
        self.preprocess = preprocess

        with open(f'./data/{split}.csv', 'r') as f:
            lines = f.readlines()
        self.filename, self.y, self.a = [], [], []
        for line in lines[1:]:
            _, filename, y = line.rstrip().split(',')
            self.filename.append(filename)
            self.y.append(int(y))

    def __getitem__(self, sample_n):
        im = Image.open(f'./data/{self.split}/{self.filename[sample_n]}').convert('RGB')
        return self.preprocess(im), self.y[sample_n]
    
    def __len__(self):
        return len(self.y)


weights = models.ResNet50_Weights.IMAGENET1K_V1
resnet   = models.resnet50(weights=weights).eval()
resnet.fc = torch.nn.Identity()
preprocess = weights.transforms()
resnet.eval()
resnet.cuda()

datasets = {split: Dataset(split, preprocess) for split in ['train', 'val']}

for split in ['train', 'val']:
    train_dl = torch.utils.data.DataLoader(
        datasets[split],
        batch_size = 256,
        shuffle = False,
        num_workers = 10,
        pin_memory = True,
    )

    for images, labels in tqdm(train_dl):
        images = images.cuda()
        
        with torch.no_grad():
            features = resnet(images)
            print(features.shape)
            sys.exit(0)