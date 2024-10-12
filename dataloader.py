import os
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import pickle

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class BMIDataset(Dataset):
    def __init__(self, p, image_folder, image_id, split transform=None):
        self.p = pickle.load(open('/home/nguyenbt/nobackup/data/2019_Mhse_Height_Data/' + split + '.pickle', 'rb'), encoding='latin1')
        self.image_folder = image_folder

        self.image_id = str(p["image_id"].decode('latin1'))
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.csv.iloc[idx, 4])
        image = Image.open(image_path)

        # check the channel number
        if image.mode != 'RGB':
            image = image.convert('RGB')

        y = self.csv.loc[idx, self.y_col_name]

        if self.transform:
            image = self.transform(image)

        return image, y