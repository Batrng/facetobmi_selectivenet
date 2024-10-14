import os
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import random_split

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
class BMIDataset(Dataset):
    def __init__(self, csv_path, image_folder_fullbody, image_folder_face, y_col_name, transform=None):
        self.csv = pd.read_csv(csv_path)
        self.image_folder_fullbody = image_folder_fullbody
        self.image_folder_face = image_folder_face
        # Drop the rows where the image does not exist
        #images = os.listdir(image_folder_face)
        #self.csv = self.csv[self.csv['image_id'].isin(images)]
        #self.csv.reset_index(drop=True, inplace=True)

        self.y_col_name = y_col_name
        #self.transform = transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        image_path_fullbody = os.path.join(self.image_folder_fullbody, self.csv.iloc[idx, 0])
        image_fullbody = Image.open(image_path_fullbody)
        image_path_face = os.path.join(self.image_folder_face, self.csv.iloc[idx, 0])
        image_face = Image.open(image_path_face)

        # check the channel number
        if image_fullbody.mode != 'RGB':
            image_fullbody = image_fullbody.convert('RGB')

        if image_face.mode != 'RGB':
            image_face = image_face.convert('RGB')
        y = self.csv.loc[idx, self.y_col_name]

        #if self.transform:
        #    image = self.transform(image)

        return image_fullbody, image_face, y

def get_dataloaders(batch_size=16, augmented=True, vit_transformed=True, show_sample=False):
    bmi_dataset = BMIDataset('/home/nguyenbt/nobackup/face-to-bmi-vit/height.csv', '/home/nguyenbt/nobackup/data/2019_Mhse_Height_Data/combined_fullbody/', '/home/nguyenbt/nobackup/data/2019_Mhse_Height_Data/combined_face/', 'height', ToTensor())
    if show_sample:
        train_dataset, val_dataset, test_dataset = train_val_test_split(bmi_dataset, augmented, vit_transformed=False)
        #show_sample_image(train_dataset)
    train_dataset, val_dataset, test_dataset = train_val_test_split(bmi_dataset, augmented, vit_transformed=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,  shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,  shuffle=False)
    return train_loader, test_loader, val_loader

def train_val_test_split(dataset, augmented=True, vit_transformed=True):
    val_size = int(0.1 * len(dataset))
    test_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    #if augmented:
    #    train_dataset = AugmentedBMIDataset(train_dataset, augmentation_transforms)

    #if vit_transformed:
    #    train_dataset = VitTransformedDataset(train_dataset)
    #    val_dataset = VitTransformedDataset(val_dataset)
    #    test_dataset = VitTransformedDataset(test_dataset)

    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    get_dataloaders(augmented=False, show_sample=True)