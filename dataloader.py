import os
import pandas as pd
import random
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import random_split
from torchvision import transforms as T
from torchvision.transforms.functional import adjust_contrast, adjust_brightness

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, InterpolationMode
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
        self.transform = transform

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

        if self.transform:
            image_fullbody = self.transform(image_fullbody)
            image_face = self.transform(image_face)

        return image_fullbody, image_face, y

class AugmentedBMIDataset(Dataset):
    def __init__(self, original_dataset, transforms=None):
        self.original_dataset = original_dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.original_dataset)  # No multiplication

    def __getitem__(self, idx):
        # Get the original images and target
        image_fullbody, image_face, y = self.original_dataset[idx]

        # Apply transformations with a probability
        if self.transforms and random.random() > 0.5:  # 50% chance to apply transformations
            image_fullbody = self.transforms(image_fullbody)
            image_face = self.transforms(image_face)

        return image_fullbody, image_face, y
    
    def get_image_name(self, idx):
        return self.image_paths[idx]
    '''
class AugmentedBMIDataset(Dataset):
    def __init__(self, original_dataset, transforms=None):
        self.original_dataset = original_dataset
        self.transforms = transforms

    def __len__(self):
        return 5 * len(self.original_dataset)

    def __getitem__(self, idx):
        image_fullbody, image_face, y = self.original_dataset[idx // 5]

        if self.transforms and (idx % 5 != 0):
            image_fullbody = self.transforms(image_fullbody)
            image_face = self.transforms(image_face)
        return image_fullbody, image_face, y
'''
class RandomDistortion(torch.nn.Module):
    def __init__(self, probability=0.25, grid_width=2, grid_height=2, magnitude=8):
        super().__init__()
        self.probability = probability
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.magnitude = magnitude

    def forward(self, img):
        if torch.rand(1).item() < self.probability:
            return T.functional.affine(img, 0, [0, 0], 1, [self.magnitude, self.magnitude], interpolation=T.InterpolationMode.NEAREST, fill=[0, 0, 0])
        else:
            return img

class RandomAdjustContrast(torch.nn.Module):
    def __init__(self, probability=.5, min_factor=0.8, max_factor=1.2):
        super().__init__()
        self.probability = probability
        self.min_factor = min_factor
        self.max_factor = max_factor

    def forward(self, img):
        if torch.rand(1).item() < self.probability:
            factor = torch.rand(1).item() * (self.max_factor - self.min_factor) + self.min_factor
            return adjust_contrast(img, factor)
        else:
            return img

augmentation_transforms = T.Compose([
    T.RandomRotation(5),
    T.RandomHorizontalFlip(p=0.5),
    RandomDistortion(probability=0.25, grid_width=2, grid_height=2, magnitude=8),
    T.RandomApply([T.ColorJitter(brightness=(0.5, 1.5), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(0.0, 0.1))], p=1),
    RandomAdjustContrast(probability=0.5, min_factor=0.8, max_factor=1.2),
    T.Lambda(lambda img: adjust_brightness(img, torch.rand(1).item() + 0.5))
])

def get_dataloaders(batch_size=16, augmented=True, vit_transformed=True, show_sample=False):
    bmi_dataset = BMIDataset('/home/nguyenbt/nobackup/face-to-bmi-vit/archive/height_prerun.csv', '/home/nguyenbt/nobackup/data/2019_Mhse_Height_Data/combined_fullbody_prerun/', '/home/nguyenbt/nobackup/data/2019_Mhse_Height_Data/combined_face_prerun', 'height', ToTensor())
    if show_sample:
        train_dataset, val_dataset, test_dataset = train_val_test_split(bmi_dataset, augmented, vit_transformed=False)
        #show_sample_image(train_dataset)
    train_dataset, val_dataset, test_dataset = train_val_test_split(bmi_dataset, augmented, vit_transformed=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers= 16, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers= 16, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers= 16, pin_memory=True, shuffle=False)
    return train_loader, test_loader, val_loader

def train_val_test_split(dataset, augmented=True, vit_transformed=True):
    val_size = int(0.1 * len(dataset))
    test_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    if augmented:
        train_dataset = AugmentedBMIDataset(train_dataset, augmentation_transforms)

    #if vit_transformed:
    #    train_dataset = VitTransformedDataset(train_dataset)
    #    val_dataset = VitTransformedDataset(val_dataset)
    #    test_dataset = VitTransformedDataset(test_dataset)

    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    get_dataloaders(augmented=False, show_sample=True)