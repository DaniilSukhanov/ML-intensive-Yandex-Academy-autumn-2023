import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms

class ImageToImageDataset(Dataset):
    def __init__(self, input_dir, target_dir=None, transform=None, mode='train'):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.mode = mode

        self.filenames = [f for f in os.listdir(input_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.filenames[idx])
        input_image = Image.open(input_path).convert('L')  

        if self.transform:
            input_image = self.transform(input_image)

        if self.mode == 'train':
            target_path = os.path.join(self.target_dir, self.filenames[idx])
            target_image = Image.open(target_path).convert('L')

            return input_image, target_image
        else:
            return input_image

class ImageToNumDataset(Dataset):
    def __init__(self, img_dir, transform=None, answers_file=None):
        self.img_dir = img_dir
        self.transform = transform
        self.answers_file = answers_file

        if self.answers_file is not None:
            self.img_labels = pd.read_csv(answers_file)
        else:
            self.img_labels = None
        
        self.image_filenames = [file for file in os.listdir(img_dir) if file.endswith('.png')]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        if self.img_labels is not None:
            label = self.img_labels.iloc[idx, 1]
            return image, label
        else:
            return image
