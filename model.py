# Imports
import os, sys
from pathlib import Path
import json
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, DataLoader
from torch import ByteTensor, Tensor, as_tensor
import cv2 as cv

# Constants
PATH_ROOT = Path(r"F:\ml-parsing-project\table-parse-split")
COMPLEXITY = 1

# Paths
pathData = PATH_ROOT / 'data' / f'fake_{COMPLEXITY}'

# Data
class TableDataset(Dataset):
    def __init__(self, dir_data,
                    image_format='.jpg',
                    transform=None, target_transform=None):
        # class A:
        #  pass
        # self = A()

        # Store in self 
        dir_data = Path(dir_data)
        self.dir_images = dir_data / 'images'
        self.dir_features = dir_data / 'features'
        self.dir_labels = dir_data / 'labels'
        self.transform = transform
        self.target_transform = target_transform
        self.image_format = image_format
        self.image_size = (600, 400)

        # Get filepaths
        self.image_fileEntries = list(os.scandir(self.dir_images))
        self.feature_fileEntries = list(os.scandir(self.dir_features))
        self.label_fileEntries = list(os.scandir(self.dir_labels))

        items_images   = set([os.path.splitext(file.name)[0] for file in self.image_fileEntries])
        items_features = set([os.path.splitext(file.name)[0] for file in self.feature_fileEntries])
        items_labels   = set([os.path.splitext(file.name)[0] for file in self.label_fileEntries])

        # Verify consistency between image/feature/label
        self.items = sorted(list(items_images.intersection(items_features).intersection(items_labels)))
        assert len(self.items) == len(items_images) == len(items_features) == len(items_labels), 'Set of images, features and labels do not match.'

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):     # idx = 0
        # Generate paths for a specific item
        pathImage = str(self.dir_images / f'{self.items[idx]}{self.image_format}')
        pathLabel = str(self.dir_labels / f'{self.items[idx]}.json')
        pathFeatures = str(self.dir_features / f'{self.items[idx]}.json')

        # Load sample
        sample = {}

        # Load sample | Image (not sure if this is working properly)
        sample['image'] = read_image(pathImage, mode=ImageReadMode.GRAY)
        
        # Load sample | Label
        with open(pathLabel, 'r') as f:
            labelData = json.load(f)
        sample['label'] = {}
        sample['label']['row'] = Tensor(labelData['row'])
    
        # Load sample | Features
        with open(pathFeatures, 'r') as f:
            featuresData = json.load(f)
        sample['features'] = {}
        sample['features']['row_avg'] = Tensor(featuresData['row_avg'])
        sample['features']['row_whites'] = Tensor(featuresData['row_whites'])

        # Optional transform
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        if self.target_transform:
            for labelType in sample['label']:
                sample['label'][labelType] = self.target_transform(sample['label'][labelType])

        # Return sample
        return sample    


dataset_train = TableDataset(dir_data=pathData / 'train')
dataloader_train = DataLoader(dataset=dataset_train, batch_size=6, shuffle=True)

dataset_val = TableDataset(dir_data=pathData / 'val')
dataloader_val = DataLoader(dataset=dataset_val, batch_size=6, shuffle=True)

# sample = next(iter(dataloader_train))
# img = sample['image'][0].squeeze().numpy()
# cv.imshow('img', img); cv.waitKey(0)
# label = sample['label']['row'][0]
