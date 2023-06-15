# Imports
from pathlib import Path
import os, json
from functools import cache
from typing import Union

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode


from model import Meta, Sample, Features, Targets, COMMON_VARIABLES, COMMON_GLOBAL_VARIABLES, ROW_VARIABLES, COL_VARIABLES

class TableDataset(Dataset):
    def __init__(self, dir_data,
                    device='cuda', image_format='.png'):
        # Store in self 
        dir_data = Path(dir_data)
        self.device = device
        self.dir_images = dir_data / 'images'
        self.dir_features = dir_data / 'features'
        self.dir_labels = dir_data / 'labels'
        self.dir_meta = dir_data / 'meta'
        self.image_format = image_format

        # Get filepaths
        self.image_fileEntries = list(os.listdir(self.dir_images))
        self.feature_fileEntries = list(os.listdir(self.dir_features))
        self.label_fileEntries = list(os.listdir(self.dir_labels))
        self.meta_fileEntries = list(os.listdir(self.dir_meta))

        items_images   = set([os.path.splitext(file)[0] for file in self.image_fileEntries])
        items_features = set([os.path.splitext(file)[0] for file in self.feature_fileEntries])
        items_labels   = set([os.path.splitext(file)[0] for file in self.label_fileEntries])
        items_meta     = set([os.path.splitext(file)[0] for file in self.meta_fileEntries])

        # Verify consistency between image/feature/label
        self.items = sorted(list(items_images.intersection(items_features).intersection(items_labels).intersection(items_meta)))
        assert len(self.items) == len(items_images) == len(items_features) == len(items_labels) == len(items_meta), 'Set of images, meta, features and labels do not match.'

    def __len__(self):
        return len(self.items)  
    
    @cache
    def __getitem__(self, idx):     # idx = 0
        # Generate paths for a specific item
        pathImage = str(self.dir_images / f'{self.items[idx]}{self.image_format}')
        pathLabel = str(self.dir_labels / f'{self.items[idx]}.json')
        pathMeta = str(self.dir_meta / f'{self.items[idx]}.json')
        pathFeatures = str(self.dir_features / f'{self.items[idx]}.json')

        # Load sample
        # Load sample | Meta
        with open(pathMeta, 'r') as f:
            metaData = json.load(f)
        meta = Meta(path_image=pathImage, **metaData)

        # Load sample | Label
        with open(pathLabel, 'r') as f:
            labelData = json.load(f)
        row_targets = torch.tensor(labelData['row'])
        col_targets = torch.tensor(labelData['col'])
        row_separator_count = torch.tensor(torch.where(torch.diff(row_targets) == 1)[0].numel(), dtype=torch.int32)
        col_separator_count = torch.tensor(torch.where(torch.diff(col_targets) == 1)[0].numel(), dtype=torch.int32)
        
        targets = Targets(row_line=row_targets.unsqueeze(-1).to(self.device), row_separator_count = row_separator_count.to(self.device),
                            col_line=col_targets.unsqueeze(-1).to(self.device), col_separator_count = col_separator_count.to(self.device))
    
        # Load sample | Features
        with open(pathFeatures, 'r') as f:
            featuresData = json.load(f)

        # Load sample | Features | Rows
        features_row = torch.cat(tensors=[torch.tensor(featuresData[common_variable.format('row')]).unsqueeze(-1) for common_variable in COMMON_VARIABLES], dim=1)
        features_row = torch.cat([features_row, *[torch.tensor(featuresData[row_variable]).unsqueeze(-1) for row_variable in ROW_VARIABLES]], dim=1)
        features_row_global = torch.cat([torch.tensor([featuresData[global_common_variable.format('row')]]).broadcast_to((features_row.shape[0], 1)) for global_common_variable in COMMON_GLOBAL_VARIABLES], dim=1)

        # Load sample | Features | Cols
        features_col = torch.cat(tensors=[torch.tensor(featuresData[common_variable.format('col')]).unsqueeze(-1) for common_variable in COMMON_VARIABLES], dim=1)
        features_col = torch.cat([features_col, *[torch.tensor(featuresData[col_variable]).unsqueeze(-1) for col_variable in COL_VARIABLES]], dim=1)
        features_col_global = torch.cat([torch.tensor([featuresData[global_common_variable.format('col')]]).broadcast_to((features_col.shape[0], 1)) for global_common_variable in COMMON_GLOBAL_VARIABLES], dim=1)
    
        # Load sample | Features | Image (0-1 float)
        image = read_image(pathImage, mode=ImageReadMode.GRAY).to(dtype=torch.float32) / 255

        # Collect in namedtuples
        features = Features(row=features_row, col=features_col, row_global=features_row_global, col_global=features_col_global, image=image)
        sample = Sample(features=features, targets=targets, meta=meta)

        # Return sample
        return sample
    
def get_dataloader(dir_data:Union[Path, str], batch_size:int=1, shuffle:bool=True, device='cuda', image_format:str='.png') -> DataLoader:
    # Parameters
    dir_data = Path(dir_data)
    image_format = f'.{image_format}' if not image_format.startswith('.') else image_format

    # Checks
    assert batch_size == 1, 'Batch sizes larger than one currently not supported (clashes with flexible image format)'

    # Return dataloader
    return DataLoader(dataset=TableDataset(dir_data=dir_data, device=device, image_format=image_format),
                      batch_size=batch_size, shuffle=shuffle)