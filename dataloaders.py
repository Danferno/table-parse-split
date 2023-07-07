# Imports
from pathlib import Path
import os, json
from functools import cache
from typing import Union

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode


from model import (Meta, Sample, Features, Targets,
                   SeparatorTargets, SeparatorFeatures,
                   COMMON_VARIABLES, COMMON_VARIABLES_SEPARATORLEVEL, COMMON_GLOBAL_VARIABLES, ROW_VARIABLES, COL_VARIABLES)

class LineDataset(Dataset):
    def __init__(self, dir_data, ground_truth, legacy_folder_names=False,
                    device='cuda', image_format='.png'):
        # Store in self 
        dir_data = Path(dir_data)
        self.legacy_folder_names = legacy_folder_names
        self.ground_truth = ground_truth
        self.device = device

        if legacy_folder_names:
            self.dir_images = dir_data / 'images'
            self.dir_features = dir_data / 'features'
            self.dir_labels = dir_data / 'labels'
            self.dir_meta = dir_data / 'meta'
        else:
            self.dir_images = dir_data / 'tables_images'
            self.dir_features = dir_data / 'features_lineLevel'
            self.dir_labels = dir_data / 'targets_lineLevel'
            self.dir_meta = dir_data / 'meta_lineLevel'
        self.image_format = image_format

        # Get filepaths
        self.image_fileEntries     = list(os.listdir(self.dir_images))
        self.feature_fileEntries   = list(os.listdir(self.dir_features))
        self.meta_fileEntries      = list(os.listdir(self.dir_meta))
        if self.ground_truth:
            self.label_fileEntries = list(os.listdir(self.dir_labels))

        items_images     = set([os.path.splitext(file)[0] for file in self.image_fileEntries])
        items_features   = set([os.path.splitext(file)[0] for file in self.feature_fileEntries])
        items_meta       = set([os.path.splitext(file)[0] for file in self.meta_fileEntries])
        if self.ground_truth:
            items_labels = set([os.path.splitext(file)[0] for file in self.label_fileEntries])

        # Verify consistency between image/feature/label
        if ground_truth:
            self.items = sorted(list(items_images.intersection(items_features).intersection(items_labels).intersection(items_meta)))
            assert len(self.items) == len(items_images) == len(items_features) == len(items_labels) == len(items_meta), 'Set of images, meta, features and labels do not match.'
        else:
            self.items = sorted(list(items_images.intersection(items_features).intersection(items_meta)))
            assert len(self.items) == len(items_images) == len(items_features) == len(items_meta), 'Set of images, meta and features do not match.'

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
        if self.ground_truth:
            with open(pathLabel, 'r') as f:
                labelData = json.load(f)
            row_targets = torch.tensor(labelData['row'])
            col_targets = torch.tensor(labelData['col'])
            row_separator_count = torch.tensor(torch.where(torch.diff(row_targets) == 1)[0].numel(), dtype=torch.int32)
            col_separator_count = torch.tensor(torch.where(torch.diff(col_targets) == 1)[0].numel(), dtype=torch.int32)
            
            targets = Targets(row_line=row_targets.unsqueeze(-1).to(self.device), row_separator_count = row_separator_count.to(self.device),
                                col_line=col_targets.unsqueeze(-1).to(self.device), col_separator_count = col_separator_count.to(self.device))
        else:
            targets = torch.zeros(size=(1,1)).to(self.device)
    
        # Load sample | Features
        with open(pathFeatures, 'r') as f:
            featuresData = json.load(f)

        # Load sample | Features | Rows
        features_row = torch.cat(tensors=[torch.tensor(featuresData[common_variable.format('row')]).unsqueeze(-1) for common_variable in COMMON_VARIABLES], dim=1)
        features_row_specific = torch.cat([torch.tensor(featuresData[row_variable]).unsqueeze(-1) for row_variable in ROW_VARIABLES], dim=1)
        features_row_adhoc = (torch.arange(start=0, end=features_row.shape[0])/features_row.shape[0]).unsqueeze(-1)
        features_row = torch.cat([features_row, features_row_specific, features_row_adhoc], dim=1)

        features_row_global = torch.cat([torch.tensor([featuresData[global_common_variable.format('row')]]).broadcast_to((features_row.shape[0], 1)) for global_common_variable in COMMON_GLOBAL_VARIABLES], dim=1)

        # Load sample | Features | Cols
        features_col = torch.cat(tensors=[torch.tensor(featuresData[common_variable.format('col')]).unsqueeze(-1) for common_variable in COMMON_VARIABLES], dim=1)
        features_col_specific = torch.cat([torch.tensor(featuresData[col_variable]).unsqueeze(-1) for col_variable in COL_VARIABLES], dim=1)
        features_col_adhoc = (torch.arange(start=0, end=features_col.shape[0])/features_col.shape[0]).unsqueeze(-1)
        features_col = torch.cat([features_col, features_col_specific, features_col_adhoc], dim=1)

        features_col_global = torch.cat([torch.tensor([featuresData[global_common_variable.format('col')]]).broadcast_to((features_col.shape[0], 1)) for global_common_variable in COMMON_GLOBAL_VARIABLES], dim=1)
    
        # Load sample | Features | Image (0-1 float)
        image = read_image(pathImage, mode=ImageReadMode.GRAY).to(dtype=torch.float32) / 255

        # Collect in namedtuples
        features = Features(row=features_row, col=features_col, row_global=features_row_global, col_global=features_col_global, image=image)
        sample = Sample(features=features, targets=targets, meta=meta)

        # Return sample
        return sample
    
class SeparatorDataset(Dataset):
    def __init__(self, dir_data, legacy_folder_names=False, ground_truth=False, dir_data_all=None, dir_predictions_line=None, device='cuda', image_format='.png'):
        # Store in self
        dir_data = Path(dir_data)
        dir_data_all = dir_data_all or dir_data.parent / 'all'
        self.ground_truth = ground_truth
        self.device = device
        self.dir_images = dir_data / 'tables_images'
        self.dir_meta_lineLevel = dir_data / 'meta_lineLevel'

        if ground_truth:
            self.dir_features = dir_data_all / 'features_separatorLevel'
            self.dir_targets = dir_data_all / 'targets_separatorLevel'
            self.dir_predictions_line = dir_predictions_line or dir_data_all / 'predictions_lineLevel'
        else:
            self.dir_features = dir_data / 'features_separatorLevel'
            self.dir_predictions_line = dir_predictions_line or dir_data / 'predictions_lineLevel'

        if legacy_folder_names:
            self.dir_images = dir_data / 'images'
            self.dir_meta_lineLevel = dir_data / 'meta'
        
        self.image_format = image_format

        # Get items
        self.items = sorted([os.path.splitext(filename)[0] for filename in os.listdir(self.dir_images)])

        # Exclude items without row or column separators
        tables_without_separators = []
        for tableName in self.items:        # tableName = self.items[0]
            path_proposedSeparators = str(self.dir_predictions_line / f'{tableName}.json')
            with open(path_proposedSeparators, 'r') as f:
                proposedSeparatorData = json.load(f)
            if not all([len(value) > 0 for value in proposedSeparatorData.values()]):
                tables_without_separators.append(tableName)

        if tables_without_separators:
            print(f'WARNING: Excluding {len(tables_without_separators)} tables from the data because they do not contain row or column separator predictions')
            self.items = list(set(self.items) - set(tables_without_separators))

        
    def __len__(self):
        return len(self.items)
    
    @cache
    def __getitem__(self, idx):
        # Generate paths for specific items
        tableName = self.items[idx]
        path_image = str(self.dir_images / f'{tableName}{self.image_format}')
        path_meta_lineLevel = str(self.dir_meta_lineLevel / f'{self.items[idx]}.json')
        path_features = str(self.dir_features / f'{tableName}.json')
        path_proposedSeparators = str(self.dir_predictions_line / f'{tableName}.json')

        # Load sample
        # Load sample | Meta
        with open(path_meta_lineLevel, 'r') as f:
            metaData = json.load(f)
        meta = Meta(path_image=path_image, **metaData)

        # Load sample | Targets
        if self.ground_truth:
            path_targets = str(self.dir_targets / f'{tableName}.json')
            with open(path_targets, 'r') as f:
                targetData = json.load(f)
            targets_row = torch.tensor(targetData['row'], dtype=torch.int32)
            targets_col = torch.tensor(targetData['col'], dtype=torch.int32)

            targets = SeparatorTargets(row=targets_row.unsqueeze(-1).to(self.device),
                                    col=targets_col.unsqueeze(-1).to(self.device))
        else:
            targets = torch.zeros(size=(1,1)).to(self.device)
        
        # Load sample | Features
        with open(path_features, 'r') as f:
            featureData = json.load(f)
        with open(path_proposedSeparators, 'r') as f:
            proposedSeparatorData = json.load(f)
        
        features_row = torch.cat(tensors=[torch.tensor(featureData[common_variable.format('row')]).unsqueeze(-1) for common_variable in COMMON_VARIABLES_SEPARATORLEVEL], dim=-1)
        features_row_min = torch.cat(tensors=[torch.tensor(featureData[common_variable.format('row')+'_min']).unsqueeze(-1) for common_variable in COMMON_VARIABLES], dim=-1)
        features_row_max = torch.cat(tensors=[torch.tensor(featureData[common_variable.format('row')+'_max']).unsqueeze(-1) for common_variable in COMMON_VARIABLES], dim=-1)
        features_row_global = torch.cat([torch.tensor([featureData[global_common_variable.format('row')]]).broadcast_to((features_row.shape[0], 1)) for global_common_variable in COMMON_GLOBAL_VARIABLES], dim=1)
        features_row = torch.cat([features_row, features_row_min, features_row_max, features_row_global], dim=-1)

        features_col = torch.cat(tensors=[torch.tensor(featureData[common_variable.format('col')]).unsqueeze(-1) for common_variable in COMMON_VARIABLES_SEPARATORLEVEL], dim=-1)
        features_col_min = torch.cat(tensors=[torch.tensor(featureData[common_variable.format('col')+'_min']).unsqueeze(-1) for common_variable in COMMON_VARIABLES], dim=-1)
        features_col_max = torch.cat(tensors=[torch.tensor(featureData[common_variable.format('col')+'_max']).unsqueeze(-1) for common_variable in COMMON_VARIABLES], dim=-1)
        features_col_global = torch.cat([torch.tensor([featureData[global_common_variable.format('col')]]).broadcast_to((features_col.shape[0], 1)) for global_common_variable in COMMON_GLOBAL_VARIABLES], dim=1)
        features_col = torch.cat([features_col, features_col_min, features_col_max, features_col_global], dim=-1)

        proposedSeparators_row = torch.tensor(proposedSeparatorData['row_separator_predictions'])
        proposedSeparators_col = torch.tensor(proposedSeparatorData['col_separator_predictions'])

        image = read_image(path_image, mode=ImageReadMode.GRAY).to(dtype=torch.float32) / 255

        features = SeparatorFeatures(row=features_row, col=features_col, image=image, proposedSeparators_row=proposedSeparators_row, proposedSeparators_col=proposedSeparators_col)        

        # Return sample
        sample = Sample(meta=meta, features=features, targets=targets)
        return sample
    
def get_dataloader_lineLevel(dir_data:Union[Path, str], ground_truth=False, legacy_folder_names=False, batch_size:int=1, shuffle:bool=True, device='cuda', image_format:str='.png') -> DataLoader:
    # Parameters
    dir_data = Path(dir_data)
    image_format = f'.{image_format}' if not image_format.startswith('.') else image_format

    # Checks
    assert batch_size == 1, 'Batch sizes larger than one currently not supported (clashes with flexible image format)'

    # Return dataloader
    return DataLoader(dataset=LineDataset(dir_data=dir_data, ground_truth=ground_truth, legacy_folder_names=legacy_folder_names, device=device, image_format=image_format),
                      batch_size=batch_size, shuffle=shuffle)

def get_dataloader_separatorLevel(dir_data:Union[Path, str], ground_truth=False, batch_size:int=1, shuffle:bool=True, device='cuda', image_format:str='.png', dir_data_all=None):
    # Parameters
    dir_data = Path(dir_data)
    dir_data_all = dir_data_all or dir_data.parent / 'all'
    image_format = f'.{image_format}' if not image_format.startswith('.') else image_format

    # Checks
    assert batch_size == 1, 'Batch sizes larger than one currently not supported (clashes with flexible image format)'

    # Return dataloader
    return DataLoader(dataset=SeparatorDataset(dir_data=dir_data, ground_truth=ground_truth, dir_data_all=dir_data_all, image_format=image_format, device=device), batch_size=batch_size, shuffle=shuffle)


if __name__ == '__main__':
    PATH_ROOT = Path(r'F:\ml-parsing-project\table-parse-split\data\real_narrow')
    get_dataloader_separatorLevel(dir_data=PATH_ROOT / 'train')