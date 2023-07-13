# Imports
from pathlib import Path
import os, json
from functools import cache
from typing import Union

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.io import read_image, ImageReadMode
from torch.nn.functional import pad
from einops import repeat
from collections import defaultdict, OrderedDict
from random import shuffle


from model import (Meta, Sample, Features, Targets,
                   SeparatorTargets, SeparatorFeatures,
                   COMMON_VARIABLES, COMMON_VARIABLES_SEPARATORLEVEL, COMMON_GLOBAL_VARIABLES, ROW_VARIABLES, COL_VARIABLES)
# Batch facilitators
class CollateFn:
    def __call__(self, batch):
        # Elements to pad
        elementsToPad = {'targets': ['row_line', 'col_line'], 'features': ['row', 'col', 'row_global', 'col_global']}

        # Get largest length
        rowLengths = [sample.features.row.shape[0] for sample in batch]
        colLengths = [sample.features.col.shape[0] for sample in batch]
        maxRowLength = max(rowLengths)
        maxColLength = max(colLengths)

        # Pad
        newTargets = defaultdict(list)
        newFeatures = defaultdict(list)
        newMeta = defaultdict(list)
        for idx, sample in enumerate(batch):        # idx = 0; sample = newSamples[idx]
            padLengths = dict(row = maxRowLength - rowLengths[idx], col = maxColLength - colLengths[idx])
            targets = sample.targets._asdict()
            features = sample.features._asdict()
            meta = sample.meta._asdict()
            img = features['image']
            
            # Pad | Targets                        
            for element in targets.keys():      # element = next(iter(targets.keys()))
                padLength = padLengths['row'] if 'row' in element else padLengths['col']
                if element in elementsToPad['targets']:
                    init = targets[element]
                    padTensor = repeat(init[-1], pattern='last -> repeat last', repeat=padLength)
                    newTargets[element].append(torch.concat([init, padTensor]))
                else:
                    newTargets[element].append(targets[element])

            # Pad | Features
            for element in features.keys():      # element = next(iter(features.keys()))
                padLength = padLengths['row'] if 'row' in element else padLengths['col']
                if element in elementsToPad['features']:
                    init = features[element]
                    padTensor = repeat(init[-1], pattern='last -> repeat last', repeat=padLength)
                    newFeatures[element].append(torch.concat([init, padTensor]))
                elif element == 'image':
                    continue
                else:
                    newFeatures[element].append(features[element])

            # Pad | Image
            newFeatures['image'].append(pad(img, pad=(0, padLengths['col'], 0, padLengths['row']), mode='replicate'))

            # Pad | Meta
            for element in meta.keys():
                newMeta[element].append(meta[element])

        # Stack and return
        return Sample(features=Features(**{key: torch.stack(value) for key, value in newFeatures.items()}),
                            targets= Targets(**{key: torch.stack(value) for key, value in newTargets.items() }),
                            meta=Meta(**newMeta))

class BucketBatchSampler(Sampler):
    # want inputs to be an array
    def __init__(self, dataset, batch_size, shuffle):
        self.batch_size = batch_size
        self.dataset = dataset
        self.batch_list = self._generate_batch_map()
        self.num_batches = len(self.batch_list)
        self.shuffle = shuffle

    def chunkify(self, lst, chunk_size):
        return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

    def _generate_batch_map(self):
        # Get image sizes for each sample
        samples = {idx: sample.meta.size_image for idx, sample in enumerate(self.dataset)}

        # Sort by X and Y coordinates
        samples_sorted = sorted(samples.items(), key=lambda item: (item[1]['row'], item[1]['col']))
        sample_indices = list(map(lambda entry: entry[0], samples_sorted))

        # Split into samples of size batch_size
        batch_list = self.chunkify(sample_indices, self.batch_size)
        return batch_list

    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        self.batch_list = self._generate_batch_map()
        if self.shuffle:
            shuffle(self.batch_list)
        for i in self.batch_list:
            yield i

# Datasets
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

        # Load sample | Meta
        with open(pathMeta, 'r') as f:
            metaData = json.load(f)
        meta = Meta(path_image=pathImage, size_image={'row': image.shape[1], 'col': image.shape[2]}, **metaData)

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

        # Load sample | Meta
        with open(path_meta_lineLevel, 'r') as f:
            metaData = json.load(f)
        meta = Meta(path_image=path_image, size_image=image.shape, **metaData)

        # Return sample
        sample = Sample(meta=meta, features=features, targets=targets)
        return sample
    
def get_dataloader_lineLevel(dir_data:Union[Path, str], ground_truth=False, legacy_folder_names=False, batch_size:int=1, shuffle:bool=True, device='cuda', image_format:str='.png') -> DataLoader:
    # Parameters
    dir_data = Path(dir_data)
    image_format = f'.{image_format}' if not image_format.startswith('.') else image_format

    # Return dataloader
    dataset = LineDataset(dir_data=dir_data, ground_truth=ground_truth, legacy_folder_names=legacy_folder_names, device=device, image_format=image_format)
    batch_sampler = BucketBatchSampler(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return DataLoader(dataset=dataset, batch_sampler=batch_sampler, collate_fn=CollateFn())

def get_dataloader_separatorLevel(dir_data:Union[Path, str], ground_truth=False, batch_size:int=1, shuffle:bool=True, device='cuda', image_format:str='.png', dir_data_all=None):
    # Parameters
    dir_data = Path(dir_data)
    dir_data_all = dir_data_all or dir_data.parent / 'all'
    image_format = f'.{image_format}' if not image_format.startswith('.') else image_format

    # Return dataloader
    return DataLoader(dataset=SeparatorDataset(dir_data=dir_data, ground_truth=ground_truth, dir_data_all=dir_data_all, image_format=image_format, device=device), batch_size=batch_size, shuffle=shuffle)


if __name__ == '__main__':
    PATH_ROOT = Path(r'F:\ml-parsing-project\table-parse-split\data\tableparse_round2\splits')
    dataloader = get_dataloader_lineLevel(dir_data=PATH_ROOT / 'val', batch_size=3, ground_truth=True)

    for batch in dataloader:
        print(batch.meta.size_image)