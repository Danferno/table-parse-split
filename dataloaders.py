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

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from math import ceil, sqrt
import warnings


from models import (Meta, Sample, Features, Targets,
                   SeparatorTargets, SeparatorFeatures,
                   COMMON_VARIABLES, COMMON_VARIABLES_SEPARATORLEVEL, COMMON_GLOBAL_VARIABLES, ROW_VARIABLES, COL_VARIABLES)
# Batch facilitators
class CollateFnLine:
    def __call__(self, batch):
        # Elements to pad
        elementsToPad = {'targets': ['row_line', 'col_line'], 'features': ['row', 'col', 'row_global', 'col_global']}

        # Get largest length
        rowLengths = [sample.meta.size_image['row'] for sample in batch]
        colLengths = [sample.meta.size_image['col'] for sample in batch]
        maxRowLength = max(rowLengths)
        maxColLength = max(colLengths)

        # Pad
        newTargets = defaultdict(list)
        newFeatures = defaultdict(list)
        newMeta = defaultdict(list)
        for idx, sample in enumerate(batch):        # idx = 0; sample = batch[idx]
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
class CollateFnSeparator:
    def __init__(self, ground_truth):
        super().__init__()

        # Parameters
        self.ground_truth = ground_truth

    def __call__(self, batch):
        # Elements to pad
        elementsToPad = {'targets': ['row', 'col'], 'features': ['row', 'col', 'row_global', 'col_global']}

        # Get largest length (image & separators)
        rowLengths = [sample.meta.size_image['row'] for sample in batch]
        colLengths = [sample.meta.size_image['col'] for sample in batch]
        maxRowLength = max(rowLengths)
        maxColLength = max(colLengths)

        rowSeparatorCount = [sample.features.row.shape[0] for sample in batch]
        colSeparatorCount = [sample.features.col.shape[0] for sample in batch]
        maxRowSeparatorCount = max(rowSeparatorCount)
        maxColSeparatorCount = max(colSeparatorCount)

        # Pad
        newTargets = defaultdict(list)
        newFeatures = defaultdict(list)
        newMeta = defaultdict(list)
        for idx, sample in enumerate(batch):        # idx = 1; sample = batch[idx]
            padLengths = dict(row = maxRowLength - rowLengths[idx], col = maxColLength - colLengths[idx])
            padSeparators = dict(row = maxRowSeparatorCount - rowSeparatorCount[idx], col = maxColSeparatorCount - colSeparatorCount[idx])

            if self.ground_truth:
                targets = sample.targets._asdict()
            features = sample.features._asdict()
            meta = sample.meta._asdict()
            img = features['image']
            
            # Pad | Image
            newFeatures['image'].append(pad(img, pad=(0, padLengths['col'], 0, padLengths['row']), mode='replicate'))
            
            if self.ground_truth:
                # Pad | Targets                        
                for element in targets.keys():      # element = next(iter(targets.keys()))
                    padLength = padSeparators['row'] if 'row' in element else padSeparators['col']
                    if element in elementsToPad['targets']:
                        init = targets[element]
                        padTensor = repeat(init[-1], pattern='last -> repeat last', repeat=padLength)
                        newTargets[element].append(torch.concat([init, padTensor]))
                    else:
                        newTargets[element].append(targets[element])
            else:
                for field in SeparatorTargets._fields:
                    newTargets[field].append(torch.zeros(size=(1,1), device=sample.targets.device))

            # Pad | Features
            for element in features.keys():      # element = next(iter(features.keys()))
                padLength = padSeparators['row'] if 'row' in element else padSeparators['col']
                if element in elementsToPad['features']:
                    init = features[element]
                    padTensor = repeat(init[-1], pattern='last -> repeat last', repeat=padLength)
                    newFeatures[element].append(torch.concat([init, padTensor]))
                elif element in ['image', 'proposedSeparators_row', 'proposedSeparators_col']:
                    continue
                else:
                    newFeatures[element].append(features[element])

            # Pad | Proposed separators
            for orientation in ['row', 'col']:      # orientation = 'row'
                element = f'proposedSeparators_{orientation}'
                padLength = padSeparators[orientation]
                init = features[element]
                lastPixel = maxRowLength if orientation == 'row' else maxColLength
                padTensor = repeat(torch.tensor([lastPixel-2, lastPixel-1]), pattern='last -> repeat last', repeat=padLength)
                newFeatures[element].append(torch.concat([init, padTensor]))

            # Pad | Meta
            for element in meta.keys():
                newMeta[element].append(meta[element])

        # Stack and return
        return Sample(features = SeparatorFeatures(**{key: torch.stack(value) for key, value in newFeatures.items()}),
                      targets  = SeparatorTargets(**{key: torch.stack(value) for key, value in newTargets.items() }),
                      meta     = Meta(**newMeta))

class BucketBatchSampler(Sampler):
    # want inputs to be an array
    def __init__(self, dataset, batch_size, shuffle, bin_approach=None, path_out_plot=None, show_naive=False):
        self.batch_size = batch_size
        self.dataset = dataset
        self.shuffle = shuffle
        self.bin_approach = bin_approach
        self.path_out_plot = path_out_plot 
        self.show_naive = show_naive

        self.batch_list = self._generate_batch_map()
        self.num_batches = len(self.batch_list)

    def chunkify(self, lst, chunk_size):
        return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

    @staticmethod
    def visualize_bins(df, cols_to_bin, outpath):
        vizdf = df.copy()
        vizdf['withinID'] = vizdf.groupby('batch_bucket_id').cumcount()
        vizdf[f'{cols_to_bin[0]}_jit'] = vizdf[cols_to_bin[0]] + np.random.uniform(low=0, high=0.8, size=len(vizdf))
        vizdf[f'{cols_to_bin[1]}_jit'] = vizdf[cols_to_bin[1]] + np.random.uniform(low=0, high=0.8, size=len(vizdf))

        rectangles = vizdf.groupby('batch_bucket_id').agg({cols_to_bin[0]: ['min', 'max'], cols_to_bin[1]: ['min', 'max']})
        rectangles.columns = rectangles.columns.map('_'.join)
        rectangles['x0'] = rectangles[f'{cols_to_bin[0]}_min'] - 0.1
        rectangles['y0'] = rectangles[f'{cols_to_bin[1]}_min'] - 0.1
        rectangles['x1'] = rectangles[f'{cols_to_bin[0]}_max'] + 0.9
        rectangles['y1'] = rectangles[f'{cols_to_bin[1]}_max'] + 0.9
        rectangles['width']  = rectangles['x1'] - rectangles['x0']
        rectangles['height'] = rectangles['y1'] - rectangles['y0']

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sns.scatterplot(x=f'{cols_to_bin[0]}_jit', y=f'{cols_to_bin[1]}_jit', data=vizdf, marker='o', s=15, hue='batch_bucket_id', palette=sns.color_palette('Paired'), legend=False)
        ax = plt.gca()
        for _, rectangle in rectangles.iterrows():
            xy, width, height = (rectangle['x0'], rectangle['y0']), rectangle['width'], rectangle['height']
            _ = ax.add_patch(Rectangle(xy=xy, width=width, height=height, fill=None, alpha=0.2))
        plt.ioff()
        plt.savefig(outpath, dpi=300, bbox_inches='tight', format='png')

    @staticmethod
    def greedy_binner(df, cols_to_bin, batch_size, path_out_plot=None, show_naive=False):
        def greedy_binner_innerloop(df_notok, batch_size, cols_to_bin, df_ok=pd.DataFrame()):
            # Split into buckets
            desired_bins = ceil(len(df_notok) / batch_size)

            if desired_bins > 2:
                bin_count = ceil(sqrt(desired_bins))
                _, x_edges, y_edges = np.histogram2d(x=df_notok[cols_to_bin[0]], y=df_notok[cols_to_bin[1]], bins=[bin_count, bin_count])

                # Assign obs to large buckets
                df_notok['bucket_1'] = np.digitize(df_notok[cols_to_bin[0]], x_edges) - 1
                df_notok['bucket_2'] = np.digitize(df_notok[cols_to_bin[1]], y_edges) - 1
                df_notok['bucket_id'] = df_notok.groupby(['bucket_1', 'bucket_2']).ngroup()
                df_notok = df_notok.drop(columns=['bucket_1', 'bucket_2'])

                # Split large buckets into batch_sized buckets
                df_notok['within_bucket_n'] = df_notok.groupby('bucket_id').cumcount()
                df_notok['batch_bucket_id_within'] = df_notok['within_bucket_n'] // batch_size
                df_notok['batch_bucket_id'] = df_notok.groupby(['bucket_id', 'batch_bucket_id_within']).ngroup()
                df_notok['batch_bucket_size'] = df_notok.groupby(['batch_bucket_id'])[df_notok.columns[0]].transform('size')

                # Split OK and too-small buckets
                df_ok_new = df_notok.loc[(df_notok['batch_bucket_size'] == batch_size), ['batch_bucket_id']]
                df_notok = df_notok.loc[(~df_notok.index.isin(df_ok_new.index)), [cols_to_bin[0], cols_to_bin[1]]]
            
            else:
                df_notok['batch_bucket_id'] = (df_notok.reset_index().index < 5).astype(np.int8)
                df_ok_new = df_notok[['batch_bucket_id']]
                df_notok = df_notok.loc[(~df_notok.index.isin(df_ok_new.index)), [cols_to_bin[0], cols_to_bin[1]]]

            # Merge
            df_ok = pd.merge(left=df_ok, right=df_ok_new, how='outer', left_index=True, right_index=True)
            if 'batch_bucket_id_x' in df_ok.columns:
                df_ok['batch_bucket_id'] = df_ok.groupby(['batch_bucket_id_x', 'batch_bucket_id_y'], dropna=False).ngroup()
                df_ok = df_ok.drop(columns=['batch_bucket_id_x', 'batch_bucket_id_y'])

            print(f'Attempted to make {desired_bins:>4d} bins. Left with {len(df_notok):>5d} obs to distribute.')
            return df_ok, df_notok

        if len(cols_to_bin) == 1:
            raise ValueError('Single column binner not yet implemented')
        elif len(cols_to_bin) == 2:
            # Initialize
            df_notok = df.copy()
            df_ok = pd.DataFrame()
            delta_obs = len(df_notok)

            # Bin greedily
            while len(df_notok) & (delta_obs):
                obs_to_distribute_initial = len(df_notok)
                df_ok, df_notok = greedy_binner_innerloop(df_notok=df_notok, df_ok=df_ok, batch_size=batch_size, cols_to_bin=cols_to_bin)
                delta_obs = obs_to_distribute_initial - len(df_notok)

            df_ok = df_ok.merge(right=df[cols_to_bin], left_index=True, right_index=True)
        else:
            raise ValueError('Dims > 2 not yet implemented')
        
        if path_out_plot:
            BucketBatchSampler.visualize_bins(df=df_ok, cols_to_bin=cols_to_bin, outpath=path_out_plot)
        
        if (path_out_plot is not None) & (show_naive):
            df_naive = df.copy()
            df_naive['batch_bucket_id'] = df_naive.reset_index().index // batch_size
            BucketBatchSampler.visualize_bins(df=df_naive, cols_to_bin=cols_to_bin, outpath=os.path.splitext(path_out_plot)[0]+'_naive'+os.path.splitext(path_out_plot)[1])

        batch_list = df_ok.groupby('batch_bucket_id').apply(lambda group: group.index.tolist()).tolist()

        return batch_list

    def _generate_batch_map(self):
        # Set parameters for bin approach
        if self.bin_approach == 'separator':
            samples = [{'idx': idx, **sample.meta.count_separators} for idx, sample in enumerate(self.dataset)]         # Separator counts
            df = pd.DataFrame.from_records(samples).set_index('idx', drop=True)
            cols_to_bin = ['row', 'col']
        elif self.bin_approach == 'line':
            samples = [{'idx': idx, **sample.meta.size_image} for idx, sample in enumerate(self.dataset)]               # Image sizes
            df = pd.DataFrame.from_records(samples).set_index('idx', drop=True)
            cols_to_bin = ['row', 'col']
        else:
            raise ValueError('Not yet implemented')
        
        # Apply binner
        batch_list = BucketBatchSampler.greedy_binner(df=df, cols_to_bin=cols_to_bin, batch_size=self.batch_size, path_out_plot=self.path_out_plot, show_naive=self.show_naive)
        return batch_list

    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return self.num_batches

    def __iter__(self):
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
        meta = Meta(path_image=pathImage, size_image={'row': image.shape[1], 'col': image.shape[2]}, **metaData, count_separators=None)

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
        self.dir_features = dir_data / 'features_separatorLevel'
        self.dir_predictions_line = dir_predictions_line or dir_data / 'predictions_lineLevel'

        if ground_truth:
            self.dir_targets = dir_data / 'targets_separatorLevel'


        if legacy_folder_names:
            self.dir_images = dir_data / 'images'
            self.dir_meta_lineLevel = dir_data / 'meta'
        
        self.image_format = image_format

        # Get items
        self.items = sorted([os.path.splitext(filename)[0] for filename in os.listdir(self.dir_predictions_line)])

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
        meta = Meta(path_image=path_image, size_image={'row': image.shape[1], 'col': image.shape[2]}, count_separators={'row': features_row.shape[0], 'col': features_col.shape[0]}, **metaData)

        # Return sample
        sample = Sample(meta=meta, features=features, targets=targets)
        return sample
    
def get_dataloader_lineLevel(dir_data:Union[Path, str], ground_truth=False, legacy_folder_names=False, batch_size:int=1, shuffle:bool=True, device='cuda', image_format:str='.png', show_naive=False) -> DataLoader:
    # Parameters
    dir_data = Path(dir_data)
    image_format = f'.{image_format}' if not image_format.startswith('.') else image_format

    # Return dataloader
    dataset = LineDataset(dir_data=dir_data, ground_truth=ground_truth, legacy_folder_names=legacy_folder_names, device=device, image_format=image_format)
    batch_sampler = BucketBatchSampler(dataset=dataset, batch_size=batch_size, shuffle=shuffle, bin_approach='line', path_out_plot=dir_data / 'bin_plot_lineLevel.png', show_naive=show_naive)
    return DataLoader(dataset=dataset, batch_sampler=batch_sampler, collate_fn=CollateFnLine())

def get_dataloader_separatorLevel(dir_data:Union[Path, str], ground_truth=False, batch_size:int=1, shuffle:bool=True, device='cuda', image_format:str='.png', show_naive:bool=False):
    # Parameters
    dir_data = Path(dir_data)
    image_format = f'.{image_format}' if not image_format.startswith('.') else image_format

    # Return dataloader
    dataset = SeparatorDataset(dir_data=dir_data, ground_truth=ground_truth, image_format=image_format, device=device)
    batch_sampler = BucketBatchSampler(dataset=dataset, batch_size=batch_size, shuffle=shuffle, bin_approach='separator', path_out_plot=dir_data / 'bin_plot_separatorLevel.png', show_naive=show_naive)
    collate_fn = CollateFnSeparator(ground_truth=ground_truth)
    return DataLoader(dataset=dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)


if __name__ == '__main__':
    PATH_ROOT = Path(r'F:\ml-parsing-project\table-parse-split\data\tableparse_round2\splits')
    # dataloader =      get_dataloader_lineLevel(dir_data=PATH_ROOT / 'train', batch_size=5, ground_truth=True, show_naive=True)
    dataloader = get_dataloader_separatorLevel(dir_data=PATH_ROOT / 'train', ground_truth=True, batch_size=5, show_naive=True)

    for batch in dataloader:
        print(f'Image: {batch.meta.size_image}')
        print(f'Separator: {batch.meta.count_separators}')
        print(f'-----')