# Imports
import os
from pathlib import Path
from lxml import etree
from math import floor, ceil
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import cv2 as cv
import shutil
import matplotlib.pyplot as plt
from typing import Literal
from datetime import datetime
import json
import pandas as pd
import random
from uuid import uuid4

# Constants
PATH_ROOT = Path(r"F:\ml-parsing-project\table-parse-split")
PATH_DATA = PATH_ROOT / 'data'
IMAGE_FORMAT = '.jpg'
COMPLEXITY = ['avg-matters', 'dash-matters']

# Path stuff
pathOut = PATH_DATA / f'fake_{len(COMPLEXITY)}'
pathAll = pathOut / 'all'

# Classes and helper functions
def replaceDirs(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        shutil.rmtree(path)
        os.makedirs(path)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
class Block(dict):
    def __init__(self, purpose, color_average, pattern):
        self.purpose = purpose
        self.color_average = color_average if color_average is not None else None
        self.pattern = pattern
    def __repr__(self) -> str:
        return f"({self.purpose[:7]}, color {self.color_average}, {self.pattern})"
    
# Make folders
replaceDirs(pathOut)
replaceDirs(pathAll)
replaceDirs(pathAll / 'images')
replaceDirs(pathAll / 'labels')
replaceDirs(pathAll / 'features')

# Block generators
def typeToBlock(blockType, sizeOptions):
    dim1 = sizeOptions['separator_size'] if blockType.purpose == 'separator' else sizeOptions['block_size']
    dim2 = sizeOptions['otherDim_size']

    if blockType.purpose == 'no-separator':
        return None

    if blockType.pattern == 'uniform':
        block = np.full(fill_value=blockType.color_average, shape=(dim1, dim2), dtype=np.float32)
    elif blockType.pattern == 'dash50':
        block_dim1 = np.tile(A=[0, 1], reps=dim2 // 2).astype(np.float32)
        block = np.broadcast_to(block_dim1, shape=(dim1, dim2))
    else:
        raise ValueError(f'{blockType.pattern} pattern not supported')    

    return block

def generateBlock(separatorType, contentType, sizeOptions):
    # Content block
    block = typeToBlock(blockType=contentType, sizeOptions=sizeOptions)
    
    # Add separator block
    separator_block = typeToBlock(blockType=separatorType, sizeOptions=sizeOptions)
    if separator_block is not None:
        separator_startRow = sizeOptions['separator_location']
        separator_endRow = separator_startRow + separator_block.shape[0]
        block[separator_startRow:separator_endRow] = separator_block
    
    # for idx, row in enumerate(block):
    #     print(idx, ': ', row)

    return block

def generateGroundTruth(separatorType, sizeOptions):
    dimFull = sizeOptions['block_size']
    dimSeparator = sizeOptions['separator_size']

    separator_start = sizeOptions['separator_location']
    separator_end = separator_start + dimSeparator

    groundTruth = np.zeros(shape=(dimFull))
    if separatorType.purpose == 'separator':
        groundTruth[separator_start:separator_end] = 1

    return groundTruth

# Generate fake data
def generateSample(complexity=COMPLEXITY):
    '''Complexity:
        1: Square B/W image with thick, uniform rows '''
    # Shapes
    row_blockCount = 10
    col_blockCount = 10
    size_rowBlock = 60
    size_colBlock = 40
    img_shape = (row_blockCount * size_rowBlock, col_blockCount * size_colBlock)
    
    separator_row_size = size_rowBlock // 4
    separator_row_location = (size_rowBlock - separator_row_size) // 2

    row_size_options = {
        'block_size': size_rowBlock,
        'otherDim_size': size_colBlock*col_blockCount,
        'separator_size': separator_row_size,
        'separator_location': separator_row_location,
    }

    # Base image
    img_base = np.zeros(shape=img_shape, dtype=np.float32)
    
    # Block classes
    classes_separators = [Block(purpose='no-separator', color_average=None, pattern=None)]
    classes_content = []
    if 'avg-matters' in complexity:
        classes_separators.append(Block(purpose='separator', color_average=1, pattern='uniform'))
        classes_content.append(Block(purpose='content', color_average=0, pattern='uniform'))
    if 'dash-matters' in complexity:
        classes_separators.append(Block(purpose='separator', color_average=0.5, pattern='dash50'))
        classes_content.append(Block(purpose='content', color_average=0.5, pattern='uniform'))

    # Rows
    # Rows | Generate visual
    row_separatorTypes = np.random.choice(classes_separators, size=row_blockCount, replace=True)
    row_contentTypes = np.random.choice(classes_content, size=row_blockCount, replace=True)
    rowBlocks = [generateBlock(separatorType=row_separatorTypes[i], contentType=row_contentTypes[i], sizeOptions=row_size_options) for i in range(row_blockCount)]        # i = 1
    rowContribution = np.concatenate(rowBlocks)

    # Rows | Get separator features
    row_separator_gt = np.concatenate([generateGroundTruth(separatorType=row_separatorTypes[i], sizeOptions=row_size_options) for i in range(row_blockCount)])

    # Combine
    # Combine | Image
    img = img_base + rowContribution
    img = np.clip(img, a_min=0, a_max=1)

    # Combine | Ground Truth
    gt = {}
    gt['row'] = row_separator_gt

    # Extract features
    features = {}
    features['row_absDiff'] = np.asarray([np.absolute(np.diff(row)).mean() for row in img])
    features['row_avg'] = np.asarray([np.mean(row) for row in img])
    
    # Save
    name = str(uuid4())
    cv.imwrite(filename=str(pathAll /'images' / f'{name}.jpg'), img=img*255)

    with open(pathAll / 'labels' / f'{name}.json', 'w') as groundTruthFile:
        json.dump(gt, groundTruthFile, cls=NumpyEncoder)
    with open(pathAll / 'features' / f'{name}.json', 'w') as featureFile:
        json.dump(features, featureFile, cls=NumpyEncoder)
    
for i in tqdm(range(120), desc=f"Generating fake images of complexity {COMPLEXITY}"):
    generateSample()


# Split into train/val/test
def splitData(pathIn=pathAll, trainRatio=0.8, valRatio=0.1):
    items = [os.path.splitext(entry.name)[0] for entry in os.scandir(pathIn / 'images')]
    random.shuffle(items)

    dataSplit = {}
    dataSplit['train'], dataSplit['val'], dataSplit['test'] = np.split(items, indices_or_sections=[int(len(items)*trainRatio), int(len(items)*(trainRatio+valRatio))])

    for subgroup in dataSplit:      # subgroup = list(dataSplit.keys())[0]
        destPath = pathIn.parent / subgroup
        os.makedirs(destPath / 'images'); os.makedirs(destPath / 'labels'); os.makedirs(destPath / 'features')

        for item in tqdm(dataSplit[subgroup], desc=f"Copying from all > {subgroup}"):        # item = dataSplit[subgroup][0]
            _ = shutil.copyfile(src=pathIn / 'images'   / f'{item}.jpg',  dst=destPath / 'images' / f'{item}.jpg')
            _ = shutil.copyfile(src=pathIn / 'labels'   / f'{item}.json', dst=destPath / 'labels' / f'{item}.json')
            _ = shutil.copyfile(src=pathIn / 'features' / f'{item}.json', dst=destPath / 'features' / f'{item}.json')

splitData()