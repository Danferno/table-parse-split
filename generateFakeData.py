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
COMPLEXITY = 1

# Path stuff
pathOut = PATH_DATA / f'fake_{COMPLEXITY}'
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
    
# Make folders
replaceDirs(pathOut)
replaceDirs(pathAll)
replaceDirs(pathAll / 'images')
replaceDirs(pathAll / 'labels')
replaceDirs(pathAll / 'features')

# Generate fake data
def generateSeparatorBlock(blockSize, separator_size, separator_location, separator_include, otherSize, complexity=COMPLEXITY):
    # Base block
    block = np.zeros((blockSize, otherSize), dtype=np.uint8)
    
    # Add separator
    if separator_include:
        block[separator_location:separator_location+separator_size, :] = 1
    
    return block

def generateSample(complexity=COMPLEXITY):
    '''Complexity:
        1: Square B/W image with thick, uniform rows '''
    
    # Shapes
    img_blockCount = (10, 10)
    shape_rowBlock = 60
    shape_colBlock = 40
    img_shape = (img_blockCount[0] * shape_rowBlock, img_blockCount[1] * shape_colBlock)
    shape_row = 15
    separatorLocation_row = floor((shape_rowBlock-shape_row) * 0.5)

    # Base image
    img_base = np.zeros(shape=img_shape, dtype=np.uint8)

    # Rows
    # Rows | Generate visual
    rowBlock_count = floor(img_shape[0]/shape_rowBlock)
    rowBlock_includeSeparatorList = np.random.randint(2, size=rowBlock_count)
    rowBlocks = [generateSeparatorBlock(blockSize=shape_rowBlock, separator_size=shape_row, separator_location=separatorLocation_row, separator_include=includeSeparator, otherSize=img_shape[1]) for includeSeparator in rowBlock_includeSeparatorList]
    rowContribution = np.concatenate(rowBlocks)

    # Rows | Get separator features
    row_separator_gt = rowContribution.max(axis=1)

    # Combine
    # Combine | Image
    img = img_base + rowContribution
    img[img != 0] = 1
    img = img.astype('float32')

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

    # Show
    # test = img*255
    # cv.imshow("image", test)
    # cv.waitKey(0)
    
for i in tqdm(range(80), desc=f"Generating fake images of complexity {COMPLEXITY}"):
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