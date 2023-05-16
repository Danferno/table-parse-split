# Imports
import os
from pathlib import Path
from lxml import etree
from math import floor, ceil
from copy import deepcopy
from tqdm import tqdm
import tabledetect
import shutil
import numpy as np
import cv2 as cv
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
pathOut_images = pathOut / 'images'
pathOut_labels = pathOut / 'labels'
os.makedirs(pathOut, exist_ok=True)
os.makedirs(pathOut_images, exist_ok=True)
os.makedirs(pathOut_labels, exist_ok=True)

# Generate fake data
def generateSeparatorBlock(blockSize, separator_size, separator_location, separator_include, otherSize, complexity=COMPLEXITY):
    # Base block
    block = np.zeros((blockSize, otherSize), dtype=np.uint8)
    
    # Add separator
    if separator_include:
        block[separator_location:separator_location+separator_size, :] = 1
    
    return block

def generateImage(complexity=COMPLEXITY):
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

    # Combine | Ground Truth
    gt = {'row': row_separator_gt.tolist()}

    # Extract features
    

    # Show
    # test = img*255
    # cv.imshow("image", test)
    # cv.waitKey(0)

    # Save
    name = str(uuid4())
    cv.imwrite(filename=str(pathOut_images / f'{name}.jpg'), img=img*255)

    with open(pathOut_labels / f'{name}.json', 'w') as groundTruthFile:
        json.dump(gt, groundTruthFile)
    





    