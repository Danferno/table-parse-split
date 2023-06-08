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
from joblib import Parallel, delayed
import random
from uuid import uuid4
import pytesseract

# Constants
PARALLEL = True
TEXT_THROUGH_OCR = False
PRECISION = np.float32
SAMPLE_SIZE = 300

PATH_ROOT = Path(r"F:\ml-parsing-project\table-parse-split")
PATH_DATA = PATH_ROOT / 'data'
IMAGE_FORMAT = '.jpg'

# THRESHOLD_GRAY = 255; THRESHOLD_METHOD = cv.THRESH_BINARY+cv.THRESH_OTSU      # unrestricted thresholding
TEXT_LEFT = 5
TEXT_BOTTOM_FACTOR = 4
THRESHOLD_GRAY = 100; THRESHOLD_METHOD = cv.THRESH_BINARY                        # restricted (because lots of gray in our fake images)
SAVE_ANNOTATED_TEXT = True


# COMPLEXITY = ['avg-matters']
# COMPLEXITY = ['avg-matters', 'dash-matters']
# COMPLEXITY = ['avg-matters', 'dash-matters', 'include-cols']
COMPLEXITY = ['avg-matters', 'dash-matters', 'include-cols', 'include-capital']


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
    def __init__(self, purpose, color_average, pattern, text_capital=None):
        self.purpose = purpose
        self.color_average = color_average if color_average is not None else None
        self.pattern = pattern

        if ('include-capital' in COMPLEXITY) and (text_capital is None) and (purpose == 'separator'):
            self.text_capital = True
        else:
            self.text_capital = False
    def __repr__(self) -> str:
        return f"({self.purpose[:7]}, capital {self.text_capital}, color {self.color_average}, {self.pattern})"

def getSpellLengths(inarray):
        """ run length encoding. Partial credit to R rle function. 
            Multi datatype arrays catered for including non Numpy
            returns: runlengths
            source: https://stackoverflow.com/a/32681075/7909205"""
        
        ia = np.asarray(inarray)                # force numpy
        n = len(ia)
        if n == 0: 
            return (None, None, None)
        else:
            y = ia[1:] != ia[:-1]                # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)    # must include last element posi
            z = np.diff(np.append(-1, i)) / n        # run lengths
            return z

def splitData(pathIn=pathAll, trainRatio=0.8, valRatio=0.1, imageFormat='.jpg'):
    items = [os.path.splitext(entry.name)[0] for entry in os.scandir(pathIn / 'images')]
    random.shuffle(items)

    dataSplit = {}
    dataSplit['train'], dataSplit['val'], dataSplit['test'] = np.split(items, indices_or_sections=[int(len(items)*trainRatio), int(len(items)*(trainRatio+valRatio))])

    for subgroup in dataSplit:      # subgroup = list(dataSplit.keys())[0]
        destPath = pathIn.parent / subgroup
        os.makedirs(destPath / 'images', exist_ok=True); os.makedirs(destPath / 'labels', exist_ok=True); os.makedirs(destPath / 'features', exist_ok=True); os.makedirs(destPath / 'meta', exist_ok=True)

        for item in tqdm(dataSplit[subgroup], desc=f"Copying from all > {subgroup}"):        # item = dataSplit[subgroup][0]
            _ = shutil.copyfile(src=pathIn / 'images'   / f'{item}{imageFormat}',  dst=destPath / 'images'   / f'{item}{imageFormat}')
            _ = shutil.copyfile(src=pathIn / 'labels'   / f'{item}.json', dst=destPath / 'labels'   / f'{item}.json')
            _ = shutil.copyfile(src=pathIn / 'features' / f'{item}.json', dst=destPath / 'features' / f'{item}.json')
            _ = shutil.copyfile(src=pathIn / 'meta'     / f'{item}.json', dst=destPath / 'meta'     / f'{item}.json')


# Make folders
replaceDirs(pathOut)
replaceDirs(pathAll)
replaceDirs(pathAll / 'images')
replaceDirs(pathAll / 'labels')
replaceDirs(pathAll / 'features')
replaceDirs(pathAll / 'images_text')

# Block generators
def typeToBlock(blockType, options):
    # Dimensions
    dim1 = options['separator_size'] if blockType.purpose in ['separator', 'fake-separator'] else options['block_size']
    dim2 = options['otherDim_size']

    # Purpose
    if blockType.purpose == 'no-separator':
        return None
    
    # Pattern
    if blockType.pattern == 'uniform':
        block = np.full(fill_value=blockType.color_average, shape=(dim1, dim2), dtype=np.float32)
    elif blockType.pattern == 'dash50':
        pattern = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        block_dim1 = np.tile(A=pattern, reps=dim2 // len(pattern)).astype(np.float32)
        block = np.broadcast_to(block_dim1, shape=(dim1, dim2))
    else:
        raise ValueError(f'{blockType.pattern} pattern not supported')    

    return block

def generateBlock(separatorType, contentType, options):
    # Content block
    block = typeToBlock(blockType=contentType, options=options)
    
    # Add separator block
    separator_block = typeToBlock(blockType=separatorType, options=options)
    if separator_block is not None:
        separator_start = options['separator_start']
        separator_end = separator_start + separator_block.shape[0]
        block[separator_start:separator_end] = separator_block
    
    # Text
    if options['orientation'] == 'row':
        text = 'Text' if separatorType.text_capital else 'text'
        textBlock = cv.putText(img=(block*255).astype(np.uint8), text=text, org=(TEXT_LEFT, options['block_size']//TEXT_BOTTOM_FACTOR), fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.6, color=(0,))
        block = np.round(textBlock.astype(np.float32) / 255, decimals=1)

    return block

def generateGroundTruth(separatorType, options):
    dimFull = options['block_size']
    dimSeparator = options['separator_size']

    separator_start = options['separator_start']
    separator_end = separator_start + dimSeparator

    groundTruth = np.zeros(shape=(dimFull))
    if separatorType.purpose == 'separator':
        groundTruth[separator_start:separator_end] = 1

    return groundTruth

def generateTextFeature(idx, separatorType, row_options):
    capital_or_numeric = (separatorType.text_capital)
    left = TEXT_LEFT - 2
    bottom = row_options['block_size'] // TEXT_BOTTOM_FACTOR + idx*row_options['block_size'] + 3
    right = left + 40
    top = max(0, bottom - 17)

    firstletter_capitalOrNumeric = {'xmin': left, 'xmax': right, 'ymin': top, 'ymax': bottom, 'conf': np.random.randint(low=50, high=100), 'firstletter_capitalOrNumeric': capital_or_numeric}
    return firstletter_capitalOrNumeric

# Generate fake data
def generateSample(complexity=COMPLEXITY):
    '''Complexity:
        1: Square B/W image with thick, uniform rows 
        2: + dashed borders
        3: + columns
        4: + text'''
    # Shapes
    row_blockCount = 10
    col_blockCount = 10
    size_rowBlock = 60
    size_colBlock = 80
    img_shape = (row_blockCount * size_rowBlock, col_blockCount * size_colBlock)
    
    separator_row_size = size_rowBlock // 8
    separator_row_location = (size_rowBlock - separator_row_size) // 2

    separator_col_size = size_colBlock // 12
    separator_col_location = ((size_colBlock // separator_col_size) - 2) * separator_col_size

    row_options = {
        'block_size': size_rowBlock,
        'otherDim_size': size_colBlock*col_blockCount,
        'separator_size': separator_row_size,
        'separator_start': separator_row_location,
        'orientation': 'row'
    }

    col_options = {
        'block_size': size_colBlock,
        'otherDim_size': size_rowBlock*row_blockCount,
        'separator_size': separator_col_size,
        'separator_start': separator_col_location,
        'orientation': 'col'
    }

    # Base image
    img_base = np.zeros(shape=img_shape, dtype=np.float32)
    
    # Block classes
    classes_separators_common = [Block(purpose='no-separator', color_average=None, pattern=None)]
    classes_content = []
    if 'avg-matters' in complexity:
        classes_separators_common.append(Block(purpose='separator', color_average=0, pattern='uniform'))
        classes_content.append(Block(purpose='content', color_average=1, pattern='uniform'))
    if 'dash-matters' in complexity:
        classes_separators_common.append(Block(purpose='separator', color_average=0.5, pattern='dash50'))
        classes_content.append(Block(purpose='content', color_average=0.5, pattern='uniform'))
    
    classes_separators_row = classes_separators_common.copy()
    classes_separators_col = classes_separators_common.copy()
    if 'include-capital' in complexity:
        classes_separators_row.append(Block(purpose='fake-separator', color_average=0, pattern='uniform', text_capital=False))
    

    # Rows
    # Rows | Generate visual
    row_separatorTypes = np.random.choice(classes_separators_row, size=row_blockCount, replace=True)
    row_contentTypes = np.random.choice(classes_content, size=row_blockCount, replace=True)
    rowBlocks = [generateBlock(separatorType=row_separatorTypes[i], contentType=row_contentTypes[i], options=row_options) for i in range(row_blockCount)]        # i = 1
    rowContribution = np.concatenate(rowBlocks)

    # Rows | Get separator features
    row_separator_gt = np.concatenate([generateGroundTruth(separatorType=row_separatorTypes[i], options=row_options) for i in range(row_blockCount)])

    # Cols
    if 'include-cols' in COMPLEXITY:   
        # Cols | Generate visual
        col_separatorTypes = np.random.choice(classes_separators_col, size=col_blockCount, replace=True)
        col_contentTypes = np.random.choice(classes_content, size=col_blockCount, replace=True)
        colBlocks = [generateBlock(separatorType=col_separatorTypes[i], contentType=col_contentTypes[i], options=col_options) for i in range(col_blockCount)]        # i = 1
        colContribution = np.concatenate(colBlocks).T

        # Cols | Get separator features
        col_separator_gt = np.concatenate([generateGroundTruth(separatorType=col_separatorTypes[i], options=col_options) for i in range(col_blockCount)])
    else:
        colContribution = np.zeros_like(rowContribution)
        col_separator_gt = np.zeros(shape=(img_shape[1],))

    # Combine
    # Combine | Image
    img = img_base + rowContribution
    img = np.minimum(img, colContribution)
    img = np.clip(img, a_min=0, a_max=1)
    _, img_cv = cv.threshold((img*255).astype(np.uint8), THRESHOLD_GRAY, 255, THRESHOLD_METHOD)
    
    name = str(uuid4())[:16]
    img = img_cv.astype(PRECISION)/255

    # t = (img*255).astype(np.uint8)
    # cv.imshow('orig', )
    #cv.imshow('thresh', img_cv)

    # Combine | Ground Truth
    gt = {}
    gt['row'] = row_separator_gt
    gt['col'] = col_separator_gt

    # Extract features
    # Features | Visual
    features = {}
    features['row_absDiff'] = (np.asarray([np.absolute(np.diff(row)).mean() for row in img]))
    features['row_avg'] = np.asarray([np.mean(row) for row in img])
    
    features['col_absDiff'] =   np.asarray([np.absolute(np.diff(col)).mean() for col in img.T])
    features['col_avg'] = np.asarray([np.mean(col) for col in img.T])

    row_spell_lengths = [getSpellLengths(row) for row in img]
    features['row_spell_mean'] = [np.mean(row_spell_length) for row_spell_length in row_spell_lengths]
    features['row_spell_sd'] = [np.std(row_spell_length) for row_spell_length in row_spell_lengths]

    col_spell_lengths = [getSpellLengths(col) for col in img.T]
    features['col_spell_mean'] = [np.mean(col_spell_length) for col_spell_length in col_spell_lengths]
    features['col_spell_sd'] = [np.std(col_spell_length) for col_spell_length in col_spell_lengths]

    # Features | Text | Find text
    if TEXT_THROUGH_OCR:
        # Features | Text | Prepare image for OCR
        img_ocr = img_cv.copy()
    
        # Features | Text | OCR
        df = pytesseract.image_to_data(image=img_ocr, lang='eng', config=r'--psm 4 --oem 3', output_type=pytesseract.Output.DATAFRAME)
        df = df.loc[(df.level == 5) & (df.conf > 40), ['left', 'top', 'width', 'height', 'conf', 'text']]
        if len(df):
            df['text'] = df['text'].str.replace('[^a-zA-Z0-9]', '', regex=True)
            df = df.loc[df.text.str.len() >= 2]
            df['right'] = df['left'] + df['width']
            df['bottom'] = df['top'] + df['height']
            df['conf'] = df['conf'].astype(int)
            df['firstletter_capital_or_number'] = (df['text'].str[0].str.isupper() | df['text'].str[0].str.isdigit())*1

            # Features | Text | Convert to feature
            df = df.astype({col: int for col in ['left', 'right', 'bottom', 'top', 'capital_or_number']})
            firstletter_capitalOrNumeric = df[['left', 'top', 'right', 'bottom', 'conf', 'firstletter_capital_or_number']]
        else:
            raise Exception('No text found')
    else:
        firstletter_capitalOrNumeric = pd.DataFrame.from_records([generateTextFeature(idx, separatorType, row_options=row_options) for idx, separatorType in enumerate(row_separatorTypes)])
  
    # Features | Text | Assign capital/number status to each row [TD: take max if multiple rows]
    df = firstletter_capitalOrNumeric.copy()
    df = df.loc[(df.conf > 0.5)]
    df = df[['ymin', 'firstletter_capitalOrNumeric']]
    df = df.set_index('ymin').reindex(range(0, img.shape[0]))
    df['firstletter_capitalOrNumeric'] = df['firstletter_capitalOrNumeric'].fillna(method='ffill')
    df['firstletter_capitalOrNumeric'] = df['firstletter_capitalOrNumeric'].fillna(value=False)
    df.to_csv('temp.csv')
    row_firstletter_capitalOrNumeric = df['firstletter_capitalOrNumeric'].astype(int).to_numpy().astype(np.uint8)
    features['row_firstletter_capitalOrNumeric'] = row_firstletter_capitalOrNumeric

    if SAVE_ANNOTATED_TEXT:
        # Get img
        img_annot = img_cv.copy()

        # Add text boxes
        for idx, row in firstletter_capitalOrNumeric.iterrows():
            color = 160 if row['firstletter_capitalOrNumeric'] else 40
            cv.rectangle(img=img_annot, pt1=(row['xmin'], row['ymin']), pt2=(row['xmax'], row['ymax']), color=color, thickness=2)

        # Add row-level indicators
        indicator_capitalOrNumeric = np.expand_dims(row_firstletter_capitalOrNumeric, 1)
        indicator_capitalOrNumeric = np.broadcast_to(indicator_capitalOrNumeric, shape=[indicator_capitalOrNumeric.shape[0], 40])
        indicator_capitalOrNumeric = indicator_capitalOrNumeric * 200
        img_annot = np.concatenate([img_annot, indicator_capitalOrNumeric], axis=1)
        pathImage_annotated = str(pathAll /'images_text' / f'{name}.jpg')
        cv.imwrite(filename=pathImage_annotated, img=img_annot)

    # Save
    imagePath = str(pathAll /'images' / f'{name}.jpg')
    cv.imwrite(filename=imagePath, img=img_cv)

    with open(pathAll / 'labels' / f'{name}.json', 'w') as groundTruthFile:
        json.dump(gt, groundTruthFile, cls=NumpyEncoder)
    with open(pathAll / 'features' / f'{name}.json', 'w') as featureFile:
        json.dump(features, featureFile, cls=NumpyEncoder)

if __name__ == '__main__':
    if PARALLEL:
        _ = Parallel(n_jobs=8, backend='loky', verbose=6)(delayed(generateSample)() for i in range(SAMPLE_SIZE))
    else:
        for i in tqdm(range(SAMPLE_SIZE), desc=f"Generating fake images of complexity {COMPLEXITY}"):
            generateSample()
    # Split into train/val/test
    splitData()



