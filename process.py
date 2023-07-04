# TD: seems to drop last column?

# Imports
import torch    # type : ignore
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
import easyocr
import fitz     # type : ignore
import cv2 as cv
from tqdm import tqdm
import json
import os
from lxml import etree
import logging

from collections import namedtuple, Counter
from PIL import Image, ImageFont, ImageDraw

import utils
from model import TableLineModel, TableSeparatorModel
import dataloaders
from joblib import Parallel, delayed

# Constants
COLOR_CELL = (102, 153, 255, int(0.05*255))      # light blue
COLOR_OUTLINE = (255, 255, 255, int(0.6*255))
OPACITY_ORIGINAL = int(0.15*255)

# Helper functions
def preds_to_separators(predArray, threshold=0.8, setToMidpoint=False, addPaddingSeparators=True, paddingSeparator=None):
    ''' Left: first one (= inclusive), right: first zero (= exclusive)'''
    # Tensor > np.array on cpu
    if isinstance(predArray, torch.Tensor):
        predArray = predArray.cpu().numpy().squeeze()
        
    is_separator = (predArray > threshold)
    diff_in_separator_modus = np.diff(is_separator.astype(np.int8))
    separators_start = np.where(diff_in_separator_modus == 1)[0]
    separators_end = np.where(diff_in_separator_modus == -1)[0]
    separators = np.stack([separators_start, separators_end], axis=1)

    # Optionally add padding separators
    if addPaddingSeparators:
        separators = np.concatenate([paddingSeparator, separators], axis=0)
    
    # Convert wide separator to midpoint
    if setToMidpoint:
        separator_means = np.floor(separators.mean(axis=1)).astype(np.int32)
        separators = np.stack([separator_means, separator_means+1], axis=1)

    return separators
def parse_separators(separatorArray:list, padding, size, setToMidpoint=True):
    try:   
        # Set to midpoint
        if setToMidpoint:
            separator_means = np.floor(separatorArray.mean(axis=1)).astype(np.int32)
            separatorArray = np.stack([separator_means, separator_means+1], axis=1)
        
        # Add padding separator
        separatorArray = np.concatenate((np.array([[padding-1, padding]]), separatorArray, np.array([[size-padding, size-padding+1]])), axis=0)

    except (np.AxisError, ValueError):
        separatorArray = np.concatenate((np.array([[padding-1, padding]]), np.array([[size-padding, size-padding+1]])), axis=0)

    return separatorArray
def get_first_non_null_values(df):
    # Get first value
    header_candidates = df.iloc[:5].fillna(method='bfill', axis=0).iloc[:1].copy()

    # Replace empty values by 'empty'
    header_candidates = header_candidates.fillna('empty')

    # Replace numeric values by "numeric"
    numericCols = [col for col in header_candidates.columns if header_candidates[col].replace(',', '', regex=True).str.isnumeric().all()]
    header_candidates.loc[:, numericCols] = 'numeric'

    # Get values    
    header_candidates = header_candidates.values.squeeze()
    if header_candidates.shape == ():
        header_candidates = np.expand_dims(header_candidates, 0)
    return header_candidates
def number_duplicates(l):
    counter = Counter()

    for v in l:
        counter[v] += 1
        if counter[v]>1:
            yield v+f'_{counter[v]}'
        else:
            yield v
def boxes_intersect(box, box_target):
    overlap_x = ((box['x0'] >= box_target['x0']) & (box['x0'] < box_target['x1'])) | ((box['x1'] >= box_target['x0']) & (box['x0'] < box_target['x0']))
    overlap_y = ((box['y0'] >= box_target['y0']) & (box['y0'] < box_target['y1'])) | ((box['y1'] >= box_target['y0']) & (box['y0'] < box_target['y0']))

    return all([overlap_x, overlap_y])
def scale_cell_to_dpi(cell, dpi_start, dpi_target):
    for key in ['x0', 'x1', 'y0', 'y1']:
        cell[key] = int((cell[key]*dpi_target/dpi_start).round(0))
    return cell
def arrayToXml(array:list, xmlRoot:etree.Element, label:str, orientation:str, tableBbox:np.array):
    for start, end in array:
        xml_obj = etree.SubElement(xmlRoot, 'object')
        xml_label = etree.SubElement(xml_obj, 'name'); xml_label.text = label

        xml_bbox = etree.SubElement(xml_obj, 'bndbox')
        ymin = etree.SubElement(xml_bbox, 'ymin'); ymin.text = str(start)     if orientation == 'row' else tableBbox['ymin']
        ymax = etree.SubElement(xml_bbox, 'ymax'); ymax.text = str(end)       if orientation == 'row' else tableBbox['ymax']
        xmin = etree.SubElement(xml_bbox, 'xmin'); xmin.text = str(start)     if orientation == 'col' else tableBbox['xmin']
        xmax = etree.SubElement(xml_bbox, 'xmax'); xmax.text = str(end)       if orientation == 'col' else tableBbox['xmax']
def imgAndWords_to_features(img, textDf_table:pd.DataFrame, precision=np.float32, debug=False):
    # Prepare image
    _, img_cv = cv.threshold(np.array(img), 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    img01 = img_cv.astype(precision)/255

    # Features
    features = {}

    # Features | Visual
    features['row_absDiff'] = (np.asarray([np.absolute(np.diff(row)).mean() for row in img01]))
    features['row_avg'] = np.asarray([np.mean(row) for row in img01])  
    
    features['col_absDiff'] =   np.asarray([np.absolute(np.diff(col)).mean() for col in img01.T])
    features['col_avg'] = np.asarray([np.mean(col) for col in img01.T])

    row_spell_lengths = [utils.getSpellLengths(row) for row in img01]
    features['row_spell_mean'] = [np.mean(row_spell_length) for row_spell_length in row_spell_lengths]
    features['row_spell_sd'] = [np.std(row_spell_length) for row_spell_length in row_spell_lengths]

    col_spell_lengths = [utils.getSpellLengths(col) for col in img01.T]
    features['col_spell_mean'] = [np.mean(col_spell_length) for col_spell_length in col_spell_lengths]
    features['col_spell_sd'] = [np.std(col_spell_length) for col_spell_length in col_spell_lengths]

    features['global_rowAvg_p0'], features['global_rowAvg_p5'], features['global_rowAvg_p10'] = np.percentile(features['row_avg'], q=[0, 5, 10])
    features['global_colAvg_p0'], features['global_colAvg_p5'], features['global_colAvg_p10'] = np.percentile(features['col_avg'], q=[0, 5, 10])

    # Features | Text       
    # Features | Text | Text like start
    # Features | Text | Text like start | Identify words with text like start of row/col
    textDf = textDf_table.sort_values(by=['blockno', 'lineno', 'wordno']).drop(columns=['wordno', 'ocrLabel', 'conf']).reset_index(drop=True)
    textDf['lineno_seq'] = (textDf['blockno']*1000 + textDf['lineno']).rank(method='dense').astype(int)
    textDf['text_like_start_row'] = ((textDf['text'].str[0].str.isupper()) | (textDf['text'].str[0].str.isdigit()) | (textDf['text'].str[:5].str.contains(r'[\.\)\-]', regex=True)) )*1
    
    # Features | Text | Text like rowstart
    # Features | Text | Text like rowstart | Identify rows with text like start
    textDf_line = textDf.drop_duplicates(subset=['blockno', 'lineno'], keep='first').drop(columns=['blockno', 'left', 'right'])
    textDf_line = textDf_line.drop_duplicates(subset='top', keep='first').reset_index(drop=True)
    
    # Features | Text | Text like rowstart | Gather info on next line
    textDf_line['F1_text_like_start_row'] = textDf_line['text_like_start_row'].shift(periods=-1).fillna(1).astype(int)
    textDf_line = textDf_line.sort_values(by=['top', 'bottom', 'lineno', 'lineno_seq']).astype({'lineno_seq': int})

    # Features | Text | Text like rowstart | Assign lines to img rows
    textDf_row = pd.DataFrame(data=np.arange(img01.shape[0], dtype=np.int32), columns=['top'])
    textDf_row = pd.merge_asof(left=textDf_row, right=textDf_line[['top', 'bottom', 'lineno', 'lineno_seq']], left_on='top', right_on='top', direction='backward')
    textDf_row['lineno_seq'] = textDf_row['lineno_seq'].fillna(value=0)                                                     # Before first text: 0
    textDf_row.loc[textDf_row['top'] > textDf_row['bottom'].max(), 'lineno_seq'] = textDf_row['lineno_seq'].max() + 1       # After  last  text: max + 1
    lastLineNo = textDf_row['lineno_seq'].max()

    # Features | Text | Text like rowstart | Identify 'img rows' between text lines
    textDf_row['between_textlines'] = textDf_row['top'] > textDf_row['bottom']
    textDf_row.loc[textDf_row['lineno_seq'] == lastLineNo, 'between_textlines'] = False
    
    # Features | Text | Text like rowstart | Identify 'img rows' between text lines | For the start and end lines, set everything that isn't padding to separator (edge case of all padding: 1/5th set to separator)
    height_line0 = textDf_row.loc[textDf_row['lineno_seq'] == 0].index.max()
    try:
        firstNonWhite_line0 = np.where(features['row_avg'][textDf_row.loc[textDf_row['lineno_seq'] == 0].index] < 1)[0][0]
    except IndexError:
        firstNonWhite_line0 = height_line0
    countNonWhite_line0 = height_line0 - firstNonWhite_line0

    if countNonWhite_line0:
        textDf_row.loc[(textDf_row.index >= firstNonWhite_line0) & (textDf_row['lineno_seq'] == 0), 'between_textlines'] = True
    else:
        textDf_row.loc[(textDf_row.index >= (height_line0 // 5 * 4)) & (textDf_row['lineno_seq'] == 0), 'between_textlines'] = True

    try:
        lastNonWhite_lineLast = np.where(np.flip(features['row_avg'][textDf_row.loc[textDf_row['lineno_seq'] == textDf_row['lineno_seq'].max()].index]) < 1)[0][0]       # distance from end
    except IndexError:
        height_lineLast = textDf_row.loc[textDf_row['lineno_seq'] == lastLineNo, 'lineno_seq'].count()
        lastNonWhite_lineLast = height_lineLast // 5
    textDf_row.loc[(textDf_row.index <= len(textDf_row.index)-lastNonWhite_lineLast) & (textDf_row['lineno_seq'] == textDf_row['lineno_seq'].max()), 'between_textlines'] = True

    # Features | Text | Text like rowstart | Identify 'img rows' between text lines | Assign half of separator to neighbouring lines (so we can verify that all lines contain separators)
    lastRow_line0 = textDf_row.loc[textDf_row['lineno_seq'] == 0].index.max()
    middleOfSeparator_line0 = firstNonWhite_line0 + (lastRow_line0 - firstNonWhite_line0) // 2
    textDf_row.loc[middleOfSeparator_line0:lastRow_line0+1, 'lineno_seq'] = 1

    firstRow_lineLast = textDf_row.loc[textDf_row['lineno_seq'] == lastLineNo].index.min()
    separatorRowCount_lineLast = textDf_row.loc[(textDf_row['lineno_seq'] == lastLineNo), 'between_textlines'].sum()
    middleOfSeparator_lineLast = firstRow_lineLast + (separatorRowCount_lineLast // 2)
    textDf_row.loc[firstRow_lineLast:middleOfSeparator_lineLast+1, 'lineno_seq'] = lastLineNo - 1
    
    textDf_row.loc[middleOfSeparator_line0:lastRow_line0+1, 'lineno_seq'] = 1  
    textDf_row['lineno_seq'] = textDf_row['lineno_seq'].astype(int)
    
    if debug:
        betweenText_max_per_line = textDf_row.groupby('lineno_seq')['between_textlines'].transform(max)
        if betweenText_max_per_line.min() == False:
            raise Exception('Not all lines contain separators')

    # Features | Text | Text like rowstart | Merge textline-level text_like_start_row info
    textDf_row = textDf_row.merge(right=textDf_line[['lineno_seq', 'text_like_start_row', 'F1_text_like_start_row']], left_on='lineno_seq', right_on='lineno_seq', how='outer').fillna({'F1_text_like_start_row': 1}).astype({'F1_text_like_start_row':int}).dropna(axis='index', subset='between_textlines')
    textDf_row['between_textlines_like_rowstart'] = textDf_row['between_textlines'] & textDf_row['F1_text_like_start_row']

    # Features | Text | Text like rowstart | Add to features
    features['row_between_textlines']               = textDf_row['between_textlines'].to_numpy().astype(np.uint8)
    features['row_between_textlines_like_rowstart'] = textDf_row['between_textlines_like_rowstart'].to_numpy().astype(np.uint8)

    # Features | Text | Text like colstart
    # Features | Text | Text like colstart | Count for each column how often the nearest row to its right contains start-like text
    def convert_bbox_to_array_value(row, target_array):
        '''Fills in elements of bbox in target_array with 255 if bbox contains text_like_start_col, otherwise fills in 128'''
        if row['text_like_start_col']:
            fill = 255
        else:
            fill = 128

        top, bottom, left, right = row['top'], row['bottom'], row['left'], row['right']
        target_array[top:bottom+1, left:right+1] = fill

    def count_nearest_right_values(in_array, values_to_count):
        counts = {value_to_count: np.zeros(shape=in_array.shape[1]) for value_to_count in values_to_count}

        for colNumber in range(in_array.shape[1]):        # colNumber = 0
            right_of_col_array = np.ndarray.view(in_array)[:, colNumber+1:]                            # Reduce array to elements to the right of the col
            try:
                # (maybe) TD: only include values at the x-th percentile of the indexes
                firstnonzero_in_row_index = (right_of_col_array != 0).argmax(axis=1)                                                # Get index of first non-zero element per row
                firstnonzero_in_row_value = right_of_col_array[np.arange(right_of_col_array.shape[0]), firstnonzero_in_row_index]   # Get value of first non-zero element per row
            except ValueError:
                firstnonzero_in_row_value = np.zeros(shape=in_array.shape[0])                                                       # Handle case of last col (no values to the right) > always zero
            for value_to_count in values_to_count:
                counts[value_to_count][colNumber] = (firstnonzero_in_row_value == value_to_count).sum()                             # Count occurence of each relevant value

        return counts

    text_like_start_array = np.zeros_like(img01, dtype=np.uint32)
    textDf['text_like_start_col'] = ((textDf['text'].str[0].str.isupper()) | (textDf['text'].str[0].str.isdigit()) | ((textDf['text'].str[0].isin(['.', '(', ')', '-'])) & (textDf['text'].str.len() > 1)) )*1
    textDf.apply(lambda row: convert_bbox_to_array_value(row=row, target_array=text_like_start_array), axis=1)     # Modifies text_like_start_array
    counts = count_nearest_right_values(text_like_start_array, values_to_count=[128, 255])

    nearest_right_is_text = (counts[128]+counts[255]).astype(np.uint32)
    nearest_right_is_text[nearest_right_is_text == 0] = 1
    nearest_right_is_startlike_share = counts[255]/nearest_right_is_text
    features['col_nearest_right_is_startlike_share'] = nearest_right_is_startlike_share    

    # Features | Text | Words crossed per row/col
    textDf_WC = textDf.copy().sort_values(by=['left', 'top', 'right', 'bottom'])
    left, right, top, bottom = (textDf_WC[dim].values for dim in ['left', 'right', 'top', 'bottom'])
    
    features['row_wordsCrossed_count'] = np.array([utils.index_to_bboxCross(index=index, mins=top, maxes=bottom) for index in range(img01.shape[0])])
    features['col_wordsCrossed_count'] = np.array([utils.index_to_bboxCross(index=index, mins=left, maxes=right) for index in range(img01.shape[1])])

    features['row_wordsCrossed_relToMax'] = features['row_wordsCrossed_count'] / features['row_wordsCrossed_count'].max() if features['row_wordsCrossed_count'].max() != 0 else features['row_wordsCrossed_count']
    features['col_wordsCrossed_relToMax'] = features['col_wordsCrossed_count'] / features['col_wordsCrossed_count'].max() if features['col_wordsCrossed_count'].max() != 0 else features['col_wordsCrossed_count']

    # Features | Text | Outside text rectangle
    if len(textDf):
        textRectangle = {}
        textRectangle['left'] = textDf['left'].min()
        textRectangle['top'] = textDf['top'].min()
        textRectangle['right'] = textDf['right'].max()
        textRectangle['bottom'] = textDf['bottom'].max()

        row_in_textrectangle = np.zeros(shape=img01.shape[0], dtype=np.uint8); row_in_textrectangle[textRectangle['top']:textRectangle['bottom']+1] = 1
        col_in_textrectangle = np.zeros(shape=img01.shape[1], dtype=np.uint8); col_in_textrectangle[textRectangle['left']:textRectangle['right']+1] = 1
    
    else:
        row_in_textrectangle = np.ones(shape=img01.shape[0], dtype=np.uint8)
        col_in_textrectangle = np.ones(shape=img01.shape[1], dtype=np.uint8);

    features['row_in_textrectangle'] = row_in_textrectangle
    features['col_in_textrectangle'] = col_in_textrectangle

    # Return
    return img01, img_cv, features
def vocLabels_to_groundTruth(pathTargets, img, features, adjust_labels_to_textboxes=False, add_edge_separators=False):
    # Parse textbox-information
    if adjust_labels_to_textboxes:
        row_between_textlines = features['row_between_textlines']
        col_wordsCrossed_relToMax = features['col_wordsCrossed_relToMax']
        # Row
        textline_boundaries_horizontal = np.diff(row_between_textlines.astype(np.int8), append=0)
        if textline_boundaries_horizontal.max() < 1:
            textline_boundaries_horizontal[-1] = 0
        textline_boundaries_horizontal = np.column_stack([np.where(textline_boundaries_horizontal == 1)[0], np.where(textline_boundaries_horizontal == -1)[0]])

        # Column
        textline_boundaries_vertical = np.diff((col_wordsCrossed_relToMax < 0.001).astype(np.int8), append=0)
        if textline_boundaries_vertical.max() < 1:
            textline_boundaries_vertical[-1] = 0
        else:
            textline_boundaries_vertical[0] = 1
        textline_boundaries_vertical = np.column_stack([np.where(textline_boundaries_vertical == 1)[0], np.where(textline_boundaries_vertical == -1)[0]])      

    # Parse xml
    root = etree.parse(pathTargets)
    objectCount = len(root.findall('.//object'))
    if objectCount:
        rows = root.findall('object[name="row separator"]')
        cols = root.findall('object[name="column separator"]')
        spanners = root.findall('object[name="spanning cell interior"]')

        # Get separator locations
        row_separators = [(int(row.find('.//ymin').text), int(row.find('.//ymax').text)) for row in rows]
        row_separators = sorted(row_separators, key= lambda x: x[0])

        if (adjust_labels_to_textboxes) and (len(textline_boundaries_horizontal) > 0):
            row_separators = utils.adjust_initialBoundaries_to_betterBoundariesB(arrayInitial=row_separators, arrayBetter=textline_boundaries_horizontal)

        col_separators = [(int(col.find('.//xmin').text), int(col.find('.//xmax').text)) for col in cols]
        col_separators = sorted(col_separators, key= lambda x: x[0])
        if (adjust_labels_to_textboxes) and (len(textline_boundaries_vertical) > 0):
            col_separators = utils.adjust_initialBoundaries_to_betterBoundariesB(arrayInitial=col_separators, arrayBetter=textline_boundaries_vertical)

        # Optionally add edge borders (top, left, right , bottom)
        #   Excluding these can confuse the model, as they really look like borders
        if add_edge_separators:
            # Row
            if len(row_separators) > 0:
                first_nonwhite = np.where(features['row_avg'] < 1)[0][0]
                try:
                    first_withtext = np.where(features['row_wordsCrossed_count'] > 0)[0][0]
                except IndexError:
                    first_withtext = 0
                if (first_nonwhite < first_withtext) and (first_withtext - first_nonwhite > 2) and (first_withtext < row_separators[0][0]):
                    top_separator = (first_nonwhite, first_withtext-1)
                    row_separators.insert(0, top_separator)

                last_nonwhite = np.where(features['row_avg'] < 1)[0][-1]
                try:
                    last_withtext = np.where(features['row_wordsCrossed_count'] > 0)[0][-1]
                except IndexError:
                    last_withtext = features['row_wordsCrossed_count'].size
                if (last_nonwhite > last_withtext) and (last_nonwhite - last_withtext > 2) and (last_withtext > row_separators[-1][0]):
                    bot_separator = (last_withtext+1, last_nonwhite)
                    row_separators.append(bot_separator)

            # Column
            if len(col_separators) > 0:
                first_nonwhite = np.where(features['col_avg'] < 1)[0][0]
                try:
                    first_withtext = np.where(features['col_wordsCrossed_count'] > 0)[0][0]
                except:
                    first_withtext = 0
                if (first_nonwhite < first_withtext) and (first_withtext - first_nonwhite > 2) and (first_withtext < col_separators[0][0]):
                    left_separator = (first_nonwhite, first_withtext-1)
                    col_separators.insert(0, left_separator)

                last_nonwhite = np.where(features['col_avg'] < 1)[0][-1]
                try:
                    last_withtext = np.where(features['col_wordsCrossed_count'] > 0)[0][-1]
                except:
                    last_withtext = features['col_wordsCrossed_count'].size
                if (last_nonwhite > last_withtext) and (last_nonwhite - last_withtext > 2) and (last_withtext > col_separators[-1][0]):
                    right_separator = (last_withtext+1, last_nonwhite)
                    col_separators.append(right_separator)

        # Create ground truth arrays
        gt_row = np.zeros(shape=img.shape[0], dtype=np.uint8)
        for separator in row_separators:        # separator = row_separators[0]
            gt_row[separator[0]:separator[1]+1] = 1
        
        gt_col = np.zeros(shape=img.shape[1], dtype=np.uint8)
        for separator in col_separators:        # separator = row_separators[0]
            gt_col[separator[0]:separator[1]+1] = 1
    else:
        gt_row = np.zeros(shape=img.shape[0], dtype=np.uint8)
        gt_col = np.zeros(shape=img.shape[1], dtype=np.uint8)
    
    # Adjust to textboxes
    gt = {}
    gt['row'] = gt_row
    gt['col'] = gt_col
    return gt


# Function
def detect_from_pdf(path_data, path_out, path_model_file_detect=None, device='cuda', replace_dirs='warn', draw_images=False, padding=40):
    ''' Detect tables'''
    ...

# Line-level | Preprocess | Single PDF
def preprocess_lineLevel_singlePdf(pdfNameAndPage, path_out, path_out_features, path_skew=None, path_out_targets=None, path_annotations=None, path_out_images_annotated=None, path_words=None, path_pdfs=None, path_images=None, path_bboxes=None, replace_dirs='warn', image_format='.png',
                                           split_stub_page='-p', split_stub_table='_t', dpi_pymupdf=72, dpi_model=150, dpi_ocr=300, padding=40, draw_images=True,
                                            ground_truth=False, adjust_targets_to_textboxes=False, add_edge_separators=False, path_images_annotators=None, path_skew_annotators=None, disable_progressbar=False):
    # Parameters
    path_out = Path(path_out)
    path_images = path_images or path_out / 'tables_images'
    path_bboxes = path_bboxes or path_out / 'tables_bboxes'
    path_pdfs = path_pdfs or path_out / 'pdfs'
    path_words = path_words or path_out / 'words'
    path_annotations = path_annotations or path_out / 'annotations'
    path_skew = path_skew or path_out / 'meta' / 'skewAngles'

    path_out_features = path_out_features or path_out / 'features_lineLevel'
    path_out_targets = path_out_targets or path_out / 'targets_lineLevel'
    path_out_meta_lineLevel = path_out / 'meta_lineLevel'
    if ground_truth:
        path_out_images_annotated = path_out_images_annotated or path_out / 'tables_annotated_featuresAndTargets'
    else:
        path_out_images_annotated = path_out_images_annotated or path_out / 'tables_annotated_features'
    

    # Parse dict
    pdfName, pageNumbers = pdfNameAndPage

    # Open pdf
    pdfPath = path_pdfs / f'{pdfName}.pdf'
    doc:fitz.Document = fitz.open(pdfPath)

    # Features singlepdf | Loop over pages
    for pageNumber in tqdm(pageNumbers, position=1, leave=False, desc='Words | Looping over pages', total=len(pageNumbers), disable=disable_progressbar):        # pageNumber = pageNumbers[0]
        page:fitz.Page = doc.load_page(page_id=pageNumber)
        pageName = f'{pdfName}{split_stub_page}{pageNumber}'
        
        # Page | Table bboxes
        page_mediabox_size = page.mediabox_size if page.rotation in [0, 180] else fitz.Point(page.mediabox_size[1], page.mediabox_size[0])
        fitzBoxes = utils.yolo_to_fitzBox(yoloPath=path_bboxes / f'{pageName}.txt' , targetPdfSize=page_mediabox_size)

        # Page | Angle
        try:
            with open(path_skew / f'{pageName}.txt', 'r') as f:
                angle = float(f.read())
        except FileNotFoundError:
            angle = 0

        # Page | Text
        textDf_page = pd.read_parquet(path_words / f'{pageName}.pq')
        textDf_page.loc[:, ['left', 'right', 'top', 'bottom']] = textDf_page.loc[:, ['left', 'right', 'top', 'bottom']] * (dpi_pymupdf / dpi_ocr)

        # Page | Loop over tables
        for tableIteration, fitzBox in enumerate(fitzBoxes):      # tableIteration = 0; fitzBox = fitzBoxes[tableIteration]
            # Table
            tableName = f'{pageName}{split_stub_table}{tableIteration}'
            tableRect = fitz.Rect([round(coord) for coord in fitzBox])

            # Table | Img
            img = Image.open(path_images / f'{tableName}{image_format}')

            # Table | Ensure annotator saw same image (table numbering was inconsistent in the past)
            if ground_truth:
                img_tight = page.get_pixmap(dpi=dpi_model, clip=tableRect, alpha=False, colorspace=fitz.csGRAY)
                img_tight = Image.frombytes(mode='L', size=(img_tight.width, img_tight.height), data=img_tight.samples)
                img = Image.new(img_tight.mode, (img_tight.width+padding*2, img_tight.height+padding*2), 255)
                img.paste(img_tight, (padding, padding))
                img, angle = utils.deskew_img_from_file(pageName=pageName, img=img, path_skewfiles=path_skew_annotators)            

                # Ensure nothing went wrong in table numbering (not always consistent)
                img_sent_to_annotator = Image.open(path_images_annotators / f'{tableName}.jpg').convert(mode='L')
                similarity_index = utils.calculate_image_similarity(img1=img, img2=img_sent_to_annotator)
                skip_table = False
                if similarity_index < 0.85:
                    # Retry with other tables
                    similarity_indices = {}
                    for tableIteration, _ in enumerate(fitzBoxes):      # tableIteration = next(enumerate(fitzBoxes))[0]
                        tableRect_temp = fitz.Rect(fitzBoxes[tableIteration]['xy'])
                        img_tight_temp = page.get_pixmap(dpi=dpi_model, clip=tableRect_temp, alpha=False, colorspace=fitz.csGRAY)
                        img_tight_temp = Image.frombytes(mode='L', size=(img_tight_temp.width, img_tight_temp.height), data=img_tight_temp.samples)
                        img_temp = Image.new(img_tight_temp.mode, (img_tight_temp.width+padding*2, img_tight_temp.height+padding*2), 255)
                        img_temp.paste(img_tight_temp, (padding, padding))
                        img_temp, angle_temp = utils.deskew_img_from_file(pageName=pageName, img=img_temp, path_skewfiles=path_skew_annotators)            
                        similarity_index = utils.calculate_image_similarity(img1=img_temp, img2=img_sent_to_annotator)
                        similarity_indices[tableIteration] = similarity_index

                    most_similar_tableIteration = max(similarity_indices, key=similarity_indices.get)
                    if max(similarity_indices.values()) < 0.85:
                        errors += 1
                        skip_table = True

                    # Adapt image to most similar (note, do not update tablename!)
                    tableRect = fitz.Rect(fitzBoxes[most_similar_tableIteration]['xy'])
                    img_tight = page.get_pixmap(dpi=dpi_model, clip=tableRect, alpha=False, colorspace=fitz.csGRAY)
                    img_tight = Image.frombytes(mode='L', size=(img_tight.width, img_tight.height), data=img_tight.samples)
                    img = Image.new(img_tight.mode, (img_tight.width+padding*2, img_tight.height+padding*2), 255)
                    img.paste(img_tight, (padding, padding))             
                    img, angle = utils.deskew_img_from_file(pageName=pageName, img=img, path_skewfiles=path_skew_annotators)            

                if skip_table:
                    continue

            # Table | Text
            textDf_table = textDf_page.loc[(textDf_page['top'] >= tableRect.y0) & (textDf_page['left'] >= tableRect.x0) & (textDf_page['bottom'] <= tableRect.y1) & (textDf_page['right'] <= tableRect.x1)]        # clip to table

            # Table | Text | Convert to padded table image pixel coordinates
            textDf_table.loc[:, ['left', 'right']] = textDf_table[['left', 'right']] - tableRect.x0
            textDf_table.loc[:, ['top', 'bottom']] = textDf_table[['top', 'bottom']] - tableRect.y0
            textDf_table.loc[:, ['left', 'right', 'top', 'bottom']] = textDf_table[['left', 'right', 'top', 'bottom']] * (dpi_model / dpi_pymupdf) + padding
            textDf_table.loc[:, ['left', 'right', 'top', 'bottom']] = textDf_table[['left', 'right', 'top', 'bottom']].round(0)
            textDf_table = textDf_table.astype({edge: int for edge in ['left', 'right', 'top', 'bottom']})
            if 'conf' not in textDf_table.columns:
                textDf_table['conf'] = 100

            # Table | Generate features
            img01, img_cv, features = imgAndWords_to_features(img=img, textDf_table=textDf_table)
            with open(path_out_features / f'{tableName}.json', 'w') as f:
                json.dump(features, f, cls=utils.NumpyEncoder)

            # Table | Generate ground truth
            if ground_truth:
                gt = vocLabels_to_groundTruth(pathTargets=path_annotations, img=img01, features=features, adjust_labels_to_textboxes=adjust_targets_to_textboxes, add_edge_separators=add_edge_separators)
                with open(path_out_targets / f'{tableName}.json', 'w') as f:
                    json.dump(gt, f, cls=utils.NumpyEncoder)

            # Table | Meta
            meta = {}
            meta['table_coords'] = dict(x0=tableRect.x0, x1=tableRect.x1, y0=tableRect.y0, y1=tableRect.y1)
            meta['dpi_pdf'] = dpi_pymupdf
            meta['dpi_model'] = dpi_model
            meta['dpi_words'] = dpi_ocr
            meta['padding_model'] = padding
            meta['name_stem'] = Path(pdfName).stem
            meta['image_angle'] = angle

            with open(path_out_meta_lineLevel / f'{tableName}.json', 'w') as f:
                json.dump(meta, f)


            # Table | Visual
            if draw_images:
                # Table | Visual | Image with ground truth and text feature
                img_annot = img_cv.copy()
                row_annot = []
                col_annot = []

                # Table | Visual | Image with ground truth and text feature | Text Feature
                # Table | Visual | Image with ground truth and text feature | Text Feature | Text rectangle
                row_in_textrectangle = utils.convert_01_array_to_visual(features['row_in_textrectangle'], width=20)
                row_annot.append(row_in_textrectangle)

                col_in_textrectangle = utils.convert_01_array_to_visual(features['col_in_textrectangle'], width=20)
                col_annot.append(col_in_textrectangle)
                
                # Table | Visual | Image with ground truth and text feature | Text Feature | Textline like rowstart
                indicator_textline_like_rowstart = utils.convert_01_array_to_visual(features['row_between_textlines_like_rowstart'], width=20)
                row_annot.append(indicator_textline_like_rowstart)

                # Table | Visual | Image with ground truth and text feature | Text Feature | Nearest right is startlike
                indicator_nearest_right_is_startlike = utils.convert_01_array_to_visual(features['col_nearest_right_is_startlike_share'], width=20)
                col_annot.append(indicator_nearest_right_is_startlike)

                # Table | Visual | Image with ground truth and text feature | Text Feature | Words crossed (lighter = fewer words crossed)
                wc_row = utils.convert_01_array_to_visual(features['row_wordsCrossed_relToMax'], invert=True, width=20)
                row_annot.append(wc_row)

                wc_col = utils.convert_01_array_to_visual(features['col_wordsCrossed_relToMax'], invert=True, width=20)
                col_annot.append(wc_col)

                # Table | Visual | Image with ground truth and text feature | Add feature bars
                row_annot = np.concatenate(row_annot, axis=1)
                img_annot = np.concatenate([img_annot, row_annot], axis=1)

                col_annot = np.concatenate(col_annot, axis=1).T
                col_annot = np.concatenate([col_annot, np.full(shape=(col_annot.shape[0], row_annot.shape[1]), fill_value=255, dtype=np.uint8)], axis=1)
                img_annot = np.concatenate([img_annot, col_annot], axis=0)

                # Table | Visual | Image with ground truth and text feature | Add ground truth
                if ground_truth:
                    img_gt_row = np.full(img_annot.shape, fill_value=255, dtype=np.uint8)
                    indicator_gt_row = utils.convert_01_array_to_visual(1-gt['row'], width=img_annot.shape[1])
                    img_gt_row[:indicator_gt_row.shape[0], :indicator_gt_row.shape[1]] = indicator_gt_row
                    img_gt_row = cv.cvtColor(img_gt_row, code=cv.COLOR_GRAY2RGB)
                    img_gt_row[:, :, 0] = 255
                    img_gt_row = Image.fromarray(img_gt_row).convert('RGBA')
                    img_gt_row.putalpha(int(0.1*255))

                    img_gt_col = np.full(img_annot.shape, fill_value=255, dtype=np.uint8)
                    indicator_gt_col = utils.convert_01_array_to_visual(1-gt['col'], width=img_annot.shape[0]).T
                    img_gt_col[:indicator_gt_col.shape[0], :indicator_gt_col.shape[1]] = indicator_gt_col
                    img_gt_col = cv.cvtColor(img_gt_col, code=cv.COLOR_GRAY2RGB)
                    img_gt_col[:, :, 1] = 255
                    img_gt_col = Image.fromarray(img_gt_col).convert('RGBA')
                    img_gt_col.putalpha(int(0.3*255))

                    img_annot_color = Image.fromarray(cv.cvtColor(img_annot, code=cv.COLOR_GRAY2RGB)).convert('RGBA')
                    img_gt = Image.alpha_composite(img_gt_col, img_gt_row)
                    img_complete = Image.alpha_composite(img_annot_color, img_gt).convert('RGB')
                    img_complete.save(path_out_images_annotated / f'{tableName}{image_format}')
                else:
                    Image.fromarray(img_annot).save(path_out_images_annotated / f'{tableName}{image_format}')


# Line-level | Preprocess | Directory level
def preprocess_lineLevel(path_images, path_pdfs, path_out, path_data_skew=None, 
                                dpi_ocr=300,
                                replace_dirs='warn', n_workers=-1, verbosity=logging.INFO, languages_easyocr=['nl', 'fr', 'de', 'en'], languages_tesseract='nld+fra+deu+eng', gpu=True, split_stub_page='-p', split_stub_table='_t',
                                ground_truth=False, draw_images=True,
                                config_pytesseract_fast = r'--tessdata-dir "C:\Program Files\Tesseract-OCR\tessdata_fast" --oem 3 --psm 11',
                                config_pytesseract_legacy = r'--tessdata-dir "C:\Program Files\Tesseract-OCR\tessdata_legacy_best" --oem 0 --psm 11'):
    # Parameters
    path_out_words = path_out / 'words'
    path_out_features = path_out / 'features_lineLevel'
    path_out_meta_lineLevel = path_out / 'meta_lineLevel'
    path_out_images_annotated = path_out / 'tables_annotated_featuresAndTargets' if ground_truth else path_out / 'tables_annotated_features'       
    # reader = easyocr.Reader(lang_list=languages_easyocr, gpu=gpu, quantize=True)

    # Make folders
    utils.makeDirs(path_out_words, replaceDirs='overwrite')
    utils.makeDirs(path_out_features, replaceDirs=replace_dirs)
    utils.makeDirs(path_out_meta_lineLevel, replaceDirs=replace_dirs)
    if draw_images:
        utils.makeDirs(path_out_images_annotated, replaceDirs=replace_dirs)

    # Assemble list of pages within pdfs that contain tables
    pdfNames = set([entry.name.split(split_stub_page)[0] for entry in os.scandir(path_images)])
    pdfNamesAndPages = {pdfName: list(set([int(entry.split(split_stub_page)[1].split(split_stub_table)[0]) for entry in glob(pathname=f'{pdfName}*', root_dir=path_images, recursive=False, include_hidden=False)])) for pdfName in pdfNames}

    # Generate words files and feature files
    # TD: add code to start easyocr endpoint
    if n_workers == 1:
        for pdfNameAndPage in tqdm(pdfNamesAndPages.items(), desc='Process line-level | Looping over files', smoothing=0.1):         # pdfNameAndPage = list(pdfNamesAndPages.items())[0]    
            utils.pdf_to_words(pdfNameAndPage=pdfNameAndPage, path_pdfs=path_pdfs, path_data_skew=path_data_skew, path_out_words=path_out_words,
                            dpi_ocr=dpi_ocr, split_stub_page=split_stub_page,
                            languages_tesseract=languages_tesseract, config_pytesseract_fast=config_pytesseract_fast, config_pytesseract_legacy=config_pytesseract_legacy,
                            draw_images=True)
            preprocess_lineLevel_singlePdf(pdfNameAndPage=pdfNameAndPage, path_out=path_out, path_out_features=path_out_features,
                                                        replace_dirs=replace_dirs)
    else:
        results = Parallel(n_jobs=n_workers, backend='loky', verbose=verbosity // 2)(delayed(utils.pdf_to_words)(
            pdfNameAndPage, path_pdfs=path_pdfs, path_data_skew=path_data_skew, path_out_words=path_out_words,
            dpi_ocr=dpi_ocr, split_stub_page=split_stub_page,
            languages_tesseract=languages_tesseract, config_pytesseract_fast=config_pytesseract_fast, config_pytesseract_legacy=config_pytesseract_legacy,
            draw_images=True, disable_progressbar=True) 
            for pdfNameAndPage in pdfNamesAndPages.items())
        results = Parallel(n_jobs=n_workers, backend='loky', verbose=verbosity // 2)(delayed(preprocess_lineLevel_singlePdf)(
            pdfNameAndPage=pdfNameAndPage, path_out=path_out, path_out_features=path_out_features,
            replace_dirs=replace_dirs, disable_progressbar=True) 
            for pdfNameAndPage in pdfNamesAndPages.items())
    


# Line-level | Predict
def predict_lineLevel(path_model_file, path_data, ground_truth=False, legacy_folder_names=False, path_predictions_line=None, device='cuda', replace_dirs='warn'):
    # Parse parameters
    path_model_file = Path(path_model_file); path_data = Path(path_data)

    # Make folders
    path_predictions_line = path_predictions_line or path_data / 'predictions_lineLevel'
    utils.makeDirs(path_predictions_line, replaceDirs=replace_dirs)

    # Load model
    model = TableLineModel().to(device)
    model.load_state_dict(torch.load(path_model_file))
    model.eval()
    dataloader = dataloaders.get_dataloader_lineLevel(dir_data=path_data, ground_truth=ground_truth, legacy_folder_names=legacy_folder_names)
    
    # Loop over batches
    batch_size = dataloader.batch_size
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Process | Predict line-level | Looping over batches'):
            # Predict
            preds = model(batch.features)

            # Save
            for sampleNumber in range(batch_size):
                # Save | Get table name
                image_path = Path(batch.meta.path_image[sampleNumber])
                name_full = image_path.stem

                # Save | Get separators
                separators_row = preds_to_separators(predArray=preds.row[sampleNumber], setToMidpoint=False, addPaddingSeparators=False)
                separators_col = preds_to_separators(predArray=preds.col[sampleNumber], setToMidpoint=False, addPaddingSeparators=False)

                pred_dict = {'row_separator_predictions': separators_row.tolist(), 'col_separator_predictions': separators_col.tolist()}

                with open(path_predictions_line / f'{name_full}.json', 'w') as f:
                    json.dump(pred_dict, f)
    with open(path_predictions_line.parent / 'path_model_predictions_lineLevel.txt', 'w') as f:
        f.write(f'Model path: {path_model_file}')

# Separator-level | Preprocess
def preprocess_separatorLevel(path_model_line, path_data, path_words=None, ground_truth=False, legacy_folder_names=False, path_predictions_line=None, path_features_separator=None, path_targets_separator=None, path_annotated_images=None, draw_images=False, image_format='.png', replace_dirs='warn'):
    # Parse parameters
    path_data = Path(path_data)
    path_words = path_words or path_data / 'words'
    path_predictions_line = path_predictions_line or path_data / 'predictions_lineLevel'
    path_features_separator = path_features_separator or path_data / 'features_separatorLevel'
    path_targets_separator = path_targets_separator or path_data / 'targets_separatorLevel'
    path_annotated_images = path_annotated_images or path_data / 'images_annotated_separatorLevel'
    path_meta = path_data / 'meta_lineLevel' if not legacy_folder_names else path_data / 'meta'

    # Make folders
    utils.makeDirs(path_features_separator, replaceDirs=replace_dirs)
    if ground_truth:
        utils.makeDirs(path_targets_separator, replaceDirs=replace_dirs)
        if draw_images:
            utils.makeDirs(path_annotated_images, replaceDirs=replace_dirs)

    # Predict line-level separator indicators
    predict_lineLevel(path_model_file=path_model_line, path_data=path_data, ground_truth=ground_truth, replace_dirs=replace_dirs, legacy_folder_names=legacy_folder_names)

    # Loop over tables
    tableNames = [os.path.splitext(filename)[0] for filename in os.listdir(path_predictions_line)]
    for tableName in tqdm(tableNames, desc='Process | Generate separator features | Looping over tables'):        # tableName = tableNames[0]
        # Load 
        # Load | Meta info
        with open(path_meta / f'{tableName}.json', 'r') as f:
            metaInfo = json.load(f)

        # Load | Text
        wordsDf = utils.pageWords_to_tableWords(path_words=path_words, tableName=tableName, metaInfo=metaInfo)
        wordsDf['text'] = wordsDf['text'].str.replace('.', '')

        # Load | Separator proposals
        with open(path_predictions_line / f'{tableName}.json', 'r') as f:
            separators = json.load(f)
        separators_row = separators['row_separator_predictions']
        separators_col = separators['col_separator_predictions']

        if min([len(separator) for separator in separators.values()]) == 0:
            continue
        
        separators_midpoints_row = [sum(separator)//2 for separator in separators_row]
        separators_midpoints_col = [sum(separator)//2 for separator in separators_col]

        # Load | Line level targets
        if ground_truth:
            with open(path_data / 'labels' / f'{tableName}.json', 'r') as f:
                targets_lineLevel = json.load(f)
            targets_lineLevel_row = targets_lineLevel['row']
            targets_lineLevel_col = targets_lineLevel['col']

        # Prepare
        # Prepare | Get text between separators
        edgeBottom = wordsDf['bottom'].max()
        texts_between_separators_row = []
        for idx, _ in enumerate(separators_midpoints_row):
            firstEdge = separators_midpoints_row[idx]
            try:
                lastEdge = separators_midpoints_row[idx+1]
            except IndexError:
                lastEdge = edgeBottom
            texts = wordsDf.loc[(wordsDf['top'] > firstEdge) & (wordsDf['bottom'] <= lastEdge), 'text'].to_list()
            texts = [text for text in texts if text != '']
            texts_between_separators_row.append(texts)

        edgeRight = wordsDf['right'].max()
        texts_between_separators_col = []
        for idx, _ in enumerate(separators_midpoints_col):
            firstEdge = separators_midpoints_col[idx]
            try:
                lastEdge = separators_midpoints_col[idx+1]
            except IndexError:
                lastEdge = edgeRight
            texts = wordsDf.loc[(wordsDf['left'] > firstEdge) & (wordsDf['right'] <= lastEdge), 'text'].to_list()
            texts = [text for text in texts if text != '']
            texts_between_separators_col.append(texts)

        # Features separator-level
        features = {}

        # Features | Any text between separators
        features['text_between_separators_row'] = [int(len(texts) > 0) for texts in texts_between_separators_row]
        features['text_between_separators_col'] = [int(len(texts) > 0) for texts in texts_between_separators_col]

        # Features | Save
        with open(path_features_separator / f'{tableName}.json', 'w') as f:
            json.dump(features, f)    

        
        # Targets
        if ground_truth:
            targets_separatorLevel_row = []
            for separator in separators_row:
                targets_lineLevel_inSeparator = np.array(targets_lineLevel_row)[separator[0]:separator[1]]

                contains_separatorLines = max(targets_lineLevel_inSeparator)
                
                pattern = np.diff(targets_lineLevel_inSeparator)
                try:
                    firstEndOfSeparator = np.where(pattern == -1)[0][0]
                    lastStartOfSeparator = np.where(pattern == 1)[0][-1]
                    contains_multiple_spells = (firstEndOfSeparator < lastStartOfSeparator)
                except IndexError:
                    contains_multiple_spells = False

                proposal_is_separator = int((contains_separatorLines) and not (contains_multiple_spells))
                targets_separatorLevel_row.append(proposal_is_separator)

            targets_separatorLevel_col = []
            for separator in separators_col:    # separator = separators_col[0]
                targets_lineLevel_inSeparator = np.array(targets_lineLevel_col)[separator[0]:separator[1]]

                contains_separatorLines = max(targets_lineLevel_inSeparator)
                
                pattern = np.diff(targets_lineLevel_inSeparator)
                try:
                    firstEndOfSeparator = np.where(pattern == -1)[0][0]
                    lastStartOfSeparator = np.where(pattern == 1)[0][-1]
                    contains_multiple_spells = (firstEndOfSeparator < lastStartOfSeparator)
                except IndexError:
                    contains_multiple_spells = False

                proposal_is_separator = int((contains_separatorLines) and not (contains_multiple_spells))
                targets_separatorLevel_col.append(proposal_is_separator)
                    
            # Targets separator-level
            targets = {}
            targets['row'] = targets_separatorLevel_row
            targets['col'] = targets_separatorLevel_col

            with open(path_targets_separator / f'{tableName}.json', 'w') as f:
                json.dump(targets, f) 

        # Visualise
        if draw_images and ground_truth:
            # Load image
            img = Image.open(path_data / 'images' / f'{tableName}{image_format}').convert('RGBA')
            
            # Draw | Row
            overlay_row = Image.new('RGBA', img.size, (0, 0, 0 ,0))
            img_annot_row = ImageDraw.Draw(overlay_row)
            
            for idx, separator in enumerate(separators_row):
                shape = [0, separator[0], img.width, separator[1]]
                color = utils.COLOR_CORRECT if targets_separatorLevel_row[idx] == 1 else utils.COLOR_WRONG
                img_annot_row.rectangle(xy=shape, fill=color)

            # Draw | Col
            overlay_col = Image.new('RGBA', img.size, (0, 0, 0 ,0))
            img_annot_col = ImageDraw.Draw(overlay_col)
            for idx, separator in enumerate(separators_col):
                shape = [separator[0], 0, separator[1], img.height]
                color = utils.COLOR_CORRECT if targets_separatorLevel_col[idx] == 1 else utils.COLOR_WRONG
                img_annot_col.rectangle(xy=shape, fill=color)

            # Save image
            img = Image.alpha_composite(img, overlay_row)
            img = Image.alpha_composite(img, overlay_col).convert('RGB')
            img.save(path_annotated_images / f'{tableName}.png')

# Inference
def predict_and_process(path_model_file, path_data, ground_truth=False, path_words=None, path_pdfs=None, device='cuda', replace_dirs='warn', path_processed=None, padding=40, draw_text_scale=1, truth_threshold=0.5,
                        out_data=True, out_images=False, out_labels_separators=False, out_labels_rows=False, subtract_one_from_pagenumer=False):
    # Parse parameters
    path_model_file = Path(path_model_file); path_data = Path(path_data)
    path_words = path_words or path_data / 'words'
    path_pdfs = path_pdfs or path_data / 'pdfs'

    # Make folders
    path_processed = path_processed or path_data / 'predictions_separatorLevel'
    path_processed_data         = path_processed / 'data'
    path_processed_images       = path_processed / 'annotated_data'
    path_processed_labels_rows  = path_processed / 'labels_rows'
    utils.makeDirs(path_processed_data, replaceDirs=replace_dirs)

    if out_images:       
        utils.makeDirs(path_processed_images, replaceDirs=replace_dirs)
    if out_labels_rows:
        utils.makeDirs(path_processed_labels_rows, replaceDirs=replace_dirs)

    # Load model
    model = TableSeparatorModel().to(device)
    model.load_state_dict(torch.load(path_model_file))
    model.eval()
    dataloader = dataloaders.get_dataloader_separatorLevel(dir_data=path_data, ground_truth=ground_truth)

    # Load OCR reader
    reader = easyocr.Reader(lang_list=['nl', 'fr', 'de', 'en'], gpu=True, quantize=True)

    # Padding separator
    TableRect = namedtuple('tableRect', field_names=['x0', 'x1', 'y0', 'y1'])
    FONT_TEXT = ImageFont.truetype('arial.ttf', size=int(26*draw_text_scale))
    FONT_BIG = ImageFont.truetype('arial.ttf', size=int(48*draw_text_scale))

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Process | Predict | Looping over batches'):
            # Compute prediction
            try:
                preds = model(batch.features)
            except RuntimeError:
                continue

            # Convert to data
            batch_size = dataloader.batch_size
            for sampleNumber in range(batch_size):

                # Words
                # Words | Parse sample name
                image_path = Path(batch.meta.path_image[sampleNumber])
                name_full = image_path.stem
                name_stem = batch.meta.name_stem[sampleNumber]

                name_pdf = f"{name_stem}.pdf"
                pageNumber = int(name_full.split('-p')[-1].split('_t')[0])
                if subtract_one_from_pagenumer:
                    pageNumber = pageNumber - 1
                tableNumber = int(name_full.split('_t')[-1])

                name_words = f"{name_stem}-p{pageNumber}.pq"
                wordsPath = path_words / f"{name_words}"

                # Words | Use pdf-based words if present
                tableRect = TableRect(**{key: value[sampleNumber].item() for key, value in batch.meta.table_coords.items()})
                dpi_pdf = batch.meta.dpi_pdf[sampleNumber].item()
                dpi_model = batch.meta.dpi_model[sampleNumber].item()
                padding_model = batch.meta.padding_model[sampleNumber].item()
                dpi_words = batch.meta.dpi_words[sampleNumber].item()
                angle = batch.meta.image_angle[sampleNumber].item()

                wordsDf = pd.read_parquet(wordsPath).drop_duplicates()
                wordsDf.loc[:, ['left', 'top', 'right', 'bottom']] = (wordsDf.loc[:, ['left', 'top', 'right', 'bottom']] * (dpi_pdf / dpi_words))
                textSource = 'ocr-based' if len(wordsDf.query('ocrLabel == "pdf"')) == 0 else 'pdf-based'

                # Extract image from pdf
                pdf = fitz.open(path_pdfs / name_pdf)
                page = pdf.load_page(pageNumber)
                tableRect_fitz = fitz.IRect(round(tableRect.x0), round(tableRect.y0), round(tableRect.x1), round(tableRect.y1))
                img = page.get_pixmap(dpi=dpi_words, clip=tableRect_fitz, colorspace=fitz.csGRAY)
                img = np.frombuffer(img.samples, dtype=np.uint8).reshape(img.height, img.width, img.n)
                _, img_array = cv.threshold(np.array(img), 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
                img_tight = Image.fromarray(img_array)
                scale_factor = dpi_words/dpi_model
                img = Image.new(img_tight.mode, (int(img_tight.width+padding*2*scale_factor), int(img_tight.height+padding*2*scale_factor)), 255)
                img.paste(img_tight, (int(padding*scale_factor), int(padding*scale_factor)))
                img = img.rotate(angle, expand=True, fillcolor='white', resample=Image.Resampling.BICUBIC)     
                
                # Cells
                # Cells | Convert predictions to boundaries
                separators_row = np.array([separator.cpu().numpy() for idx, separator in enumerate(batch.features.proposedSeparators_row[sampleNumber]) if preds.row[sampleNumber][idx] >= truth_threshold])
                separators_col = np.array([separator.cpu().numpy() for idx, separator in enumerate(batch.features.proposedSeparators_col[sampleNumber]) if preds.col[sampleNumber][idx] >= truth_threshold])
                
                separators_row_wide = parse_separators(separatorArray=separators_row, padding=padding, size=int(img.height//scale_factor), setToMidpoint=False)
                separators_col_wide = parse_separators(separatorArray=separators_col, padding=padding, size=int(img.width//scale_factor), setToMidpoint=False)

                separators_row_mid = parse_separators(separatorArray=separators_row, padding=padding, size=int(img.height//scale_factor), setToMidpoint=True)
                separators_col_mid = parse_separators(separatorArray=separators_col, padding=padding, size=int(img.width//scale_factor), setToMidpoint=True)

                # Cells | Convert boundaries to cells
                cells = [dict(x0=separators_col_mid[c][1]+1, y0=separators_row_mid[r][1]+1, x1=separators_col_mid[c+1][0], y1=separators_row_mid[r+1][0], row=r, col=c)
                            for r in range(len(separators_row_mid)-1) for c in range(len(separators_col_mid)-1)]
                cells = [scale_cell_to_dpi(cell, dpi_start=dpi_model, dpi_target=dpi_words) for cell in cells]

                if out_data or out_images:
                    # Data 
                    # Data | OCR by initial cell
                    if textSource == 'ocr-based':                     
                        img_array = np.array(img)

                        for cell in cells:
                            textList = reader.readtext(image=img_array[cell['y0']:cell['y1'], cell['x0']:cell['x1']], batch_size=60, detail=1)
                            if textList:
                                textList_sorted = sorted(textList, key=lambda el: (el[0][0][1]//15, el[0][0][0]))       # round height to the lowest X to avoid height mismatch from fucking things up
                                cell['text'] = ' '.join([el[1] for el in textList_sorted])
                            else:
                                cell['text'] = ''
                    else:
                        # Reduce wordsDf to table dimensions
                        wordsDf = wordsDf.loc[(wordsDf['top'] >= tableRect.y0) & (wordsDf['left'] >= tableRect.x0) & (wordsDf['bottom'] <= tableRect.y1) & (wordsDf['right'] <= tableRect.x1)]

                        # Adapt wordsDf coordinates to model table coordinates
                        wordsDf.loc[:, ['left', 'right']] = wordsDf.loc[:, ['left', 'right']] - tableRect.x0
                        wordsDf.loc[:, ['top', 'bottom']] = wordsDf.loc[:, ['top', 'bottom']] - tableRect.y0
                        wordsDf.loc[:, ['left', 'right', 'top', 'bottom']] = wordsDf.loc[:, ['left', 'right', 'top', 'bottom']] * (dpi_words / dpi_pdf) + padding_model * (dpi_words/dpi_model)
                        wordsDf.loc[:, ['left', 'right', 'top', 'bottom']] = wordsDf.loc[:, ['left', 'right', 'top', 'bottom']]
                        wordsDf = wordsDf.rename(columns={'left': 'x0', 'right': 'x1', 'top': 'y0', 'bottom': 'y1'})

                        # Assign text to cells
                        for cell in cells:
                            overlap = wordsDf.apply(lambda row: boxes_intersect(box=cell, box_target=row), axis=1)
                            cell['text'] = ' '.join(wordsDf.loc[overlap, 'text'])

                # Data | Convert to dataframe
                if out_data:
                    if cells:
                        df = pd.DataFrame.from_records(cells)[['row', 'col', 'text']].pivot(index='row', columns='col', values='text').replace(' ', pd.NA).replace('', pd.NA)       #.convert_dtypes(dtype_backend='pyarrow')
                        df = df.dropna(axis='columns', how='all').dropna(axis='index', how='all').reset_index(drop=True)
                    else:
                        df = pd.DataFrame()

                    if len(df):
                        # Data | Clean
                        # Data | Clean | Combine "(" ")" columns
                        uniques = {col: set(df[col].unique()) for col in df.columns}
                        onlyParenthesis_open =  [col for col, unique in uniques.items() if unique == set([pd.NA, '('])]
                        onlyParenthesis_close = [col for col, unique in uniques.items() if unique == set([pd.NA, ')'])]

                        for col in onlyParenthesis_open:
                            parenthesis_colIndex = df.columns.tolist().index(col)
                            if parenthesis_colIndex == (len(df.columns) - 1):           # Drop if last column only contains (
                                df = df.drop(columns=[col])
                            else:                                                       # Otherwise add to next column
                                target_col = df.columns[parenthesis_colIndex+1]
                                df[target_col] = df[target_col] + df[col]               
                        for col in onlyParenthesis_close:
                            parenthesis_colIndex = df.columns.tolist().index(col)
                            if parenthesis_colIndex == 0:                               # Drop if first column only contains )
                                df = df.drop(columns=[col])
                            else:                                                       # Otherwise add to previous column
                                target_col = df.columns[parenthesis_colIndex-1]
                                df[target_col] = df[target_col] + df[col]               
                                df = df.drop(columns=[col])

                        # Data | Clean | If last column only contains 1 or |, it is probably an OCR error
                        ocr_mistakes_verticalLine = set([pd.NA, '1', '|'])
                        if len(set(df.iloc[:, -1].unique()).difference(ocr_mistakes_verticalLine)) == 0:
                            df = df.drop(df.columns[-1],axis=1)

                        # Data | Clean | Column names
                        # Data | Clean | Column names | First column is probably label column (if longest string in columns)
                        longestStringLengths = {col: df.loc[1:, col].str.len().max() for col in df.columns}
                        longestString = max(longestStringLengths, key=longestStringLengths.get)
                        if longestString == 0:
                            df.loc[0, longestString] = 'Labels'

                        # Data | Clean | Column names | Replace column names by first non missing element in first five rows
                        df.columns = get_first_non_null_values(df)
                        df.columns = list(number_duplicates(df.columns))
                        df = df.drop(index=0).reset_index(drop=True)
                        
                        # Data | Clean | Drop rows with only label and code information
                        valueColumns = [col for col in df.columns if (col is pd.NA) or ((col not in ['Labels']) and not (col.startswith('Codes')))]
                        df = df.dropna(axis='index', subset=valueColumns, how='all').reset_index(drop=True)

                    # Data | Save
                    df.to_parquet(path_processed_data / f'{name_full}.pq')
                
                # Visualise
                # Visualise | Cell annotations
                if out_images:
                    table = Image.new('RGBA', img.size, (255,255,255,255))
                    img_annot = ImageDraw.Draw(table)
                    for cell in cells:
                        # img_annot.rectangle(xy=(cell['x0'], cell['y0'], cell['x1'], cell['y1']), fill='COLOR_CELL', outline=COLOR_OUTLINE, width=2)
                        try:
                            img_annot.rectangle(xy=(cell['x0'], cell['y0'], cell['x1'], cell['y1']), fill='white', outline='black', width=2)
                            if cell['text']:
                                img_annot.rectangle(xy=img_annot.textbbox((cell['x0']+1, cell['y0']+1), text=cell['text'], font=FONT_TEXT, anchor='la'), fill=(255, 255, 255, 240), outline=(255, 255, 255, 240), width=2)
                                img_annot.text(xy=(cell['x0']+1, cell['y0']+1), text=cell['text'], fill=(0, 0, 0, 255), anchor='la', font=FONT_TEXT,)
                        except ValueError:
                            tqdm.write(f'{name_full}: {cell}')
                    img_annot.text(xy=(img.width // 2, img.height - 4), text=textSource, font=FONT_BIG, fill=(0, 0, 0, 230), anchor='md')
                    
                    # Visualise | Save image
                    bg = img.convert('RGBA')
                    bg.putalpha(OPACITY_ORIGINAL)
                    tableImage = Image.alpha_composite(table, bg).convert('RGB')
                    tableImage.save(path_processed_images / f'{name_full}.png')

                # Labels
                # Labels | Separators
                if out_labels_separators:
                    raise Exception('Generating separator labels not yet implemented')
                
                # Labels | Rows, columns and separators
                if out_labels_rows:
                    rows = np.sort(np.column_stack((separators_row_wide[:-1, 1], separators_row_wide[1:, 0])), axis=1)
                    cols = np.sort(np.column_stack((separators_col_wide[:-1, 1], separators_col_wide[1:, 0])), axis=1)
                    tableBbox = dict(xmin=cols.min(), xmax=cols.max(), ymin=rows.min(), ymax=rows.max())
                    tableBbox = {key: str(value) for key, value in tableBbox.items()}

                    # Generate general xml
                    xml = etree.Element('annotation')
                    xml_size = etree.SubElement(xml, 'size')
                    xml_width  = etree.SubElement(xml_size, 'width');   xml_width.text = str(int( img.width // scale_factor))
                    xml_height = etree.SubElement(xml_size, 'height'); xml_height.text = str(int(img.height // scale_factor))

                    # Add confidence score (0: all separators 0.5; 1: all separators 0 or 1) (we take 25th percentile of confidence for rows and cols separately)
                    confidence = torch.mean(torch.stack([torch.quantile(torch.abs((preds.row[sampleNumber].squeeze() - 0.5)), q=0.5), torch.quantile(torch.abs((preds.col[sampleNumber].squeeze() - 0.5)), q=0.5)]))*2
                    xml_confidence = etree.SubElement(xml, 'confidence_score'); xml_confidence.text = f'{confidence.item():0.3f}'
                    
                    # Add table bbox
                    xml_table = etree.SubElement(xml, 'object')
                    xml_label = etree.SubElement(xml_table, 'name'); xml_label.text = 'table'
                    xml_table_bbox = etree.SubElement(xml_table, 'bndbox')
                    for edge in tableBbox:
                        _ = etree.SubElement(xml_table_bbox, edge)
                        _.text = str(tableBbox[edge])
                    

                    # Generate row/column xml
                    arrayToXml(array=rows, xmlRoot=xml, label='table row', orientation='row', tableBbox=tableBbox)
                    arrayToXml(array=cols, xmlRoot=xml, label='table column', orientation='col', tableBbox=tableBbox)

                    # Save
                    tree = etree.ElementTree(xml)
                    tree.write(path_processed_labels_rows / f'{name_full}.xml', pretty_print=True, xml_declaration=False, encoding='utf-8')


# Individual file run for testing
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    PATH_ROOT = Path(r"F:\ml-parsing-project\table-parse-split")
    path_model_file = PATH_ROOT / 'models' / 'test' / 'model_best.pt' 
    path_data = PATH_ROOT / 'data' / 'real_narrow'
    path_words = PATH_ROOT / 'data' / 'words'
    path_pdfs = PATH_ROOT / 'data' / 'pdfs'

    predict_and_process(path_model_file=path_model_file, path_data=path_data / 'val', path_words=path_words, device=device)