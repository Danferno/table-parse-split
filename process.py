# TD: seems to drop last column?

# Imports
import torch    # type : ignore
import numpy as np
import pandas as pd
from pathlib import Path
import easyocr
import fitz     # type : ignore
import cv2 as cv
from tqdm import tqdm
import json
import os
from lxml import etree

from collections import namedtuple, Counter
from PIL import Image, ImageFont, ImageDraw

import utils
from model import TableLineModel, TableSeparatorModel
import dataloaders

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


# Function
def detect_from_pdf(path_data, path_out, path_model_file_detect=None, device='cuda', replace_dirs='warn', draw_images=False, padding=40):
    ''' Detect tables'''
    ...

def predict_lineLevel(path_model_file, path_data, path_predictions_line=None, device='cuda', replace_dirs='warn'):
    # Parse parameters
    path_model_file = Path(path_model_file); path_data = Path(path_data)

    # Make folders
    path_predictions_line = path_predictions_line or path_data / 'predictions_lineLevel'
    utils.makeDirs(path_predictions_line, replaceDirs=replace_dirs)

    # Load model
    model = TableLineModel().to(device)
    model.load_state_dict(torch.load(path_model_file))
    model.eval()
    dataloader = dataloaders.get_dataloader_lineLevel(dir_data=path_data)
    
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

def generate_featuresAndTargets_separatorLevel(path_best_model_line, path_data, path_words, path_predictions_line=None, path_features_separator=None, path_targets_separator=None, path_annotated_images=None, draw_images=False, image_format='.png', replace_dirs='warn'):
    # Parse parameters
    path_data = Path(path_data)
    path_predictions_line = path_predictions_line or path_data / 'predictions_lineLevel'
    path_features_separator = path_features_separator or path_data / 'features_separatorLevel'
    path_targets_separator = path_targets_separator or path_data / 'targets_separatorLevel'
    path_annotated_images = path_annotated_images or path_data / 'images_annotated_separatorLevel'

    # Make folders
    utils.makeDirs(path_features_separator, replaceDirs=replace_dirs)
    utils.makeDirs(path_targets_separator, replaceDirs=replace_dirs)
    utils.makeDirs(path_annotated_images, replaceDirs=replace_dirs)

    # Predict line-level separator indicators
    predict_lineLevel(path_model_file=path_best_model_line, path_data=path_data, replace_dirs=replace_dirs)

    # Loop over tables
    tableNames = [os.path.splitext(filename)[0] for filename in os.listdir(path_predictions_line)]
    for tableName in tqdm(tableNames, desc='Process | Generate separator features | Looping over tables'):        # tableName = tableNames[0]
        # Load 
        # Load | Meta info
        with open(path_data / 'meta' / f'{tableName}.json', 'r') as f:
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
        if draw_images:
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


def predict_and_process(path_model_file, path_data, path_words, path_pdfs, device='cuda', replace_dirs='warn', path_processed=None, padding=40, draw_text_scale=1, truth_threshold=0.5,
                        out_data=True, out_images=False, out_labels_separators=False, out_labels_rows=True):
    # Parse parameters
    path_model_file = Path(path_model_file); path_data = Path(path_data); path_words = Path(path_words); path_pdfs = Path(path_pdfs)

    # Make folders
    path_processed = path_processed or path_model_file.parent / 'processed'
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
    dataloader = dataloaders.get_dataloader_separatorLevel(dir_data=path_data)

    # Load OCR reader
    reader = easyocr.Reader(lang_list=['nl', 'fr', 'de', 'en'], gpu=True, quantize=True)

    # Padding separator
    TableRect = namedtuple('tableRect', field_names=['x0', 'x1', 'y0', 'y1'])
    FONT_TEXT = ImageFont.truetype('arial.ttf', size=int(26*draw_text_scale))
    FONT_BIG = ImageFont.truetype('arial.ttf', size=int(48*draw_text_scale))

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Process | Predict | Looping over batches'):
            # Compute prediction
            preds = model(batch.features)

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
                page = pdf.load_page(pageNumber-1)
                img = page.get_pixmap(dpi=dpi_words, clip=(tableRect.x0, tableRect.y0, tableRect.x1, tableRect.y1), colorspace=fitz.csGRAY)
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
                        img_annot.rectangle(xy=(cell['x0'], cell['y0'], cell['x1'], cell['y1']), fill='white', outline='black', width=2)
                        if cell['text']:
                            img_annot.rectangle(xy=img_annot.textbbox((cell['x0']+1, cell['y0']+1), text=cell['text'], font=FONT_TEXT, anchor='la'), fill=(255, 255, 255, 240), outline=(255, 255, 255, 240), width=2)
                            img_annot.text(xy=(cell['x0']+1, cell['y0']+1), text=cell['text'], fill=(0, 0, 0, 255), anchor='la', font=FONT_TEXT,)
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
                    rows = np.column_stack((separators_row_wide[:-1, 1], separators_row_wide[1:, 0]))
                    cols = np.column_stack((separators_col_wide[:-1, 1], separators_col_wide[1:, 0]))
                    tableBbox = dict(xmin=cols.min(), xmax=cols.max(), ymin=rows.min(), ymax=rows.max())
                    tableBbox = {key: str(value) for key, value in tableBbox.items()}

                    # Generate general xml
                    xml = etree.Element('annotation')
                    xml_size = etree.SubElement(xml, 'size')
                    xml_width  = etree.SubElement(xml_size, 'width');   xml_width.text = str(int( img.width // scale_factor))
                    xml_height = etree.SubElement(xml_size, 'height'); xml_height.text = str(int(img.height // scale_factor))
                    
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