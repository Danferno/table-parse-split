import os, shutil
import click
import pandas as pd
import json
import pickle
from collections import namedtuple
import warnings
from io import BytesIO

import numba as nb
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import cv2 as cv
from PIL import Image,  ImageDraw, ImageFont
from deskew import determine_skew
import tabledetect
import requests
from requests.adapters import HTTPAdapter, Retry

from pathlib import Path
import logging
import random
import fitz # type : ignore

import pytesseract
import imagehash

import time
from tqdm import tqdm
from joblib import Parallel, delayed
from glob import glob

import process
import train
import evaluate

# Constants
COLOR_CORRECT = (0, 255, 0, int(0.25*255))
COLOR_WRONG = (255, 0, 0, int(0.25*255))
COLOR_MIDDLE = (128, 128, 0, int(0.25*255))

random.seed(498465464)

# Functions
def replaceDir(path):
    shutil.rmtree(path)
    os.makedirs(path)

def makeDirs(path, replaceDirs='warn'):
    try:
        os.makedirs(path)
    except FileExistsError:
        if replaceDirs == 'warn':
            if click.confirm(f'This will remove folder {path}. Are you sure you are okay with this?'):
                replaceDir(path)
            else:
                raise InterruptedError(f'User was not okay with removing {path}')
        elif replaceDirs == 'overwrite':
            pass
        elif replaceDirs:
            replaceDir(path)
        else:
            raise FileExistsError
def ensure_startsWithDot(inputString):
    return '.'+inputString if inputString[0] != '.' else inputString

@nb.vectorize([nb.int32(nb.int32, nb.int32)])
def reset_cumsum(x, y):
    return x + y if y else 0
@nb.jit(nopython=True)
def zero_start(array):
    for idx, row in enumerate(array):
        zeros = np.where(row == 0)[0]
        if len(zeros):
            array[idx, :zeros[0]] = 0
        else:
            array[idx] = 0
    return array

# Pdf to words
# Pdf to words | Helpers
BoxIntersect = namedtuple('BoxIntersect', field_names=['left', 'right', 'top', 'bottom', 'intersect', 'Index'])
def boxes_intersect(box, box_target):
    overlap_x = ((box.left >= box_target.left) & (box.left < box_target.right)) | ((box.right >= box_target.left) & (box.left < box_target.left))
    overlap_y = ((box.top >= box_target.top) & (box.top < box_target.bottom)) | ((box.bottom >= box_target.top) & (box.top < box_target.top))

    return all([overlap_x, overlap_y])
def box_intersects_boxList(box, box_targets):
    overlap = any([boxes_intersect(box, box_target=box_target) for box_target in box_targets])
    return overlap
def tighten_word_bbox(img_blackIs1, bboxRow, window_emptyrow_relative_factor=5, window_emptyrow_window_size=5):
    np.seterr(all='raise')
    
    # Get bbox pixels
    pixelsInBbox = img_blackIs1[bboxRow['top']:bboxRow['bottom']+1, bboxRow['left']:bboxRow['right']+1]

    # Exclude horizontal lines
    wide_left = max(0, bboxRow['left']-20)
    wide_right = min(img_blackIs1.shape[1], bboxRow['right']+20)
    widerPixels = img_blackIs1[bboxRow['top']:bboxRow['bottom']+1, wide_left:wide_right]
    containsLine = (widerPixels.mean(axis=1) >= 0.85)

    try:
        firstLine = np.where(containsLine)[0][0]
        shave_line_top = np.where(containsLine[firstLine:] == False)[0][0] + firstLine + 1 if firstLine < pixelsInBbox.shape[0] // 4 else 0
    except IndexError:
        shave_line_top = 0

    try:
        lastLine = np.where(containsLine)[0][-1]
        shave_line_bot = np.where(containsLine[:lastLine] == False)[0][-1] + 1 if lastLine > pixelsInBbox.shape[0] // 4 * 3 else 0
    except IndexError:
        shave_line_bot = 0

    # Exclude vertical lines
    tall_top = max(0, bboxRow['top']-20)
    tall_bottom = min(img_blackIs1.shape[0], bboxRow['bottom']+20+1)
    tallPixels = img_blackIs1[tall_top:tall_bottom, bboxRow['left']:bboxRow['right']+1]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        containsLine = (tallPixels.mean(axis=0) >= 0.9)

    try:
        firstLine = np.where(containsLine)[0][0]
        shave_line_left = np.where(containsLine[firstLine:] == False)[0][0] + firstLine + 1 if firstLine < pixelsInBbox.shape[1] // 4 else 0
    except IndexError:
        shave_line_left = 0

    try:
        lastLine = np.where(containsLine)[0][-1]
        shave_line_right = np.where(containsLine[:lastLine] == False)[0][-1] + 1 if lastLine > pixelsInBbox.shape[1] // 4 * 3 else 0
    except IndexError:
        shave_line_right = 0

    pixelsInBbox = pixelsInBbox[shave_line_top:(pixelsInBbox.shape[0]-shave_line_bot), shave_line_left:(pixelsInBbox.shape[1]-shave_line_right)]

    rowBlacks = pixelsInBbox.sum(axis=1)
    rowBlackCount = len(rowBlacks[rowBlacks!=0])
    colBlacks = pixelsInBbox.sum(axis=0)

    if rowBlackCount:
        row_fewBlacks_absolute = (rowBlacks <= 2)
        row_fewBlacks_relative = (rowBlacks <= np.mean(rowBlacks[rowBlacks!=0])/window_emptyrow_relative_factor)
        # Rolling window approach to control for text blocks that overlap previous line text
        try:      
            row_fewBlacks_window = np.median(sliding_window_view(rowBlacks, window_emptyrow_window_size+1), axis=1)
            if row_fewBlacks_window.max() < 0.01:       # For single rows (e.g. all dots) it sometimes returns all empty
                row_fewBlacks_window = np.full_like(row_fewBlacks_window, fill_value=False, dtype=bool)
            else:
                row_fewBlacks_window = (row_fewBlacks_window < 0.01)
            row_fewBlacks_window = np.insert(row_fewBlacks_window, obj=row_fewBlacks_window.size//2, values=np.full(shape=window_emptyrow_window_size, fill_value=False))     # Insert 999 in middle to recover obs lost due to window
            
        except ValueError:
            row_fewBlacks_window = np.full_like(a=row_fewBlacks_absolute, fill_value=False)
    else:
        row_fewBlacks_absolute = np.full(shape=rowBlacks.shape, fill_value=True)
        row_fewBlacks_relative = np.ndarray.view(row_fewBlacks_absolute)
        row_fewBlacks_window = np.ndarray.view(row_fewBlacks_absolute)
    
    rowLikelyEmpty = row_fewBlacks_absolute | row_fewBlacks_relative | row_fewBlacks_window      
    colLikelyEmpty = (colBlacks == 0)

    shave = {}
    try:
        shave['top'] = np.where(rowLikelyEmpty == False)[0][0]
        shave['bottom'] = np.where(np.flip(rowLikelyEmpty) == False)[0][0]
    # All empty based on abs+rel+wind
    except IndexError:
        try:
            shave['top'] = np.where(row_fewBlacks_absolute == False)[0][0]  + shave_line_top
            shave['bottom'] = np.where(row_fewBlacks_absolute == False)[0][-1]  + shave_line_top
        # All empty based on abs too
        except IndexError:
            bboxRow['toKeep'] = False
            return bboxRow
    
    shave['left'] = np.where(colLikelyEmpty == False)[0][0]
    shave['right'] = np.where(np.flip(colLikelyEmpty) == False)[0][0]

    bboxRow['left'] = bboxRow['left'] + shave['left'] + shave_line_left
    bboxRow['right'] = bboxRow['right'] - shave['right'] + shave_line_left
    bboxRow['top'] = bboxRow['top'] + shave['top'] + shave_line_top
    bboxRow['bottom'] = bboxRow['bottom'] - shave['bottom'] + shave_line_top

    return bboxRow
def harmonise_bbox_height(textDf):
    textDf['top_uniform'] = textDf.groupby(by=['blockno', 'lineno'])['top'].transform(min)
    textDf['bottom_uniform'] = textDf.groupby(by=['blockno', 'lineno'])['bottom'].transform(max)

    # Harmonise top/bottom by line | Check if always gap between lines in uniform setting
    checkGapDf = textDf.loc[:, ['lineno', 'top_uniform', 'bottom_uniform']].drop_duplicates()
    checkGapDf['next_top'] = checkGapDf['top_uniform'].shift(-1).fillna(9999999)
    checkGapDf['previous_bottom'] = checkGapDf['bottom_uniform'].shift(1).fillna(0)
    checkGapDf['uniform_OK'] = (checkGapDf['bottom_uniform'] < checkGapDf['next_top']) & (checkGapDf['top_uniform'] > checkGapDf['previous_bottom'])

    # Harmonise top/bottom by line | Only make uniform if gap always remains
    textDf = textDf.merge(right=checkGapDf[['lineno', 'uniform_OK']], on='lineno', how='left')
    textDf.loc[textDf['uniform_OK'], 'top'] = textDf.loc[textDf['uniform_OK'], 'top_uniform']
    textDf.loc[textDf['uniform_OK'], 'bottom'] = textDf.loc[textDf['uniform_OK'], 'bottom_uniform']
    textDf = textDf.drop(columns=['top_uniform', 'bottom_uniform', 'uniform_OK'])

    return textDf

# Pdf to words | Main Function
def pdf_to_words(pdfNameAndPage, path_pdfs, path_words, path_data_skew=None, languages_tesseract='nld+fra+deu+eng', dpi_ocr=300, dpi_pymupdf=72, reader_endpoint='http://127.0.0.1:8000/easyocr', split_stub_page='-p', force_new_ocr=False,
                 config_pytesseract_fast='--oem 3 --psm 11', config_pytesseract_legacy='--oem 0 --psm 11', lineno_gap=5, draw_images=False, disable_progressbar=False):
    # Parameters
    path_out_ocr = path_words / 'ocr'
    path_out_annotated = path_words / 'annotated_words'
    if force_new_ocr:
        makeDirs(path_out_ocr, replaceDirs=True)
        makeDirs(path_out_annotated, replaceDirs=True)
    else:
        makeDirs(path_out_ocr, replaceDirs='overwrite')
        makeDirs(path_out_annotated, replaceDirs='overwrite')

    # Parse dict
    pdfName, pageNumbers = pdfNameAndPage

    # Open pdf
    pdfPath = path_pdfs / f'{pdfName}.pdf'
    doc:fitz.Document = fitz.open(pdfPath)

    # Connect to reader endpoint
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, method_whitelist=False, status_forcelist=[500, 502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))

    # Get words from appropriate pages
    for pageNumber in tqdm(pageNumbers, position=1, leave=False, desc='Words | Looping over pages', total=len(pageNumbers), smoothing=0.2, disable=disable_progressbar):        # pageNumber = pageNumbers[0]
        # Words | Page | Load page
        page:fitz.Page = doc.load_page(page_id=pageNumber)
        pageName = f'{pdfName}{split_stub_page}{pageNumber}'
        path_out = path_words / f'{pageName}.pq'
        
        # Words | Page | Skip if already parsed
        if os.path.exists(path_out):
            continue

        # Words | Page | Get words from pdf directly
        textPage = page.get_textpage(flags=fitz.TEXTFLAGS_WORDS)
        words = textPage.extractWORDS()

        # Words | Page | Generate images for OCR
        page_image = page.get_pixmap(dpi=dpi_ocr, alpha=False, colorspace=fitz.csGRAY)
        img = np.frombuffer(page_image.samples, dtype=np.uint8).reshape(page_image.height, page_image.width, page_image.n)
        _, img_array = cv.threshold(np.array(img), 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        img = Image.fromarray(img_array)

        # Words | Page | OCR if necessary
        if len(words) == 0:
            textDfs = []

            # Words | Page | OCR | Deskew if required
            if path_data_skew:
                path_skew_file = path_data_skew / f'{pageName}.txt'
                if os.path.exists(path_skew_file):
                    with open(path_skew_file, 'r') as f:
                        skewAngle = float(f.readline().strip('\n'))
                        img = img.rotate(skewAngle, expand=True, fillcolor='white', resample=Image.Resampling.BICUBIC)
                        img_array = np.array(img)

            # Words | Page | OCR | PyTesseract Fast
            pathSaved = path_out_ocr / f'{pageName}_tesseractFast.pq'
            if not os.path.exists(pathSaved):
                text:pd.DataFrame = pytesseract.image_to_data(img, lang=languages_tesseract, config=config_pytesseract_fast, output_type=pytesseract.Output.DATAFRAME)
                text.to_parquet(path=pathSaved)
            else:
                text = pd.read_parquet(pathSaved)
            wordsDf = text.loc[text['conf'] > 30, ['left', 'top', 'width', 'height', 'text', 'conf']].assign(ocrLabel='tesseract-fast')
            wordsDf['right'] = wordsDf['left'] + wordsDf['width']
            wordsDf['bottom'] = wordsDf['top'] + wordsDf['height']
            wordsDf = wordsDf.drop(columns=['width', 'height'])
            textDfs.append(wordsDf)
            
            # Get words from page | OCR | From Image | PyTesseract Legacy
            pathSaved = path_out_ocr / f'{pageName}_tesseractLegacy.pq'
            if not os.path.exists(pathSaved):
                text:pd.DataFrame = pytesseract.image_to_data(img, lang=languages_tesseract, config=config_pytesseract_legacy, output_type=pytesseract.Output.DATAFRAME)
                text.to_parquet(path=pathSaved)
            else:
                text = pd.read_parquet(pathSaved)
            wordsDf = text.loc[text['conf'] > 30, ['left', 'top', 'width', 'height', 'text', 'conf']].assign(ocrLabel='tesseract-legacy')
            wordsDf['right'] = wordsDf['left'] + wordsDf['width']
            wordsDf['bottom'] = wordsDf['top'] + wordsDf['height']
            wordsDf = wordsDf.drop(columns=['width', 'height'])
            textDfs.append(wordsDf)

            # Get words from page | OCR | From Image | EasyOCR
            pathSaved = path_out_ocr / f'{pageName}_easyocr.pkl'
            if not os.path.exists(pathSaved):
                with BytesIO() as buffer:
                    pickle.dump(img_array, buffer)
                    img_array_bytes = buffer.getvalue()
                response = session.post(url=reader_endpoint, files={'img_array_pkl': img_array_bytes}, timeout=180)
                try:
                    text = pickle.loads(response.content)
                except pickle.UnpicklingError as e:
                    print(pathSaved)
                    raise e
                # try:
                #     text = reader.readtext(image=img_array, batch_size=60, detail=1)
                # except OutOfMemoryError:        # type: ignore
                #     page_image_d4 = page.get_pixmap(dpi=int(dpi_ocr/4), alpha=False, colorspace=fitz.csGRAY)
                #     img_d4 = np.frombuffer(page_image_d4.samples, dtype=np.uint8).reshape(page_image_d4.height, page_image_d4.width, page_image_d4.n)
                #     _, img_array_d4 = cv.threshold(np.array(img_d4), 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
                #     text_d4 = reader.readtext(image=img_array_d4, batch_size=60, detail=1)
                    
                #     text = []
                #     for d4 in text_d4:      # d4 = text_d4[0]
                #         bbox = d4[0]
                #         bbox = [[x*4 for x in L] for L in bbox]
                #         d1 = (bbox, *d4[1:])
                #         text.append(d1)


                with open(pathSaved, 'wb') as file:
                    pickle.dump(text, file)
            else:
                with open(pathSaved, 'rb') as file:
                    text = pickle.load(file)
            text = [{'left': el[0][0][0], 'right': el[0][1][0], 'top': el[0][0][1], 'bottom': el[0][2][1], 'text': el[1], 'conf': el[2]*100} for el in text]
            text:pd.DataFrame = pd.DataFrame.from_records(text).assign(ocrLabel="easyocr")
            wordsDf = text.loc[text['conf'] > 30]
            textDfs.append(wordsDf)

            # Combine
            df = pd.concat(textDfs)
            df[['left', 'right', 'top', 'bottom']] = df[['left', 'right', 'top', 'bottom']].round(0).astype(int)
            df['text_sparse_len'] = df['text'].str.replace(r'[^a-zA-Z0-9À-ÿ\.\(\) ]', '', regex=True).str.len()
            df = df.loc[df['text_sparse_len'] >= 1].drop(columns='text_sparse_len').reset_index(drop=True)

            # Detect intersection
            easyocr_boxes = set(df.loc[df['ocrLabel'] == 'easyocr', ['left', 'top', 'right', 'bottom']].itertuples(name='Box'))
            tessfast_boxes = set(df.loc[df['ocrLabel'] == 'tesseract-fast', ['left', 'top', 'right', 'bottom']].itertuples(name='Box'))
            tesslegacy_boxes   = set(df.loc[df['ocrLabel'] == 'tesseract-legacy', ['left', 'top', 'right', 'bottom']].itertuples(name='Box'))

            boxes_fastLegacy_intersect = [BoxIntersect(left=legacyBox.left, right=legacyBox.right, top=legacyBox.top, bottom=legacyBox.bottom, intersect=box_intersects_boxList(box=legacyBox, box_targets=tessfast_boxes), Index=legacyBox.Index) for legacyBox in tesslegacy_boxes]
            tesslegacy_extraboxes = [box for box in boxes_fastLegacy_intersect if not box.intersect]
            tess_boxes = tessfast_boxes.union(set(tesslegacy_extraboxes))
            
            boxes_tess_intersect = [BoxIntersect(left=tessBox.left, right=tessBox.right, top=tessBox.top, bottom=tessBox.bottom, intersect=box_intersects_boxList(box=tessBox, box_targets=easyocr_boxes), Index=tessBox.Index) for tessBox in tess_boxes]
            tess_extraboxes = [box for box in boxes_tess_intersect if not box.intersect]

            # Combine words
            indexesToKeep = [box.Index for box in easyocr_boxes.union(set(tess_extraboxes))]
            textDf = df.iloc[indexesToKeep]
            textDf = textDf.reindex(columns=['top', 'left', 'bottom', 'right', 'text', 'conf', 'ocrLabel'])

            # Crop bboxes
            img_blackIs1 = np.where(np.array(img) == 0, 1, 0)
            textDf['toKeep'] = True
            textDf = textDf.apply(lambda row: tighten_word_bbox(img_blackIs1=img_blackIs1, bboxRow=row), axis=1)
            textDf = textDf[textDf['toKeep']].drop(columns='toKeep')

            # Define block, line and word numbers
            textDf = textDf.sort_values(by=['top', 'left']).reset_index(drop=True)
            textDf['wordno'], textDf['lineno'], previous_bottom, line_counter, word_counter = 0, 0, 0, 0, 0
            textDf['blockno'] = 0
            
            heightCol = textDf['bottom'] - textDf['top']
            maxHeight = heightCol.mean() + 2 * heightCol.std()
            textDf = textDf.loc[heightCol <= maxHeight]

            for idx, row in textDf.iterrows():
                if ((row['bottom'] - previous_bottom) > lineno_gap) and (row['top'] > previous_bottom):
                     word_counter = 0
                     line_counter += 1
                     previous_bottom = row['bottom']
                else:
                    word_counter += 1
                textDf.loc[idx, ['lineno', 'wordno']] = line_counter, word_counter

            # Harmonise top/bottom by line (bit wonky after crop/source combination otherwise)
            textDf = harmonise_bbox_height(textDf)
            
        else:
            textDf = pd.DataFrame.from_records(words, columns=['left', 'top', 'right', 'bottom', 'text', 'blockno', 'lineno', 'wordno']).assign(**{'ocrLabel': 'pdf', 'conf':100})
            textDf['text_sparse_len'] = textDf['text'].str.replace(r'[^a-zA-Z0-9À-ÿ\.\(\) ]', '', regex=True).str.len()
            textDf = textDf.loc[(textDf['text_sparse_len'] >= 2) | (textDf['text'] == '.') | (textDf['text'].str.isalnum())].drop(columns='text_sparse_len').reset_index(drop=True)
            textDf[['left', 'right', 'top', 'bottom']] = (textDf[['left', 'right', 'top', 'bottom']] * (dpi_ocr / dpi_pymupdf)).round(0).astype(int)

            # Crop bboxes
            img_blackIs1 = np.where(np.array(img) == 0, 1, 0)
            textDf['toKeep'] = True
            textDf = textDf.apply(lambda row: tighten_word_bbox(img_blackIs1=img_blackIs1, bboxRow=row), axis=1)
            textDf = textDf[textDf['toKeep']].drop(columns='toKeep')
            
            # Harmonise top/bottom by line (bit wonky after crop/source combination otherwise)
            textDf = harmonise_bbox_height(textDf)
        
        # Get words from page | Save
        textDf.to_parquet(path_out)

        # Visualise words
        if draw_images:
            img_overlay = Image.new('RGBA', img.size, (0,0,0,0))
            img_annot = ImageDraw.Draw(img_overlay, mode='RGBA')
            colors = {
                'easyocr': (255,228,181, int(0.6*255)),
                'pdf': (255, 162, 0, int(0.5*255)),
                'other': (0, 128, 0, int(0.5*255))
            }
            font_label = ImageFont.truetype('arial.ttf', size=40)
            font_line = ImageFont.truetype('arial.ttf', size=20)
        
            for box in textDf.itertuples('box'):      
                color = colors[box.ocrLabel] if box.ocrLabel in ['easyocr', 'pdf'] else colors['other']
                img_annot.rectangle(xy=(box.left, box.top, box.right, box.bottom), fill=color)

            lineDf = textDf.sort_values(by=['lineno', 'wordno']).drop_duplicates('lineno', keep='first')
            lineLeft = max(lineDf['left'].min() - 50, 20)

            for line in lineDf.itertuples('line'):      # line = next(lineDf.itertuples('line'))
                img_annot.text(xy=(lineLeft, line.top), text=str(line.lineno), anchor='la', fill='black', font=font_line)


            img_annot.text(xy=(img.size[0] // 4 * 1, img.size[1] // 15 * 14), text='easyocr', anchor='ld', fill=colors['easyocr'], font=font_label)
            img_annot.text(xy=(img.size[0] // 4 * 2, img.size[1] // 15 * 14), text='pdf', anchor='ld', fill=colors['pdf'], font=font_label)
            img_annot.text(xy=(img.size[0] // 4 * 3, img.size[1] // 15 * 14), text='tesseract', anchor='ld', fill=colors['other'], font=font_label)

            img = Image.alpha_composite(img.convert('RGBA'), img_overlay)
            img.convert('RGB').save(path_out_annotated / f'{pageName}.png')

# Line-level features | Helpers
def yolo_to_fitzBox(yoloPath, mediabox, page):
    targetWidth = mediabox.width if page.rotation in [0, 180] else mediabox.height
    targetHeight = mediabox.height if page.rotation in [0, 180] else mediabox.width
    fitzBoxes = []
    with open(yoloPath, 'r') as yoloFile:
        for annotationLine in yoloFile:
            cat, xc, yc, w, h, *conf = [float(string.strip('\n')) for string in annotationLine.split(' ')]
            
            x0 = (xc - w/2) * targetWidth
            x1 = (xc + w/2) * targetWidth
            y0 = (yc - h/2) * targetHeight
            y1 = (yc + h/2) * targetHeight

            fitzBoxes.append([x0, y0, x1, y1])
    return fitzBoxes
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
def index_to_bboxCross(index, mins, maxes):
    return ((mins <= index) & (maxes >=index)).sum()
def convert_01_array_to_visual(array, invert=False, width=40, luminosity_max=240) -> np.array:
    luminosity = (1 - array) * luminosity_max if invert else array * luminosity_max
    luminosity = luminosity.round(0).astype(np.uint8)
    luminosity = np.expand_dims(luminosity, axis=1)
    luminosity = np.broadcast_to(luminosity, shape=(luminosity.shape[0], width))
    return luminosity
def calculate_image_similarity(img1, img2):
    min_dim0 = min(img1.height, img2.height)
    min_dim1 = min(img1.width, img2.width)

    maxDiscrepancy = max([img1.height-min_dim0, img2.height-min_dim0, img1.width-min_dim1, img2.width-min_dim1])

    if maxDiscrepancy > 20:
        return 0

    hash1 = imagehash.average_hash(img1)
    hash2 = imagehash.average_hash(img2)

    distance = hash1 - hash2
    score = 1 - (distance / len(hash1))

    return score
def adjust_initialBoundaries_to_betterBoundariesB(arrayInitial, arrayBetter):
    arrayBetter_idx = 0
    startBetter, endBetter = arrayBetter[arrayBetter_idx]
    endBetter_max = arrayBetter[:,1].max()

    for arrayInitial_idx, (startInitial, endInitial) in enumerate(arrayInitial):     # startInitial, endInitial = arrayInitial[0]
        # Advance to next better boundary: if better boundary is fully to the left of initial boundary
        while (startInitial > endBetter):
            arrayBetter_idx += 1
            try:
                startBetter, endBetter = arrayBetter[arrayBetter_idx]
            except IndexError:
                break

            if (startInitial > endBetter_max):
                break
        
        # Advance to next initial boundary: if better boundary is fully to the right of initial boundary
        if (endInitial < startBetter):
            continue

        # Update to better: if initial boundary overlaps better boundary
        else:
            arrayInitial[arrayInitial_idx] = (startBetter, endBetter)
    
    return arrayInitial
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def pageWords_to_tableWords(path_words, tableName, metaInfo, tableSplitString='_t'):
    # Technicalities
    pd.set_option('mode.copy_on_write', True)
    coordinateColumns = ['left', 'right', 'top', 'bottom']
    tableCoords = metaInfo['table_coords']
    
    # Load words
    pageName = tableName.split(tableSplitString)[0]
    wordsDf_page = pd.read_parquet(path_words / f'{pageName}.pq').drop_duplicates()

    # Rescale OCR coordinates to PDF coordinates
    wordsDf_page[coordinateColumns] = wordsDf_page[coordinateColumns] * (metaInfo['dpi_pdf'] / metaInfo['dpi_words'])

    # Reduce to table bbox
    wordsDf_table = wordsDf_page.loc[(wordsDf_page['top'] >= tableCoords['y0']) & (wordsDf_page['left'] >= tableCoords['x0']) & (wordsDf_page['bottom'] <= tableCoords['y1']) & (wordsDf_page['right'] <= tableCoords['x1'])]

    # Rescale PDF coordinates to padded table/model coordinates
    wordsDf_table[['left', 'right']] = wordsDf_table[['left', 'right']] - tableCoords['x0']
    wordsDf_table[['top', 'bottom']] = wordsDf_table[['top', 'bottom']] - tableCoords['y0']
    wordsDf_table[coordinateColumns] = wordsDf_table[coordinateColumns] * (metaInfo['dpi_model'] / metaInfo['dpi_pdf']) + metaInfo['padding_model']
    wordsDf_table[coordinateColumns] = wordsDf_table[coordinateColumns].round(0).astype(int)

    # Return df
    return wordsDf_table

def pdfToImages(path_input, path_out, image_format='.jpg', sample_size_pdfs=100, dpi=150, keep_transparency=False, n_workers=-1,
                deskew=True, verbosity=logging.INFO, replace_dirs='warn', exclude_list=[]):
    ''' Updated version of tabledetect one, allows for exclusion list'''
    # Parse parameters
    if not image_format.startswith('.'): image_format = f'.{image_format}'
    path_out = Path(path_out)
    path_out_pages = path_out / 'pages_images'
    path_out_pdfs  = path_out / 'pdfs'
    path_meta      = path_out / 'meta'

    exclude_list = set(exclude_list)

    # Create folders
    makeDirs(path_out_pages, replaceDirs=replace_dirs)
    makeDirs(path_out_pdfs, replaceDirs=replace_dirs)
    makeDirs(path_meta, replaceDirs='overwrite')
    makeDirs(path_meta / 'skewAngles', replaceDirs=replace_dirs)

    # Logging
    logger = logging.getLogger(__name__); logger.setLevel(verbosity)

    # Draw sample
    pdf_donorPool = [pdfEntry.path for pdfEntry in os.scandir(path_input) if pdfEntry.name not in exclude_list]
    pdf_sample = random.sample(pdf_donorPool, k=sample_size_pdfs)

    # Split
    def splitPdf(pdfPath, path_output):
        try:
            # Load pdf
            with open(pdfPath) as pdfFile:
                doc = fitz.open(pdfFile)
            filename = os.path.basename(pdfPath).replace('.pdf', '')

            # Convert pages to images
            for page in doc:
                # Get pixmap
                pixmap = page.get_pixmap(alpha=keep_transparency, dpi=dpi, colorspace=fitz.csGRAY)
                pageName = f"{filename}-p{page.number}"

                # Threshold to binary
                img = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
                _, img = cv.threshold(np.array(img), 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

                if deskew:
                    # Deskew
                    skewAngle = determine_skew(img, max_angle=5, min_deviation=0.5, num_peaks=40, sigma=3)
                    img = Image.fromarray(img)
                    if skewAngle:
                        img = img.rotate(skewAngle, expand=True, fillcolor='white', resample=Image.Resampling.BICUBIC)
                        with open(path_meta / 'skewAngles' / f'{pageName}.txt', 'w') as f:
                            f.write(f'{skewAngle:.2f}')
               
                # Save
                img.save(path_output / f'{pageName}{image_format}')

            # Copy pdf to destination
            shutil.copyfile(src=pdfPath, dst=path_out_pdfs / f'{filename}.pdf')
            return True
        
        except fitz.fitz.FileDataError:
            return False
        
    if n_workers == 1:
        start = time.time()
        results = [splitPdf(pdfPath=pdfPath, path_output=path_out_pages) for pdfPath in tqdm(pdf_sample, desc='Splitting PDF files')]
        logger.info(f'Splitting {len(results)} PDFs took {time.time()-start:.0f} seconds. {results.count(False)} PDFs raised an error.')

    else:
        from joblib import Parallel, delayed
        start = time.time()
        results = Parallel(n_jobs=n_workers, backend="loky", verbose=verbosity//2)(delayed(splitPdf)(pdfPath, path_out_pages) for pdfPath in pdf_sample)
        logger.info(f'Splitting {len(results)} PDFs took {time.time()-start:.0f} seconds. {results.count(False)} PDFs raised an error.')

    # Report meta
    with open(path_meta / 'pdfToImages.json', 'w') as f:
        json.dump(dict(dpi=dpi, image_format=image_format, sample_size_pdfs=sample_size_pdfs, pdfs_parsed=len(results), pdfs_failed=results.count(False)), fp=f)
def yolo_to_pilbox(yoloPath, targetImage):
        targetWidth, targetHeight = targetImage.size
        pilBoxes = []

        with open(yoloPath, 'r') as yoloFile:
            for annotationLine in yoloFile:         # annotationLine = yoloFile.readline()
                cat, xc, yc, w, h, conf = [float(string.strip('\n')) for string in annotationLine.split(' ')]
                
                x0 = round((xc - w/2) * targetWidth)
                x1 = round((xc + w/2) * targetWidth)
                y0 = round((yc - h/2) * targetHeight)
                y1 = round((yc + h/2) * targetHeight)

                pilBoxes.append([x0, y0, x1, y1])
        return pilBoxes
def detect_to_croppedTable(path_labels, path_images, path_out, padding, image_format, max_samples=None, n_workers=1, replace_dirs='warn', verbosity=logging.INFO):
    # Parameters
    path_out_tables_images = path_out / 'tables_images'
    path_meta = path_out / 'meta'
    makeDirs(path_meta, replaceDirs='overwrite')
    makeDirs(path_out_tables_images, replaceDirs=replace_dirs)

    def __label_to_crop(pageName):
        img = Image.open(path_images / f'{pageName}{image_format}')
        table_bboxes = yolo_to_pilbox(yoloPath=path_labels / f'{pageName}.txt', targetImage=img)        # x0 y0 x1 y1
        
        for idx, bbox in enumerate(table_bboxes):
            img_cropped = img.crop(bbox)
            img_padded = Image.new(img_cropped.mode, (img_cropped.width+padding*2, img_cropped.height+padding*2), 255)
            img_padded.paste(img_cropped, (padding, padding))
            img_padded.save(path_out_tables_images / f'{pageName}_t{idx}{image_format}')

    # Determine approximate number of pages to parse
    pageNames_all = [os.path.splitext(entry.name)[0] for entry in os.scandir(path_labels)]
    pageNames = []
    if max_samples:
        counter = 0
        for pageName in pageNames_all:
            pageNames.append(pageName)
            with open(path_labels / f'{pageName}.txt', 'r') as f:
                counter += len(f.readlines())
            if counter >= max_samples:
                break
    else:
        pageNames = pageNames_all

    # Crop tables
    if n_workers == 1:
        for pageName in tqdm(pageNames, desc='Cropping detected tables'):
            __label_to_crop(pageName)
    else:
        results = Parallel(n_jobs=n_workers, backend='loky', verbose=verbosity // 2)(delayed(__label_to_crop)(pageName) for pageName in pageNames)

    # Report meta
    with open(path_meta / 'detectToCroppedTables.json', 'w') as f:
        json.dump(dict(padding=padding, image_format=image_format), fp=f)
def extractSample(path_data, desired_sample_size, respect_existing_sample=False, active_learning=False,
                  path_labels=None, path_images=None, path_annotated_images=None, path_out_sample=None, n_workers=-1, replace_dirs='warn', image_format='.png'):
    # Parameters
    path_labels = path_labels or path_data / 'predictions_separatorLevel' / 'labels_rows'
    path_images = path_images or path_data / 'tables_images'
    path_out_sample = path_out_sample or path_data / 'sample'

    # Get existing sample info
    if respect_existing_sample:
        print(f'Respecting existing sample at {path_out_sample / "labels"}')
        labelFiles = os.listdir(path_out_sample / 'labels')

    else:
        print(f'Extracting sample of {desired_sample_size} from {len(os.listdir(path_labels))}')

        # Sample
        try:
            labelFiles = random.sample(os.listdir(path_labels), k=desired_sample_size)
        except ValueError:
            labelFiles = os.listdir(path_labels)
            print(f'Sample size reduced to {len(labelFiles)} (total number of tables processed)')

    makeDirs(path_out_sample / 'images', replaceDirs=replace_dirs)
    makeDirs(path_out_sample / 'labels', replaceDirs=replace_dirs)
    makeDirs(path_out_sample / 'annotated_initial', replaceDirs=replace_dirs)

    # Save to separate folder
    errors = 0
    for labelFile in tqdm(labelFiles, desc='Utils | Copying sample files'):
        tableName = os.path.splitext(labelFile)[0]
        if os.path.exists(path_labels / f'{tableName}.xml') and os.path.exists(path_images / f'{tableName}{image_format}'):
            shutil.copyfile(src=path_labels / f'{tableName}.xml', dst=path_out_sample / 'labels' / f'{tableName}.xml')
            shutil.copyfile(src=path_images / f'{tableName}{image_format}', dst=path_out_sample / 'images' / f'{tableName}{image_format}')
        else:
            errors += 1
            print(f'{tableName} not found')
    
    if errors:
        print(f'Files missing when extracting sample: {errors}')

    # Visualise annotations
    tabledetect.utils.visualise_annotation(path_images=path_out_sample / 'images', path_labels=path_out_sample / 'labels', path_output=path_out_sample / 'annotated_initial', annotation_type='tableparse-msft', split_annotation_types=True, as_area=True, n_workers=n_workers)
def splitSample(path_data, split_size, respect_existing_sample=False, path_sample=None, image_format='.png', replace_dirs='warn'):
    # Parameters
    path_sample = path_sample or path_data / 'sample'

    if respect_existing_sample:
        splitFolders = [splitFolder.replace('\\', '') for splitFolder in glob('sample_split*/', root_dir=path_sample)]
        errors = 0
        for splitFolder in splitFolders:        # splitFolder = next(iter(splitFolders))
            path_split = path_sample / splitFolder
            counter = splitFolder.split('_split')[-1]
            sample = os.scandir(path_split / 'labels')

            filenames = [os.path.splitext(entry.name)[0] for entry in sample]
            if filenames:
                makeDirs(path_sample / f'sample_split{counter}' / 'images', replaceDirs=replace_dirs)
                makeDirs(path_sample / f'sample_split{counter}' / 'labels', replaceDirs=replace_dirs)
                for filename in tqdm(filenames, desc=f'Copying files into split {counter}', leave=False):
                    if os.path.exists(path_sample / 'images' / f'{filename}{image_format}') and os.path.exists(path_sample / 'labels' / f'{filename}.xml'):
                        shutil.copyfile(src=path_sample / 'images' / f'{filename}{image_format}', dst=path_sample / f'sample_split{counter}' / 'images' / f'{filename}{image_format}')
                        shutil.copyfile(src=path_sample / 'labels' / f'{filename}.xml', dst=path_sample / f'sample_split{counter}' / 'labels' / f'{filename}.xml')
                    else:
                        errors += 1
                        tqdm.write(f'{filename} not found')
            
        if errors:
            print(f'Files missing when splitting sample: {errors}')
    else:
        # Get list of labels
        labels = list(os.scandir(path_sample / 'labels'))
        counter = 1

        while labels:
            try:
                sample = random.sample(labels, k=split_size)
            except ValueError:
                sample = labels

            filenames = [os.path.splitext(entry.name)[0] for entry in sample]
            if filenames:
                makeDirs(path_sample / f'sample_split{counter}' / 'images', replaceDirs=replace_dirs)
                makeDirs(path_sample / f'sample_split{counter}' / 'labels', replaceDirs=replace_dirs)
                for filename in tqdm(filenames, desc=f'Copying files into split {counter}', leave=False):
                    shutil.copyfile(src=path_sample / 'images' / f'{filename}{image_format}', dst=path_sample / f'sample_split{counter}' / 'images' / f'{filename}{image_format}')
                    shutil.copyfile(src=path_sample / 'labels' / f'{filename}.xml', dst=path_sample / f'sample_split{counter}' / 'labels' / f'{filename}.xml')
            counter += 1
            labels = list(set(labels) - set(sample))
def trainValTestSplit(path_data, existing_sample_paths:list=[], trainRatio=0.8, valRatio=0.1, maxVal=None, maxTest=None, replace_dirs='warn', image_format='.png', prefix=''):
    # Grab items
    items = [os.path.splitext(entry.name)[0] for entry in os.scandir(path_data / 'targets_lineLevel')]
    itemCount = len(items)
    splits = ['train', 'val', 'test']
    dataSplit = {}

    # Determine train/val/test split
    if existing_sample_paths:
        items = set(items)
        existing_sample_paths = [Path(existing_sample_path) for existing_sample_path in existing_sample_paths]
        for split in splits:
            item_scanners = [os.scandir(existing_sample_path / 'splits' / split / 'targets_lineLevel') for existing_sample_path in existing_sample_paths]
            split_items = set([os.path.splitext(entry.name)[0] for scanner in item_scanners for entry in scanner])
            dataSplit[split] = items.intersection(split_items)
            items = items - split_items
            
        # Calculate indices for remaining items
        # .. | Get desired and current counts per split
        desiredCountPerSplit = dict(train=round(trainRatio * itemCount), val=round(valRatio * itemCount))
        desiredCountPerSplit['test'] = itemCount - sum(desiredCountPerSplit.values())

        if maxVal:
            desiredCountPerSplit['val'] = min(maxVal, desiredCountPerSplit['val'])
        if maxTest:
            desiredCountPerSplit['test'] = min(maxTest, desiredCountPerSplit['test'])

        currentCountPerSplit = {key: len(value) for key, value in dataSplit.items()}
        
        # .. | Fill up from test > val > train
        def fillUp(split, dataSplit, items):
            items = list(items)
            random.shuffle(items)
            required_addition = desiredCountPerSplit[split] - currentCountPerSplit[split]
            additions = set(items[:required_addition])
            dataSplit[split] = dataSplit[split].union(additions)
            items = set(items) - set(additions)

            return dataSplit, items

        if (desiredCountPerSplit['test'] > currentCountPerSplit['test']) and items:
            dataSplit, items = fillUp('test', dataSplit, items)
        if (desiredCountPerSplit['val'] > currentCountPerSplit['val']) and items:
            dataSplit, items = fillUp('val', dataSplit, items)
        if items:
            dataSplit['train'] = dataSplit['train'].union(items)

        assert itemCount == sum([len(value) for value in dataSplit.values()])
        
    else:
        random.shuffle(items)
        dataSplit['train'], dataSplit['val'], dataSplit['test'] = np.split(items, indices_or_sections=[int(len(items)*trainRatio), int(len(items)*(trainRatio+valRatio))])

    # Copy files to train/val/test split
    for subgroup in dataSplit:      # subgroup = list(dataSplit.keys())[0]
        destPath = path_data / 'splits' / subgroup
        makeDirs(destPath / 'tables_images', replaceDirs=replace_dirs)
        makeDirs(destPath / 'targets_lineLevel', replaceDirs=replace_dirs)
        makeDirs(destPath / 'features_lineLevel', replaceDirs=replace_dirs)
        makeDirs(destPath / 'meta_lineLevel', replaceDirs=replace_dirs)

        for item in tqdm(dataSplit[subgroup], desc=f"{prefix}Copying from root > {subgroup}"):        # item = dataSplit[subgroup][0]
            _ = shutil.copyfile(src=path_data / 'tables_images'   / f'{item}{image_format}',  dst=destPath / 'tables_images'   / f'{item}{image_format}')
            _ = shutil.copyfile(src=path_data / 'targets_lineLevel'   / f'{item}.json', dst=destPath / 'targets_lineLevel'   / f'{item}.json')
            _ = shutil.copyfile(src=path_data / 'features_lineLevel' / f'{item}.json', dst=destPath / 'features_lineLevel' / f'{item}.json')
            _ = shutil.copyfile(src=path_data / 'meta_lineLevel'     / f'{item}.json', dst=destPath / 'meta_lineLevel'     / f'{item}.json')

    # Copy files to all
    makeDirs(path_data / 'splits' / 'all', replaceDirs=replace_dirs)
    for subgroup in tqdm(dataSplit, desc=f'{prefix}Copying from subgroups > all'):
        srcPath = path_data / 'splits' / subgroup
        _ = shutil.copytree(src=srcPath, dst=path_data / 'splits' / 'all', dirs_exist_ok=True)
    shutil.rmtree(path_data / 'tables_images')
    shutil.rmtree(path_data / 'targets_lineLevel')
    shutil.rmtree(path_data / 'features_lineLevel')
    shutil.rmtree(path_data / 'meta_lineLevel')
def fanOutBySplit(path_data, fanout_dirs, source='all', destinations=['train', 'val', 'test'], example_dir='features_lineLevel', prefix=None, replace_dirs='warn'):
    path_data = Path(path_data)
    path_source = path_data / source
    extensions = {fanout_dir: os.path.splitext(next(os.scandir(path_source / fanout_dir)).name)[-1] for fanout_dir in fanout_dirs}

    for destination in destinations:
        path_dest = path_data / destination
        sample = [os.path.splitext(entry.name)[0] for entry in os.scandir(path_dest / example_dir)]

        for fanout_dir in tqdm(fanout_dirs, desc=f'{prefix}Copying new directories from all > {destination}'):
            extension = extensions[fanout_dir]
            makeDirs(path_dest / fanout_dir, replaceDirs=replace_dirs)
            for filename in tqdm(sample, desc=f'{prefix} Files', position=-1, leave=False):
                try:
                    _ = shutil.copyfile(src=path_source / fanout_dir / f'{filename}{extension}', dst=path_dest / fanout_dir / f'{filename}{extension}')
                except FileNotFoundError:
                    pass

    
def generate_training_sample(path_pdfs, path_out, 
                             sample_size_pdfs, path_model_detect, path_model_parse_line, path_model_parse_separator, 
                             respect_existing_sample=False, sample_size_tables=None, desired_sample_size=None, exclude_pdfs_by_image_folder_list=[], replace_dirs='warn',
                             threshold_detect=0.7, padding_tables=40, 
                             split_stub='-p', active_learning=True, deskew=True, image_format='.png',
                             n_workers=-1, verbosity=logging.INFO):
    '''
        sample_size_pdfs: number of pdfs to parse for tables
        sample_size_tables: maximum amount of tables in final sample'''
    
    # Parse paths
    path_pdfs = Path(path_pdfs); path_out = Path(path_out)
    exclude_pdfs_by_image_folder_list = [Path(folder) for folder in exclude_pdfs_by_image_folder_list]

    # Optional: Get list of pdfs to exclude from sampling
    pdf_exclude_list = []
    if exclude_pdfs_by_image_folder_list:
        for folder in exclude_pdfs_by_image_folder_list:
            files = os.scandir(folder)
            pdfs = set([file.name.split(split_stub)[0] for file in files])
            pdf_exclude_list = pdf_exclude_list + list(pdfs)
    pdf_exclude_list = set(pdf_exclude_list)
    pdf_exclude_list = [f'{name}.pdf' for name in pdf_exclude_list]

    # Sample pdfs
    makeDirs(path_out / 'meta', replaceDirs=replace_dirs)
    pdfToImages(path_input=path_pdfs, path_out=path_out, image_format='.png', sample_size_pdfs=sample_size_pdfs, replace_dirs=replace_dirs, n_workers=n_workers, deskew=deskew)

    # Detect tables
    makeDirs(path_out / 'tables_bboxes', replaceDirs=replace_dirs)
    tabledetect.detect_table(path_weights=path_model_detect, path_input=path_out / 'pages_images', path_output=path_out / 'temp', image_format='.png', threshold_confidence=threshold_detect, save_visual_output=False)
    shutil.copytree(src=path_out / 'temp' / 'out' / 'table-detect' / 'labels', dst=path_out / 'tables_bboxes', dirs_exist_ok=True)
    shutil.rmtree(path_out / 'temp')
    
    # Generate cropped+padded tables
    detect_to_croppedTable(path_labels=path_out / 'tables_bboxes', path_images=path_out / 'pages_images', path_out=path_out, padding=padding_tables, image_format=image_format, n_workers=n_workers, max_samples=sample_size_tables, replace_dirs=replace_dirs, verbosity=verbosity)

    # Preprocess line-level features
    process.preprocess_lineLevel(path_images=path_out / 'tables_images', path_pdfs=path_out / 'pdfs', path_out=path_out, path_data_skew=path_out / 'meta' / 'skewAngles', replace_dirs=replace_dirs, verbosity=verbosity, n_workers=n_workers)

    # Apply line-level model and preprocess separator-level features
    process.preprocess_separatorLevel(path_model_line=path_model_parse_line, path_data=path_out, replace_dirs=replace_dirs)

    # Apply separator-level model and process out
    process.predict_and_process(path_model_file=path_model_parse_separator, path_data=path_out, replace_dirs=replace_dirs, out_data=False, out_images=False, out_labels_rows=True)

    # Zip sample for annotators
    extractSample(path_data=path_out, desired_sample_size=desired_sample_size, replace_dirs=replace_dirs, active_learning=active_learning, n_workers=n_workers, respect_existing_sample=respect_existing_sample)

    # Split sample
    splitSample(path_data=path_out, split_size=1000, respect_existing_sample=respect_existing_sample, replace_dirs=replace_dirs)

def train_models(name, info_samples, path_out_data, path_out_model,
                    existing_sample_paths=[],
                    epochs_line=100, epochs_separator=100, max_lr_line=0.2, max_lr_separator=0.2, batch_size=4,
                    replace_dirs='warn', image_format='.png', padding=40, device='cuda',
                    n_workers=-1, verbosity=logging.INFO):
    # Parameters
    if not isinstance(info_samples, list):
        info_samples = list(info_samples)
    path_out_data = Path(path_out_data); path_out_model = Path(path_out_model)
    image_format = ensure_startsWithDot(image_format)
    prefix = 'Train models | '

    # Derived
    path_data_project = path_out_data / name
    path_model_line = path_out_model / f'{name}_line'
    path_model_separator = path_out_model / f'{name}_separator'

    # # Collect files
    # makeDirs(path_data_project, replaceDirs=replace_dirs)
    # makeDirs(path_data_project / 'labels', replaceDirs=replace_dirs)
    # makeDirs(path_data_project / 'tables_images', replaceDirs=replace_dirs)
    # makeDirs(path_data_project / 'tables_bboxes', replaceDirs=replace_dirs)
    # makeDirs(path_data_project / 'pdfs'  , replaceDirs=replace_dirs)
    # makeDirs(path_data_project / 'skewAngles'  , replaceDirs=replace_dirs)

    # for sample in tqdm(info_samples, desc=f'{prefix}Gathering data'):
    #     sample['path_root'] = Path(sample['path_root'])
    #     shutil.copytree(src=sample['path_root'] / 'labels_tableparse', dst=path_data_project / 'labels', dirs_exist_ok=True)
    #     shutil.copytree(src=sample['path_root'] / 'tables_images', dst=path_data_project / 'tables_images', dirs_exist_ok=True)
    #     shutil.copytree(src=sample['path_root'] / 'tables_bboxes', dst=path_data_project / 'tables_bboxes', dirs_exist_ok=True)
    #     shutil.copytree(src=sample['path_root'] / 'pdfs', dst=path_data_project / 'pdfs'  , dirs_exist_ok=True)
    #     shutil.copytree(src=sample['path_root'] / 'meta' / 'skewAngles', dst=path_data_project / 'skewAngles'  , dirs_exist_ok=True)
    #     shutil.copytree(src=sample['path_root'] / 'words', dst=path_out_data / 'words'  , dirs_exist_ok=True)
    
    # # Harmonize image_format
    # imgPaths = [entry.path for entry in os.scandir(path_out_data / name / 'tables_images')]
    # for imgPath in tqdm(imgPaths, desc=f'{prefix}Harmonizing image formats to {image_format}'):
    #     if os.path.splitext(imgPath)[-1] != image_format:
    #         _ = cv.imwrite(imgPath.replace(os.path.splitext(imgPath)[-1], image_format), cv.imread(imgPath, cv.IMREAD_GRAYSCALE))
    #         os.remove(imgPath)
    #     else:
    #         continue

    # # Preprocess: raw > linelevel
    # print(f'{prefix}Preprocess line-level')
    # process.preprocess_lineLevel(ground_truth=True, padding=padding,
    #                              path_out=path_data_project,
    #                              path_images=path_data_project / 'table_images',
    #                              path_pdfs=path_data_project / 'pdfs',
    #                              path_data_skew=path_data_project / 'skewAngles',
    #                              path_words=path_out_data / 'words',
    #                                 replace_dirs=replace_dirs, verbosity=verbosity, n_workers=n_workers)
    
    # # Split into train/val/test
    # print(f'{prefix}Split into line-level')
    # trainValTestSplit(path_data=path_data_project, existing_sample_paths=existing_sample_paths,
    #                        trainRatio=0.9, valRatio=0.05, maxVal=150, maxTest=150, replace_dirs=replace_dirs)

    # Train line level model
    train.train_lineLevel(path_data_train=path_data_project / 'splits' / 'train', path_data_val=path_data_project / 'splits' / 'val', path_model=path_model_line,
          replace_dirs=replace_dirs, device=device, epochs=epochs_line, max_lr=max_lr_line, batch_size=batch_size,
          disable_weight_visualisation=True)
    evaluate.evaluate_lineLevel(path_model_file=path_model_line / 'model_best.pt', path_data=path_data_project / 'splits' / 'val', device=device, replace_dirs=replace_dirs, batch_size=batch_size)
    
    # # Preprocess: linelevel > separatorlevel
    process.preprocess_separatorLevel(path_model_line=path_model_line / 'model_best.pt', path_data=path_data_project / 'splits' / 'all', path_words=path_out_data / 'words', replace_dirs=replace_dirs, ground_truth=True, draw_images=False, padding=padding, batch_size=batch_size)
    fanOutBySplit(path_data=path_data_project / 'splits', fanout_dirs=['features_separatorLevel', 'targets_separatorLevel'], prefix=prefix, replace_dirs=replace_dirs)
    
    # # Train separator level model
    # train.train_separatorLevel(path_data_train=path_data_project / 'splits' / 'train', path_data_val=path_data_project / 'splits' / 'val', path_model=path_model_separator,
    #                            replace_dirs=replace_dirs, device=device, epochs=epochs_separator, max_lr=max_lr_separator,
    #                            disable_weight_visualisation=True)
    # evaluate.evaluate_separatorLevel(path_model_file=path_model_separator / 'model_best.pt', path_data=path_data_project / 'splits' / 'val', device=device, replace_dirs=replace_dirs)

    # # Process out
    # process.predict_and_process(path_model_file=path_model_separator / 'model_best.pt', path_data=path_data_project / 'splits' / 'val', device=device, replace_dirs=replace_dirs,
    #                 path_pdfs=path_data_project / 'pdfs', path_words=path_out_data / 'words', padding=padding, out_data=True, out_images=True, out_labels_rows=False, ground_truth=True)



if __name__ == '__main__':
    TASK = 'train_models'
    n_workers = -2
    # n_workers = 1
    PATH_TABLEPARSE = Path(r'F:\ml-parsing-project\table-parse-split')

    if TASK == 'generate_sample':
        path_pdfs = r'F:\datatog-data-dev\kb-knowledgeBase\bm-benchmark\be-unlisted\2023-05-16\samples_missing_pdfFiles'
        path_out = r'F:\ml-parsing-project\data\parse_activelearning2_png'
        path_models = Path(r'F:\ml-parsing-project\models')
        path_previous_samples = [r'F:\ml-parsing-project\data\parse_activelearning1_jpg\selected']

        sample_size_pdfs = 300
        desired_sample_size = 4000
        replace_existing_sample = True

        generate_training_sample(path_pdfs=path_pdfs, path_out=path_out, 
                                respect_existing_sample=replace_existing_sample, sample_size_pdfs=sample_size_pdfs, desired_sample_size=desired_sample_size,
                                n_workers=n_workers,
                                path_model_detect=path_models / 'codamo-tabledetect-best.pt', path_model_parse_line=path_models / 'codamo-tableparse-line-best.pt', path_model_parse_separator=path_models / 'codamo-tableparse-separator-best.pt',
                                exclude_pdfs_by_image_folder_list=path_previous_samples, active_learning=False, replace_dirs=True)
        
    if TASK == 'train_models':
        path_sample2 = Path(r"F:\ml-parsing-project\data\parse_activelearning2_png")
        info_samples = [dict(path_root = r"F:\ml-parsing-project\data\parse_activelearning1_harmonized",
                             image_format='.jpg'),
                        dict(path_root = r"F:\ml-parsing-project\data\parse_activelearning2_png",
                             image_format='.png')]
        path_out_data = PATH_TABLEPARSE / 'data'
        path_out_model = PATH_TABLEPARSE / 'models'
        name = 'tableparse_round2'
        existing_sample_paths = [r'F:\ml-parsing-project\data\parse_activelearning1_harmonized']
        epochs_line = 60
        epochs_separator = 80
        max_lr=0.1
        batch_size=4

        train_models(name=name, info_samples=info_samples, path_out_data=path_out_data, path_out_model=path_out_model,
                        epochs_line=epochs_line, epochs_separator=epochs_separator, max_lr_line=max_lr, max_lr_separator=max_lr, batch_size=batch_size,
                        replace_dirs=True, existing_sample_paths=existing_sample_paths,
                        n_workers=n_workers)