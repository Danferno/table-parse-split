# Imports
import os
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
import shutil
from collections import defaultdict
import fitz
import pandas as pd; pd.set_option("mode.copy_on_write", True)
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import cv2 as cv
import json
from lxml import etree
import pickle
import pytesseract
import easyocr
from collections import namedtuple

np.seterr(all='raise')

# Constants
DEBUG = False
PARALLEL = False
SEPARATOR_TYPE = 'wide'
COMPLEXITY_SUFFIX = 'wide'

PATH_ROOT = Path(r"F:\ml-parsing-project\table-parse-split")
PATH_DATA_TABLEDETECT = Path(r"F:\ml-parsing-project\data")
PATH_DATA_PDFS = Path(r"F:\datatog-data-dev\kb-knowledgeBase\bm-benchmark\be-unlisted")
DPI_PYMUPDF = 72
DPI_TARGET = 150
DPI_OCR = 300
PADDING = 40
LINENO_GAP = 5
PRECISION = np.float32

LANGUAGES = 'nld+fra+deu+eng'
OCR_TYPES = ['tesseract_fitz', 'tesseract_fast', 'tesseract_legacy', 'easyocr']
CONFIG_PYTESSERACT_FAST = r'--tessdata-dir "C:\Program Files\Tesseract-OCR\tessdata_fast" --oem 3 --psm 11'
CONFIG_PYTESSERACT_LEGACY = r'--tessdata-dir "C:\Program Files\Tesseract-OCR\tessdata_legacy_best" --oem 0 --psm 11'
TRANSPARANCY = int(0.25*255)
FONT_LABEL = ImageFont.truetype('arial.ttf', size=40)
FONT_LINE = ImageFont.truetype('arial.ttf', size=20)

WINDOW_EMPTYROW_RELATIVE_FACTOR = 5
WINDOW_EMPTYROW_WINDOW_SIZE = 5

LUMINOSITY_FILLER = 255

# Derived paths
pathLabels_tabledetect_in = PATH_DATA_TABLEDETECT / 'fiverDetect1_14-04-23_williamsmith' / 'labels'
pathLabels_tablesplit = PATH_ROOT / 'data' / 'labels' / SEPARATOR_TYPE
pathPdfs_in = PATH_DATA_PDFS / '2023-03-31'  / 'samples_missing_pdfFiles'

pathLabels_tabledetect_local = PATH_ROOT / 'data' / 'labels_tabledetect'
pathPdfs_local = PATH_ROOT / 'data' / 'pdfs'

pathOcr = PATH_ROOT / 'data' / 'ocr'
pathWords = PATH_ROOT / 'data' / 'words'

pathOut = PATH_ROOT / 'data' / f'real_{COMPLEXITY_SUFFIX}'
pathOut_all = pathOut / 'all'

def replaceDirs(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        shutil.rmtree(path)
        os.makedirs(path)

os.makedirs(pathPdfs_local, exist_ok=True)
os.makedirs(pathLabels_tabledetect_local, exist_ok=True)
os.makedirs(pathOut, exist_ok=True)
os.makedirs(pathOut_all, exist_ok=True)
os.makedirs(pathOcr, exist_ok=True)
os.makedirs(pathWords, exist_ok=True)
replaceDirs(pathOut_all / 'images')
replaceDirs(pathOut_all / 'labels')
replaceDirs(pathOut_all / 'features')
replaceDirs(pathOut_all / 'images_text_and_gt')

# Helper classes
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Gather
# Gather | pdfs
tablesplit_labelFiles = list(os.scandir(pathLabels_tablesplit))
pdfFilenames = set([f"{fileEntry.name.split('-p')[0]}.pdf" for fileEntry in tablesplit_labelFiles])
for pdfName in tqdm(pdfFilenames, desc='Copying pdf files to local folder'):
    _ = shutil.copyfile(src=pathPdfs_in / pdfName, dst=pathPdfs_local / pdfName)

# Gather | tabledetect labels
tabledetectFilenames = set([f"{fileEntry.name.split('_t')[0]}.txt" for fileEntry in tablesplit_labelFiles])
for tabledetectFilename in tqdm(tabledetectFilenames, desc='Copying tabledetect labels to local folder'):
    _ = shutil.copyfile(src=pathLabels_tabledetect_in / tabledetectFilename, dst=pathLabels_tabledetect_local / tabledetectFilename)

# Gather | tabledetect labels by pdf
tabledetect_labelFiles = list(os.scandir(pathLabels_tabledetect_local))
tabledetect_labelFiles_byPdf = defaultdict(list)
for tabledetect_labelFile in tabledetect_labelFiles:
    pdfName = tabledetect_labelFile.name.split('-p')[0] + '.pdf'
    tabledetect_labelFiles_byPdf[pdfName].append(tabledetect_labelFile.name)

# Gather | words per page
def tighten_word_bbox(img_blackIs1, bboxRow):
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
        row_fewBlacks_relative = (rowBlacks <= np.mean(rowBlacks[rowBlacks!=0])/WINDOW_EMPTYROW_RELATIVE_FACTOR)
        # Rolling window approach to control for text blocks that overlap previous line text
        try:      
            row_fewBlacks_window = np.median(sliding_window_view(rowBlacks, WINDOW_EMPTYROW_WINDOW_SIZE+1), axis=1)
            if row_fewBlacks_window.max() < 0.01:       # For single rows (e.g. all dots) it sometimes returns all empty
                row_fewBlacks_window = np.full_like(row_fewBlacks_window, fill_value=False, dtype=bool)
            else:
                row_fewBlacks_window = (row_fewBlacks_window < 0.01)
            row_fewBlacks_window = np.insert(row_fewBlacks_window, obj=row_fewBlacks_window.size//2, values=np.full(shape=WINDOW_EMPTYROW_WINDOW_SIZE, fill_value=False))     # Insert 999 in middle to recover obs lost due to window
            
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
    checkGapDf = textDf[['lineno', 'top_uniform', 'bottom_uniform']].drop_duplicates()
    checkGapDf['next_top'] = checkGapDf['top_uniform'].shift(-1).fillna(9999999)
    checkGapDf['previous_bottom'] = checkGapDf['bottom_uniform'].shift(1).fillna(0)
    checkGapDf['uniform_OK'] = (checkGapDf['bottom_uniform'] < checkGapDf['next_top']) & (checkGapDf['top_uniform'] > checkGapDf['previous_bottom'])

    # Harmonise top/bottom by line | Only make uniform if gap always remains
    textDf = textDf.merge(right=checkGapDf[['lineno', 'uniform_OK']], on='lineno', how='left')
    textDf.loc[textDf['uniform_OK'], 'top'] = textDf.loc[textDf['uniform_OK'], 'top_uniform']
    textDf.loc[textDf['uniform_OK'], 'bottom'] = textDf.loc[textDf['uniform_OK'], 'bottom_uniform']
    textDf = textDf.drop(columns=['top_uniform', 'bottom_uniform', 'uniform_OK'])

    return textDf

BoxIntersect = namedtuple('BoxIntersect', field_names=['left', 'right', 'top', 'bottom', 'intersect', 'Index'])
def boxes_intersect(box, box_target):
    overlap_x = ((box.left >= box_target.left) & (box.left < box_target.right)) | ((box.right >= box_target.left) & (box.left < box_target.left))
    overlap_y = ((box.top >= box_target.top) & (box.top < box_target.bottom)) | ((box.bottom >= box_target.top) & (box.top < box_target.top))

    return all([overlap_x, overlap_y])
def box_intersects_boxList(box, box_targets):
    overlap = any([boxes_intersect(box, box_target=box_target) for box_target in box_targets])
    return overlap
    
def pdf_to_words(labelFiles_byPdf_dict, reader=None):
    # Start easyocr reader
    if not reader:
        reader = easyocr.Reader(lang_list=['nl', 'fr', 'de', 'en'], gpu=True, quantize=True)
    # Parse dict
    pdfName, labelNames = labelFiles_byPdf_dict
    pageNumbers = [int(labelName.split('-p')[1].split('.')[0].replace('p', ''))-1 for labelName in labelNames]

    # Open pdf
    pdfPath = pathPdfs_local / pdfName
    doc:fitz.Document = fitz.open(pdfPath)

    # Get words from appropriate pages
    for pageIteration, _ in tqdm(enumerate(pageNumbers), position=1, leave=False, desc='Looping over pages', total=len(pageNumbers), disable=PARALLEL):
        # Get words from page | Load page
        pageNumber = pageNumbers[pageIteration]
        page:fitz.Page = doc.load_page(page_id=pageNumber)
        pageName = f'{os.path.splitext(pdfName)[0]}-p{pageNumber+1}'
        outPath = pathWords / f'{pageName}.pq'
        if os.path.exists(outPath):
            continue

        # Get words from page | Extract directly from PDF
        textPage = page.get_textpage(flags=fitz.TEXTFLAGS_WORDS)
        words = textPage.extractWORDS()

        # Get words from page | Generate pillow and np images
        page_image = page.get_pixmap(dpi=DPI_OCR, alpha=False, colorspace=fitz.csGRAY)
        img = np.frombuffer(page_image.samples, dtype=np.uint8).reshape(page_image.height, page_image.width, page_image.n)
        _, img_array = cv.threshold(np.array(img), 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        img = Image.fromarray(img_array)
        img_blackIs1 = np.where(img_array == 0, 1, 0)

        if len(words) == 0:
            # Get words from page | OCR if necessary
            textDfs = []

            # Get words from page | OCR | From Image | PyTesseract Fast
            pathSaved = pathOcr / f'{pageName}_tesseractFast.pq'
            if not os.path.exists(pathSaved):
                text:pd.DataFrame = pytesseract.image_to_data(img, lang=LANGUAGES, config=CONFIG_PYTESSERACT_FAST, output_type=pytesseract.Output.DATAFRAME)
                text.to_parquet(path=pathSaved)
            else:
                text = pd.read_parquet(pathSaved)
            wordsDf = text.loc[text['conf'] > 30, ['left', 'top', 'width', 'height', 'text', 'conf']].assign(ocrLabel='tesseract-fast')
            wordsDf['right'] = wordsDf['left'] + wordsDf['width']
            wordsDf['bottom'] = wordsDf['top'] + wordsDf['height']
            wordsDf = wordsDf.drop(columns=['width', 'height'])
            textDfs.append(wordsDf)
            
            # Get words from page | OCR | From Image | PyTesseract Legacy
            pathSaved = pathOcr / f'{pageName}_tesseractLegacy.pq'
            if not os.path.exists(pathSaved):
                text:pd.DataFrame = pytesseract.image_to_data(img, lang=LANGUAGES, config=CONFIG_PYTESSERACT_LEGACY, output_type=pytesseract.Output.DATAFRAME)
                text.to_parquet(path=pathSaved)
            else:
                text = pd.read_parquet(pathSaved)
            wordsDf = text.loc[text['conf'] > 30, ['left', 'top', 'width', 'height', 'text', 'conf']].assign(ocrLabel='tesseract-legacy')
            wordsDf['right'] = wordsDf['left'] + wordsDf['width']
            wordsDf['bottom'] = wordsDf['top'] + wordsDf['height']
            wordsDf = wordsDf.drop(columns=['width', 'height'])
            textDfs.append(wordsDf)

            # Get words from page | OCR | From Image | EasyOCR
            pathSaved = pathOcr / f'{pageName}_easyocr.pkl'
            if not os.path.exists(pathSaved):
                text = reader.readtext(image=img_array, batch_size=50, detail=1)
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
                if ((row['bottom'] - previous_bottom) > LINENO_GAP) and (row['top'] > previous_bottom):
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
            textDf[['left', 'right', 'top', 'bottom']] = (textDf[['left', 'right', 'top', 'bottom']] * (DPI_OCR / DPI_PYMUPDF)).round(0).astype(int)

            # Crop bboxes
            textDf['toKeep'] = True
            textDf = textDf.apply(lambda row: tighten_word_bbox(img_blackIs1=img_blackIs1, bboxRow=row), axis=1)
            textDf = textDf[textDf['toKeep']].drop(columns='toKeep')
            
            # Harmonise top/bottom by line (bit wonky after crop/source combination otherwise)
            textDf = harmonise_bbox_height(textDf)
        
        # Get words from page | Save
        textDf.to_parquet(outPath)

        # Visualise words
        page_image = page.get_pixmap(dpi=DPI_OCR, alpha=False, colorspace=fitz.csGRAY)
        img = Image.frombytes(mode='L', size=(page_image.width, page_image.height), data=page_image.samples)
        img_overlay = Image.new('RGBA', img.size, (0,0,0,0))
        img_annot = ImageDraw.Draw(img_overlay, mode='RGBA')
        colors = {
            'easyocr': (255,228,181, int(0.6*255)),
            'pdf': (255, 162, 0, int(0.5*255)),
            'other': (0, 128, 0, int(0.5*255))
        }
    
        for box in textDf.itertuples('box'):      
            color = colors[box.ocrLabel] if box.ocrLabel in ['easyocr', 'pdf'] else colors['other']
            img_annot.rectangle(xy=(box.left, box.top, box.right, box.bottom), fill=color)

        lineDf = textDf.sort_values(by=['lineno', 'wordno']).drop_duplicates('lineno', keep='first')
        lineLeft = max(lineDf['left'].min() - 50, 20)

        for line in lineDf.itertuples('line'):      # line = next(lineDf.itertuples('line'))
            img_annot.text(xy=(lineLeft, line.top), text=str(line.lineno), anchor='la', fill='black', font=FONT_LINE)

    
        img_annot.text(xy=(img.size[0] // 4 * 1, img.size[1] // 15 * 14), text='easyocr', anchor='ld', fill=colors['easyocr'], font=FONT_LABEL)
        img_annot.text(xy=(img.size[0] // 4 * 2, img.size[1] // 15 * 14), text='pdf', anchor='ld', fill=colors['pdf'], font=FONT_LABEL)
        img_annot.text(xy=(img.size[0] // 4 * 3, img.size[1] // 15 * 14), text='tesseract', anchor='ld', fill=colors['other'], font=FONT_LABEL)

        img = Image.alpha_composite(img.convert('RGBA'), img_overlay)
        img.convert('RGB').save(f'{os.path.splitext(outPath)[0]}.jpg', quality=30)


if PARALLEL:
    results = Parallel(n_jobs=6, backend='loky', verbose=9)(delayed(pdf_to_words)(labelFiles_byPdf_dict) for labelFiles_byPdf_dict in tabledetect_labelFiles_byPdf.items())
else:
    reader = easyocr.Reader(lang_list=['nl', 'fr', 'de', 'en'], gpu=True, quantize=True)
    for labelFiles_byPdf_dict in tqdm(tabledetect_labelFiles_byPdf.items(), desc='Gathering words'):
        result = pdf_to_words(labelFiles_byPdf_dict, reader=reader)



# Convert
# Convert | tablesplit labels > pdf coordinates
def yoloFile_to_fitzBox(yoloPath, targetPdfSize):
    targetWidth, targetHeight = targetPdfSize
    fitzBoxes = []
    with open(yoloPath, 'r') as yoloFile:
        for annotationLine in yoloFile:
            cat, xc, yc, w, h = [float(string.strip('\n')) for string in annotationLine.split(' ')]
            
            x0 = (xc - w/2) * targetWidth
            x1 = (xc + w/2) * targetWidth
            y0 = (yc - h/2) * targetHeight
            y1 = (yc + h/2) * targetHeight

            fitzBoxes.append({'xy': [x0, y0, x1, y1],})
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
    
def imgAndWords_to_features(img, textDf:pd.DataFrame, precision=np.float32):
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

    row_spell_lengths = [getSpellLengths(row) for row in img01]
    features['row_spell_mean'] = [np.mean(row_spell_length) for row_spell_length in row_spell_lengths]
    features['row_spell_sd'] = [np.std(row_spell_length) for row_spell_length in row_spell_lengths]

    col_spell_lengths = [getSpellLengths(col) for col in img01.T]
    features['col_spell_mean'] = [np.mean(col_spell_length) for col_spell_length in col_spell_lengths]
    features['col_spell_sd'] = [np.std(col_spell_length) for col_spell_length in col_spell_lengths]

    features['global_rowAvg_p0'], features['global_rowAvg_p5'], features['global_rowAvg_p10'] = np.percentile(features['row_avg'], q=[0, 5, 10])
    features['global_colAvg_p0'], features['global_colAvg_p5'], features['global_colAvg_p10'] = np.percentile(features['col_avg'], q=[0, 5, 10])

    # Features | Text       
    # Features | Text | Text like rowstart
    # Features | Text | Text like rowstart | Identify lines with text like rowstart
    textDf_CoN = textDf.sort_values(by=['blockno', 'lineno', 'wordno']).drop_duplicates(subset=['blockno', 'lineno'], keep='first').drop(columns=['left', 'right', 'blockno', 'wordno', 'ocrLabel', 'conf']).reset_index(drop=True)
    textDf_CoN['lineno_seq'] = textDf_CoN['lineno'].rank().astype(int)
    textDf_CoN['text_like_rowstart'] = (textDf_CoN['text'].str[0].str.isupper() | textDf_CoN['text'].str[0].str.isdigit() | textDf_CoN['text'].str[:5].str.contains(r'[\.\)\-]', regex=True) )*1
    textDf_CoN = textDf_CoN.drop_duplicates(subset='top', keep='first')

    # Features | Text | Text like rowstart | Generate indicator
    # Features | Text | Text like rowstart | Generate indicator | Gather info on next line
    textDf_CoN['F1_text_like_rowstart'] = textDf_CoN['text_like_rowstart'].shift(periods=-1).fillna(1).astype(int)

    # Features | Text | Text like rowstart | Generate indicator | Assign lines to img rows
    rowTextDf = pd.DataFrame(data=np.arange(img01.shape[0], dtype=np.int32), columns=['top'])
    rowTextDf = pd.merge_asof(left=rowTextDf, right=textDf_CoN[['top', 'bottom', 'lineno', 'lineno_seq']], left_on='top', right_on='top', direction='backward')
    rowTextDf['lineno_seq'] = rowTextDf['lineno_seq'].fillna(value=0)                                             # Before first text: 0
    rowTextDf.loc[rowTextDf['top'] > rowTextDf['bottom'].max(), 'lineno_seq'] = rowTextDf['lineno_seq'].max() + 1       # After  last  text: max + 1
    lastLineNo = rowTextDf['lineno_seq'].max()

    # Features | Text | Text like rowstart | Generate indicator | Identify 'img rows' between text lines
    rowTextDf['between_textlines'] = rowTextDf['top'] > rowTextDf['bottom']
    rowTextDf.loc[rowTextDf['lineno_seq'] == lastLineNo, 'between_textlines'] = False
    
    # Features | Text | Text like rowstart | Generate indicator | Identify 'img rows' between text lines | For the start and end lines, set everything that isn't padding to separator
    firstNonWhite_line0 = np.where(features['row_avg'][rowTextDf.loc[rowTextDf['lineno_seq'] == 0].index] < 1)[0][0]
    rowTextDf.loc[(rowTextDf.index >= firstNonWhite_line0) & (rowTextDf['lineno_seq'] == 0), 'between_textlines'] = True

    lastNonWhite_lineLast = np.where(np.flip(features['row_avg'][rowTextDf.loc[rowTextDf['lineno_seq'] == rowTextDf['lineno_seq'].max()].index]) < 1)[0][0]       # distance from end
    rowTextDf.loc[(rowTextDf.index <= len(rowTextDf.index)-lastNonWhite_lineLast) & (rowTextDf['lineno_seq'] == rowTextDf['lineno_seq'].max()), 'between_textlines'] = True

    # Features | Text | Text like rowstart | Generate indicator | Identify 'img rows' between text lines | Assign half of separator to neighbouring lines (so we can verify that all lines contain separators)
    lastRow_line0 = rowTextDf.loc[rowTextDf['lineno_seq'] == 0].index.max()
    middleOfSeparator_line0 = firstNonWhite_line0 + (lastRow_line0 - firstNonWhite_line0) // 2
    rowTextDf.loc[middleOfSeparator_line0:lastRow_line0+1, 'lineno_seq'] = 1

    firstRow_lineLast = rowTextDf.loc[rowTextDf['lineno_seq'] == lastLineNo].index.min()
    separatorRowCount_lineLast = rowTextDf.loc[(rowTextDf['lineno_seq'] == lastLineNo), 'between_textlines'].sum()
    middleOfSeparator_lineLast = firstRow_lineLast + (separatorRowCount_lineLast // 2)
    rowTextDf.loc[firstRow_lineLast:middleOfSeparator_lineLast+1, 'lineno_seq'] = lastLineNo - 1
    
    rowTextDf.loc[middleOfSeparator_line0:lastRow_line0+1, 'lineno_seq'] = 1  
    
    betweenText_max_per_line = rowTextDf.groupby('lineno_seq')['between_textlines'].transform(max)
    if betweenText_max_per_line.min() == False:
        raise Exception('Not all lines contain separators')

    # Features | Text | Text like rowstart | Generate indicator | Merge text_like_rowstart info
    rowTextDf = rowTextDf.merge(right=textDf_CoN[['lineno_seq', 'text_like_rowstart', 'F1_text_like_rowstart']], left_on='lineno_seq', right_on='lineno_seq')
    rowTextDf['between_textlines_like_rowstart'] = rowTextDf['between_textlines'] & rowTextDf['F1_text_like_rowstart']

    # Features | Text | Text like rowstart | Generate indicator | Add to features
    features['row_between_textlines']               = rowTextDf['between_textlines'].to_numpy().astype(np.uint8)
    features['row_between_textlines_like_rowstart'] = rowTextDf['between_textlines_like_rowstart'].to_numpy().astype(np.uint8)

    # Features | Text | Words crossed per row/col
    textDf_WC = textDf.copy().sort_values(by=['left', 'top', 'right', 'bottom'])
    left, right, top, bottom = (textDf_WC[dim].values for dim in ['left', 'right', 'top', 'bottom'])
    
    features['row_wordsCrossed_count'] = np.array([index_to_bboxCross(index=index, mins=top, maxes=bottom) for index in range(img01.shape[0])])
    features['col_wordsCrossed_count'] = np.array([index_to_bboxCross(index=index, mins=left, maxes=right) for index in range(img01.shape[1])])

    features['row_wordsCrossed_relToMax'] = features['row_wordsCrossed_count'] / features['row_wordsCrossed_count'].max() if features['row_wordsCrossed_count'].max() != 0 else features['row_wordsCrossed_count']
    features['col_wordsCrossed_relToMax'] = features['col_wordsCrossed_count'] / features['col_wordsCrossed_count'].max() if features['col_wordsCrossed_count'].max() != 0 else features['col_wordsCrossed_count']

    # Return
    return img01, img_cv, features

def vocLabels_to_groundTruth(pathLabel, img):
    # Image
    
    # Parse xml
    root = etree.parse(pathLabel)
    objectCount = len(root.findall('.//object'))
    if objectCount:
        rows = root.findall('object[name="row separator"]')
        cols = root.findall('object[name="column separator"]')
        spanners = root.findall('object[name="spanning cell interior"]')

        # Get separator locations
        row_separators = [(int(row.find('.//ymin').text), int(row.find('.//ymax').text)) for row in rows]
        row_separators = sorted(row_separators, key= lambda x: x[0])

        col_separators = [(int(col.find('.//xmin').text), int(col.find('.//xmax').text)) for col in cols]
        col_separators = sorted(col_separators, key= lambda x: x[0])

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
    
    gt = {}
    gt['row'] = gt_row
    gt['col'] = gt_col
    return gt

def processPdf(pdfName):
    # Open pdf
    pdfPath = pathPdfs_local / pdfName
    doc:fitz.Document = fitz.open(pdfPath)
    tables, errors = 0, 0

    # Get pagenumbers of tables
    pageNumbers = [int(labelName.split('-p')[1].split('.')[0].replace('p', ''))-1 for labelName in tabledetect_labelFiles_byPdf[pdfName]]
    
    # Get tables on page
    for pageIteration, _ in tqdm(enumerate(pageNumbers), position=1, leave=False, desc='Looping over pages', total=len(pageNumbers), disable=PARALLEL):
        pageNumber = pageNumbers[pageIteration]
        page:fitz.Page = doc.load_page(page_id=pageNumber)
        
        yoloPath = pathLabels_tabledetect_local / tabledetect_labelFiles_byPdf[pdfName][pageIteration]
        fitzBoxes = yoloFile_to_fitzBox(yoloPath=yoloPath, targetPdfSize=page.mediabox_size)

        # Get text on page
        pathWordsFile = pathWords / f'{os.path.splitext(pdfName)[0]}-p{pageNumber+1}.pq'
        textDf = pd.read_parquet(pathWordsFile)
        textDf[['left', 'right', 'top', 'bottom']] = textDf[['left', 'right', 'top', 'bottom']] * (DPI_PYMUPDF / DPI_OCR)

        # Loop over tables
        for tableIteration, _ in enumerate(fitzBoxes):
            tables += 1
            tableName = os.path.splitext(pdfName)[0] + f'-p{pageNumber+1}_t{tableIteration}'
            pathImg = PATH_DATA_TABLEDETECT / 'parse_activelearning1_jpg' / 'selected' / f"{tableName}.jpg"
            pathLabel = pathLabels_tablesplit / f"{tableName}.xml"
            if (not os.path.exists(pathImg)) or (not os.path.exists(pathLabel)):
                errors += 1
                continue

            tableRect = fitz.Rect(fitzBoxes[tableIteration]['xy'])
            img_tight = page.get_pixmap(dpi=DPI_TARGET, clip=tableRect, alpha=False, colorspace=fitz.csGRAY)
            img_tight = Image.frombytes(mode='L', size=(img_tight.width, img_tight.height), data=img_tight.samples)
            img = Image.new(img_tight.mode, (img_tight.width+PADDING*2, img_tight.height+PADDING*2), 255)
            img.paste(img_tight, (PADDING, PADDING))       

            # Process text
            # Process text | Reduce to table bbox
            textDf = textDf.loc[(textDf['top'] >= tableRect.y0) & (textDf['left'] >= tableRect.x0) & (textDf['bottom'] <= tableRect.y1) & (textDf['right'] <= tableRect.x1)]        # clip to table

            # Process text | Convert to padded table image pixel coordinates
            textDf[['left', 'right']] = textDf[['left', 'right']] - tableRect.x0
            textDf[['top', 'bottom']] = textDf[['top', 'bottom']] - tableRect.y0
            textDf[['left', 'right', 'top', 'bottom']] = textDf[['left', 'right', 'top', 'bottom']] * (DPI_TARGET / DPI_PYMUPDF) + PADDING
            textDf[['left', 'right', 'top', 'bottom']] = textDf[['left', 'right', 'top', 'bottom']].round(0).astype(int)

            # Generate features
            if 'conf' not in textDf.columns:
                textDf['conf'] = 100
            img01, img_cv, features = imgAndWords_to_features(img=img, textDf=textDf)

            t = textDf.copy()
            t['topI'] = t['top'].astype(int)

            # Extract ground truths
            gt = vocLabels_to_groundTruth(pathLabel=pathLabel, img=img01)

            # Save
            # Save | Visual
            # Save | Visual | Image
            pathOut_img = str(pathOut_all / 'images' / f'{tableName}.jpg')
            cv.imwrite(filename=pathOut_img, img=img_cv)

            # Save | Visual | Image with ground truth and text feature
            pathOut_img_gt_text = str(pathOut_all / 'images_text_and_gt' / f'{tableName}.jpg')
            img_annot = img_cv.copy()

            # Save | Visual | Image with ground truth and text feature | Ground truth
            indicator_gt_row = np.expand_dims(gt['row'], axis=1)
            indicator_gt_row = np.broadcast_to(indicator_gt_row, shape=[indicator_gt_row.shape[0], 40])
            indicator_gt_row = indicator_gt_row * 140
            img_annot = np.concatenate([img_annot, indicator_gt_row], axis=1)

            indicator_gt_col = np.expand_dims(gt['col'], axis=1)
            indicator_gt_col = np.broadcast_to(indicator_gt_col, shape=[indicator_gt_col.shape[0], 40])
            indicator_gt_col = indicator_gt_col * 140
            indicator_gt_col = indicator_gt_col.T
            indicator_gt_col = np.concatenate([indicator_gt_col, np.full(fill_value=LUMINOSITY_FILLER, shape=(indicator_gt_col.shape[0], indicator_gt_row.shape[1]))], axis=1)
            img_annot = np.concatenate([img_annot, indicator_gt_col], axis=0)

            # Save | Visual | Image with ground truth and text feature | Text Feature
            # Save | Visual | Image with ground truth and text feature | Text Feature | Words crossed
            wc_row = features['row_wordsCrossed_relToMax']
            wc_row = np.expand_dims(wc_row, 1)
            wc_row = (wc_row * 255).astype(np.uint8)
            wc_row = np.broadcast_to(wc_row, shape=[wc_row.shape[0], 40])
            wc_row = np.concatenate([wc_row, np.full(fill_value=LUMINOSITY_FILLER, shape=((img_annot.shape[0] - wc_row.shape[0]), wc_row.shape[1]))], axis=0)
            img_annot = np.concatenate([img_annot, wc_row], axis=1)

            wc_col = features['col_wordsCrossed_relToMax']
            wc_col = np.expand_dims(wc_col, 1)
            wc_col = (wc_col * 255).astype(np.uint8)
            wc_col = np.broadcast_to(wc_col, shape=[wc_col.shape[0], 40])
            wc_col = wc_col.T
            wc_col = np.concatenate([wc_col, np.full(fill_value=LUMINOSITY_FILLER, shape=(wc_col.shape[0], (img_annot.shape[1] - wc_col.shape[1])))], axis=1)
            img_annot = np.concatenate([img_annot, wc_col], axis=0)

            # Save | Visual | Image with ground truth and text feature | Text Feature | Capital or Numeric
            indicator_capitalOrNumeric = np.expand_dims(features['row_firstletter_capitalOrNumeric'], 1)
            indicator_capitalOrNumeric = np.broadcast_to(indicator_capitalOrNumeric, shape=[indicator_capitalOrNumeric.shape[0], 20])
            indicator_capitalOrNumeric = indicator_capitalOrNumeric * 200
            indicator_capitalOrNumeric = np.concatenate([indicator_capitalOrNumeric, np.full(fill_value=LUMINOSITY_FILLER, shape=((img_annot.shape[0] - indicator_capitalOrNumeric.shape[0]), indicator_capitalOrNumeric.shape[1])) ], axis=0)
            img_annot = np.concatenate([img_annot, indicator_capitalOrNumeric], axis=1)

            cv.imwrite(filename=pathOut_img_gt_text, img=img_annot)
            
            # Save | Features
            pathOut_features = pathOut_all / 'features' / f'{tableName}.json'
            with open(pathOut_features, 'w') as featureFile:
                json.dump(features, featureFile, cls=NumpyEncoder)

            # Save | Ground truth
            pathOut_labels = pathOut_all / 'labels' / f'{tableName}.json'
            with open(pathOut_labels, 'w') as labelFile:
                json.dump(gt, labelFile, cls=NumpyEncoder)
    
    return (tables, errors)

if PARALLEL:
    results = Parallel(n_jobs=-1, backend='loky', verbose=9)(delayed(processPdf)(pdfName) for pdfName in tabledetect_labelFiles_byPdf)
    tables, errors = zip(*results)
    tables = sum(tables)
    errors = sum(errors)

else:
    tables, errors = 0,0
    for pdfName in tqdm(tabledetect_labelFiles_byPdf, desc='Looping over PDF source files'):        # pdfName = list(tabledetect_labelFiles_byPdf.keys())[0]
        tables_pdf, errors_pdf = processPdf(pdfName)
        tables += tables_pdf
        errors += errors_pdf
    

print(f'Tables parsed: {tables}\nErrors: {errors} ({(errors/tables*100):.00f}%)')

# Split into train/val/test
from generateFakeData import splitData
splitData(pathIn=pathOut_all)