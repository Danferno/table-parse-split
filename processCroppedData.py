# Imports
import os
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
import shutil
from collections import defaultdict
import fitz
import pandas as pd
from PIL import Image, ImageDraw
import numpy as np
import cv2 as cv
import json
from lxml import etree
import pickle

# Constants
DEBUG = False
PARALLEL = True
SEPARATOR_TYPE = 'wide'
COMPLEXITY_SUFFIX = 4

PATH_ROOT = Path(r"F:\ml-parsing-project\table-parse-split")
PATH_DATA_TABLEDETECT = Path(r"F:\ml-parsing-project\data")
PATH_DATA_PDFS = Path(r"F:\datatog-data-dev\kb-knowledgeBase\bm-benchmark\be-unlisted")
DPI_PYMUPDF = 72
DPI_TARGET = 150
PADDING = 40
PRECISION = np.float32

LUMINOSITY_FILLER = 255

# Derived paths
pathLabels_tabledetect_in = PATH_DATA_TABLEDETECT / 'fiverDetect1_14-04-23_williamsmith' / 'labels'
pathLabels_tablesplit = PATH_ROOT / 'data' / 'labels' / SEPARATOR_TYPE
pathPdfs_in = PATH_DATA_PDFS / '2023-03-31'  / 'samples_missing_pdfFiles'

pathLabels_tabledetect_local = PATH_ROOT / 'data' / 'labels_tabledetect'
pathPdfs_local = PATH_ROOT / 'data' / 'pdfs'

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
def pdf_to_words(labelFiles_byPdf_dict):
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
        outPath = pathWords / f'{os.path.splitext(pdfName)[0]}-p{pageNumber+1}.pkl'
        if os.path.exists(outPath):
            continue

        # Get words from page | Extract directly from PDF
        textPage = page.get_textpage(flags=fitz.TEXTFLAGS_WORDS)
        words = textPage.extractWORDS()
        if len(words) == 0:
            # Get words from page | OCR if necessary
            textPage = page.get_textpage_ocr(flags=fitz.TEXTFLAGS_WORDS, language='nld+fra+deu+eng', dpi=300)
            words = textPage.extractWORDS()

        # Get words from page | Save
        with open(outPath, 'wb') as file:
            pickle.dump(obj=words, file=file)


if PARALLEL:
    results = Parallel(n_jobs=-1, backend='loky', verbose=9)(delayed(pdf_to_words)(labelFiles_byPdf_dict) for labelFiles_byPdf_dict in tabledetect_labelFiles_byPdf.items())
else:
    for labelFiles_byPdf_dict in tqdm(tabledetect_labelFiles_byPdf.items(), desc='Gathering words'):        # pdfName = list(tabledetect_labelFiles_byPdf.keys())[0]
        result = pdf_to_words(labelFiles_byPdf_dict)



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

def img_to_features(pathImg, textDf, precision=np.float32):
    # Prepare image
    img = Image.open(pathImg).convert('L')
    _, img_cv = cv.threshold(np.array(img), 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    img = img_cv.astype(precision)/255

    # Features
    features = {}

    # Features | Visual
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

    # Features | Text
    textDf['firstletter_capitalOrNumeric'] = (textDf['text'].str[0].str.isupper() | textDf['text'].str[0].str.isdigit())*1
    textDf = textDf.loc[(textDf.conf > 50)]
    textDf = textDf[['top', 'firstletter_capitalOrNumeric']].astype(int).drop_duplicates(subset='top', keep='first')
    textDf = textDf.set_index('top').reindex(range(0, img.shape[0]))
    textDf['firstletter_capitalOrNumeric'] = textDf['firstletter_capitalOrNumeric'].fillna(method='ffill')
    textDf['firstletter_capitalOrNumeric'] = textDf['firstletter_capitalOrNumeric'].fillna(value=0)
    features['row_firstletter_capitalOrNumeric'] = textDf['firstletter_capitalOrNumeric'].to_numpy().astype(np.uint8)

    # Return
    return img, img_cv, features

def vocLabels_to_groundTruth(pathLabel, img):
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
        pathWordsFile = pathWords / f'{os.path.splitext(pdfName)[0]}-p{pageNumber+1}.pkl'
        with open(pathWordsFile, 'rb') as file:
            words = pickle.load(file)

        for tableIteration, _ in enumerate(fitzBoxes):
            tables += 1
            tableName = os.path.splitext(pdfName)[0] + f'-p{pageNumber+1}_t{tableIteration}'
            pathImg = PATH_DATA_TABLEDETECT / 'parse_activelearning1_jpg' / 'selected' / f"{tableName}.jpg"
            pathLabel = pathLabels_tablesplit / f"{tableName}.xml"
            if (not os.path.exists(pathImg)) or (not os.path.exists(pathLabel)):
                errors += 1
                continue

            tableRect = fitz.Rect(fitzBoxes[tableIteration]['xy'])

            # Extract first word per line in table
            textDf = pd.DataFrame.from_records(words, columns=['left', 'top', 'right', 'bottom', 'text', 'blockno', 'lineno', 'wordno'])
            textDf = textDf.loc[(textDf['top'] >= tableRect.y0) & (textDf['left'] >= tableRect.x0) & (textDf['bottom'] <= tableRect.y1) & (textDf['right'] <= tableRect.x1)]        # clip to table
            textDf['text'] = textDf['text'].str.replace('[^a-zA-Z0-9]', '', regex=True)
            textDf = textDf.loc[textDf['text'].str.len() >= 2]
            textDf = textDf.sort_values(by=['blockno', 'lineno', 'wordno']).drop_duplicates(subset=['blockno', 'lineno'], keep='first')

            if DEBUG:
                # Draw on pdf
                doc.fullcopy_page(pageNumbers[0])
                page_annot = doc.load_page(-1)
                for idx, row in textDf.iterrows():      # row = df.iloc[0]
                    rect = row[['left', 'top', 'right', 'bottom']].tolist()
                    page_annot.draw_rect(rect)
                page_annot.get_pixmap(alpha=False).save('testA.jpg')

                # Draw on img
                df_page = textDf[['left', 'top', 'right', 'bottom']] * DPI_TARGET / DPI_PYMUPDF
                img_fitz = page.get_pixmap(alpha=False, dpi=DPI_TARGET)
                img = Image.frombytes(mode='RGB', size=[img_fitz.width, img_fitz.height], data=img_fitz.samples)
                
                img_annot = ImageDraw.Draw(img)
                for idx, row in df_page.iterrows():      # row = df.iloc[0]
                    img_annot.rectangle(xy=row[['left', 'top', 'right', 'bottom']], outline='red')

                img.show()

            # Draw on table
            pathImg = PATH_DATA_TABLEDETECT / 'parse_activelearning1_jpg' / 'selected' / f"{tableName}.jpg"
            df_table = textDf[['left', 'top', 'right', 'bottom']].copy()
            df_table[['left', 'right']] = df_table[['left', 'right']] - tableRect.x0
            df_table[['top', 'bottom']] = df_table[['top', 'bottom']] - tableRect.y0
            df_table = df_table * (DPI_TARGET / DPI_PYMUPDF) + PADDING

            if DEBUG:
                img_table = Image.open(pathImg)
                img_table_annot = ImageDraw.Draw(img_table)
                for idx, row in df_table.iterrows():      # row = df.iloc[0]
                    img_table_annot.rectangle(xy=row[['left', 'top', 'right', 'bottom']], outline='red')
                img_table.show()
            df_table = df_table.join(other=textDf['text'])

            # Generate features
            df_table['conf'] = 100
            img, img_cv, features = img_to_features(pathImg=pathImg, textDf=df_table)

            t = df_table.copy()
            t['topI'] = t['top'].astype(int)

            # Extract ground truths
            gt = vocLabels_to_groundTruth(pathLabel=pathLabel, img=img)

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
            indicator_capitalOrNumeric = np.expand_dims(features['row_firstletter_capitalOrNumeric'], 1)
            indicator_capitalOrNumeric = np.broadcast_to(indicator_capitalOrNumeric, shape=[indicator_capitalOrNumeric.shape[0], 40])
            indicator_capitalOrNumeric = indicator_capitalOrNumeric * 200
            indicator_capitalOrNumeric = np.concatenate([indicator_capitalOrNumeric, np.full(fill_value=LUMINOSITY_FILLER, shape=(indicator_gt_col.shape[0], indicator_capitalOrNumeric.shape[1])) ], axis=0)
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