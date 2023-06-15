# Install tesserocr: pip install https://github.com/simonflueckiger/tesserocr-windows_build/releases/download/tesserocr-v2.5.2-tesseract-5.3.0/tesserocr-2.5.2-cp311-cp311-win_amd64.whl

# Imports
import os, shutil
from pathlib import Path
from collections import defaultdict
import fitz
from tqdm import tqdm
import pickle
from joblib import Parallel, delayed
from time import time
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np

import pytesseract
from tesserocr import PyTessBaseAPI, RIL
import easyocr

from collections import namedtuple

# Constants
GENERATE_WORDS = True
VISUALIZE_WORDS = False
COMBINE_WORDS = True

PARALLEL = True

PATH_ROOT = Path(r"F:\ml-parsing-project\table-parse-split")
DPI_PYMUPDF = 72
DPI_OCR = 300
PADDING = 40
LANGUAGES = 'nld+fra+deu+eng'
OCR_TYPES = ['tesseract_fitz', 'tesseract_pytesseract_fast', 'tesseract_pytesseract_legacy', 'easyocr']

TRANSPARANCY = int(0.25*255)
FONT_LABEL = ImageFont.truetype('arial.ttf', size=40)

# Derived paths
pathPdfs_local = PATH_ROOT / 'data' / 'pdfs'
pathLabels_tabledetect_local = PATH_ROOT / 'data' / 'labels_tabledetect'

pathWords = PATH_ROOT / 'data' / 'words_ocrtest'


def replaceDirs(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        shutil.rmtree(path)
        os.makedirs(path)

# Gather labels by pdf
tabledetect_labelFiles = list(os.scandir(pathLabels_tabledetect_local))
tabledetect_labelFiles_byPdf = defaultdict(list)
for tabledetect_labelFile in tabledetect_labelFiles:
    pdfName = tabledetect_labelFile.name.split('-p')[0] + '.pdf'
    tabledetect_labelFiles_byPdf[pdfName].append(tabledetect_labelFile.name)
tabledetect_labelFiles_byPdf = dict(tabledetect_labelFiles_byPdf)

if GENERATE_WORDS:
    replaceDirs(pathWords / 'tesseract_fitz' )
    replaceDirs(pathWords / 'tesseract_pytesseract_fast' )
    replaceDirs(pathWords / 'tesseract_pytesseract_legacy' )
    replaceDirs(pathWords / 'easyocr' )


    # Gather words per page | Tesseract 
    # Gather words per page | Tesseract | Fitz
    def pdf_to_words_tesseract_fitz(labelFiles_byPdf_dict):
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
            outPath = pathWords / 'tesseract_fitz' / f'{os.path.splitext(pdfName)[0]}-p{pageNumber+1}.pq'

            # Get words from page | Extract directly from PDF
            textPage = page.get_textpage(flags=fitz.TEXTFLAGS_WORDS)
            words = textPage.extractWORDS()
            if len(words) == 0:
                # Get words from page | OCR if necessary
                textPage = page.get_textpage_ocr(flags=fitz.TEXTFLAGS_WORDS, language='nld+fra+deu+eng', dpi=300)
                words = textPage.extractWORDS()
                textDf = pd.DataFrame.from_records(words, columns=['left', 'top', 'right', 'bottom', 'text', 'blockno', 'lineno', 'wordno'])
                textDf.to_parquet(outPath)

    start_tesseract_fitz = time()
    if PARALLEL:
        results = Parallel(n_jobs=-1, backend='loky', verbose=9)(delayed(pdf_to_words_tesseract_fitz)(labelFiles_byPdf_dict) for labelFiles_byPdf_dict in tabledetect_labelFiles_byPdf.items())
    else:
        for labelFiles_byPdf_dict in tqdm(tabledetect_labelFiles_byPdf.items(), desc='Gathering words'):     
            result = pdf_to_words_tesseract_fitz(labelFiles_byPdf_dict)
    end_tesseract_fitz = time()
    print('Finished fitz')

    # Gather words per page | Tesseract | Pytesseract
    # Gather words per page | Tesseract | Pytesseract | Fast
    CONFIG_PYTESSERACT_FAST = r'--tessdata-dir "C:\Program Files\Tesseract-OCR\tessdata_fast" --oem 3 --psm 11'
    def pdf_to_words_tesseract_pytesseract_fast(labelFiles_byPdf_dict):
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
            outPath = pathWords / 'tesseract_pytesseract_fast' / f'{os.path.splitext(pdfName)[0]}-p{pageNumber+1}.pq'

            # Get words from page | Extract directly from PDF
            textPage = page.get_textpage(flags=fitz.TEXTFLAGS_WORDS)
            words = textPage.extractWORDS()
            if len(words) == 0:
                # Get words from page | OCR if necessary
                page_image = page.get_pixmap(dpi=DPI_OCR, alpha=False, colorspace=fitz.csGRAY)
                img_tight = Image.frombytes(mode='L', size=(page_image.width, page_image.height), data=page_image.samples)
                img = Image.new(img_tight.mode, (img_tight.width+PADDING*2, img_tight.height+PADDING*2), 255)
                img.paste(img_tight, (PADDING, PADDING)) 

                text:pd.DataFrame = pytesseract.image_to_data(img, lang=LANGUAGES, config=CONFIG_PYTESSERACT_FAST, output_type=pytesseract.Output.DATAFRAME)
                wordsDf = text.loc[text['conf'] > 30, ['left', 'top', 'width', 'height', 'text', 'block_num', 'line_num', 'word_num', 'conf']]
                wordsDf['right'] = wordsDf['left'] + wordsDf['width']
                wordsDf['bottom'] = wordsDf['top'] + wordsDf['height']
                wordsDf = wordsDf.drop(columns=['width', 'height'])
                wordsDf.to_parquet(outPath)
            else:
                continue

    start_tesseract_pytesseract_fast = time()
    if PARALLEL:
        results = Parallel(n_jobs=-1, backend='loky', verbose=9)(delayed(pdf_to_words_tesseract_pytesseract_fast)(labelFiles_byPdf_dict) for labelFiles_byPdf_dict in tabledetect_labelFiles_byPdf.items())
    else:
        for labelFiles_byPdf_dict in tqdm(tabledetect_labelFiles_byPdf.items(), desc='Gathering words'):       
            result = pdf_to_words_tesseract_pytesseract_fast(labelFiles_byPdf_dict)
    end_tesseract_pytesseract_fast = time()
    print('Finished pytesseract-fast')

    # Gather words per page | Tesseract | Pytesseract | Legacy
    CONFIG_PYTESSERACT_LEGACY = r'--tessdata-dir "C:\Program Files\Tesseract-OCR\tessdata_legacy_best" --oem 0 --psm 11'
    def pdf_to_words_tesseract_pytesseract_legacy(labelFiles_byPdf_dict):
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
            outPath = pathWords / 'tesseract_pytesseract_legacy' / f'{os.path.splitext(pdfName)[0]}-p{pageNumber+1}.pq'

            # Get words from page | Extract directly from PDF
            textPage = page.get_textpage(flags=fitz.TEXTFLAGS_WORDS)
            words = textPage.extractWORDS()
            if len(words) == 0:
                # Get words from page | OCR if necessary
                page_image = page.get_pixmap(dpi=DPI_OCR, alpha=False, colorspace=fitz.csGRAY)
                img_tight = Image.frombytes(mode='L', size=(page_image.width, page_image.height), data=page_image.samples)
                img = Image.new(img_tight.mode, (img_tight.width+PADDING*2, img_tight.height+PADDING*2), 255)
                img.paste(img_tight, (PADDING, PADDING)) 

                text:pd.DataFrame = pytesseract.image_to_data(img, lang=LANGUAGES, config=CONFIG_PYTESSERACT_LEGACY, output_type=pytesseract.Output.DATAFRAME)
                wordsDf = text.loc[text['conf'] > 30, ['left', 'top', 'width', 'height', 'text', 'block_num', 'line_num', 'word_num', 'conf']]
                wordsDf['right'] = wordsDf['left'] + wordsDf['width']
                wordsDf['bottom'] = wordsDf['top'] + wordsDf['height']
                wordsDf = wordsDf.drop(columns=['width', 'height'])
                wordsDf.to_parquet(outPath)
            else:
                continue

    start_tesseract_pytesseract_legacy = time()
    if PARALLEL:
        results = Parallel(n_jobs=-1, backend='loky', verbose=9)(delayed(pdf_to_words_tesseract_pytesseract_legacy)(labelFiles_byPdf_dict) for labelFiles_byPdf_dict in tabledetect_labelFiles_byPdf.items())
    else:
        for labelFiles_byPdf_dict in tqdm(tabledetect_labelFiles_byPdf.items(), desc='Gathering words'):       
            result = pdf_to_words_tesseract_pytesseract_legacy(labelFiles_byPdf_dict)
    end_tesseract_pytesseract_legacy = time()
    print('Finished pytesseract-legacy')

    # Gather words per page | EasyOCR
    def pdf_to_words_easyocr(labelFiles_byPdf_dict, reader:easyocr.Reader):
        # Parse dict
        pdfName, labelNames = labelFiles_byPdf_dict
        pageNumbers = [int(labelName.split('-p')[1].split('.')[0].replace('p', ''))-1 for labelName in labelNames]

        # Open pdf
        pdfPath = pathPdfs_local / pdfName
        doc:fitz.Document = fitz.open(pdfPath)

        # Get words from appropriate pages
        for pageIteration, _ in tqdm(enumerate(pageNumbers), position=1, leave=False, desc='Looping over pages', total=len(pageNumbers), disable=PARALLEL):     # pageIteration = 0
            # Get words from page | Load page
            pageNumber = pageNumbers[pageIteration]
            page:fitz.Page = doc.load_page(page_id=pageNumber)
            outPath = pathWords / 'easyocr' / f'{os.path.splitext(pdfName)[0]}-p{pageNumber+1}.pq'

            # Get words from page | Extract directly from PDF
            textPage = page.get_textpage(flags=fitz.TEXTFLAGS_WORDS)
            words = textPage.extractWORDS()
            if len(words) == 0:
                # Get words from page | OCR if necessary
                page_image = page.get_pixmap(dpi=DPI_OCR, alpha=False, colorspace=fitz.csGRAY)
                img_tight = Image.frombytes(mode='L', size=(page_image.width, page_image.height), data=page_image.samples)
                img = Image.new(img_tight.mode, (img_tight.width+PADDING*2, img_tight.height+PADDING*2), 255)
                img.paste(img_tight, (PADDING, PADDING)) 
                img_array = np.array(img)

                text = reader.readtext(image=img_array, batch_size=50, detail=1)
                text = [{'left': el[0][0][0], 'right': el[0][1][0], 'top': el[0][0][1], 'bottom': el[0][2][1], 'text': el[1], 'conf': el[2]*100} for el in text]
                text:pd.DataFrame = pd.DataFrame.from_records(text)

                wordsDf = text.loc[text['conf'] > 30]
                wordsDf.to_parquet(outPath)
            else:
                continue

    start_easyocr = time()
    reader = easyocr.Reader(lang_list=['nl', 'fr', 'de', 'en'], gpu=True)
    for labelFiles_byPdf_dict in tqdm(tabledetect_labelFiles_byPdf.items(), desc='Gathering words'):        # labelFiles_byPdf_dict = list(tabledetect_labelFiles_byPdf.items())[1]
        result = pdf_to_words_easyocr(labelFiles_byPdf_dict, reader=reader)
    end_easyocr = time()
    print('Finished easyocr')


    # Report
    duration_easyocr = end_easyocr - start_easyocr
    duration_tesseract_fitz = end_tesseract_fitz - start_tesseract_fitz
    duration_tesseract_pytesseract_fast = end_tesseract_pytesseract_fast - start_tesseract_pytesseract_fast
    duration_tesseract_pytesseract_legacy = end_tesseract_pytesseract_legacy - start_tesseract_pytesseract_legacy

    print(f'''Timings
        Method     |{' '*(5+2+7)} | Duration relative to easyocr (larger is worse)
        Easyocr    | {(duration_easyocr/60):5.2f} minutes | {duration_easyocr/duration_easyocr:4.2f}
        Fitz       | {(duration_tesseract_fitz/60):5.2f} minutes | {duration_tesseract_fitz/duration_easyocr:4.2f}
        Pyt-fast   | {(duration_tesseract_pytesseract_fast/60):5.2f} minutes | {duration_tesseract_pytesseract_fast/duration_easyocr:4.2f}
        Pyt-legacy | {(duration_tesseract_pytesseract_legacy/60):5.2f} minutes | {duration_tesseract_pytesseract_legacy/duration_easyocr:4.2f}
    ''')


    print('Done')







    # Gather words per page | Tesseract | Tesserocr
    # CONFIG_PYTESSERACT_FAST = r'--tessdata-dir "C:\Program Files\Tesseract-OCR\tessdata_fast" --oem 3 --psm 11'
    # def pdf_to_words_tesseract_tesserocr_fast(labelFiles_byPdf_dict, api):
    #     # Parse dict
    #     pdfName, labelNames = labelFiles_byPdf_dict
    #     pageNumbers = [int(labelName.split('-p')[1].split('.')[0].replace('p', ''))-1 for labelName in labelNames]

    #     # Open pdf
    #     pdfPath = pathPdfs_local / pdfName
    #     doc:fitz.Document = fitz.open(pdfPath)

    #     # Get words from appropriate pages
    #     for pageIteration, _ in tqdm(enumerate(pageNumbers), position=1, leave=False, desc='Looping over pages', total=len(pageNumbers), disable=PARALLEL):     # pageIteration = 0
    #         # Get words from page | Load page
    #         pageNumber = pageNumbers[pageIteration]
    #         page:fitz.Page = doc.load_page(page_id=pageNumber)
    #         outPath = pathWords / 'tesseract_tesserocr_fast' / f'{os.path.splitext(pdfName)[0]}-p{pageNumber+1}.pq'

    #         # Get words from page | Extract directly from PDF
    #         textPage = page.get_textpage(flags=fitz.TEXTFLAGS_WORDS)
    #         words = textPage.extractWORDS()
    #         if len(words) == 0:
    #             # Get words from page | OCR if necessary
    #             page_image = page.get_pixmap(dpi=DPI_TARGET, alpha=False, colorspace=fitz.csGRAY)
    #             img_tight = Image.frombytes(mode='L', size=(page_image.width, page_image.height), data=page_image.samples)
    #             img = Image.new(img_tight.mode, (img_tight.width+PADDING*2, img_tight.height+PADDING*2), 255)
    #             img.paste(img_tight, (PADDING, PADDING)) 

    #             api.SetImage(img)
    #             text = api.GetComponentImages(RIL.WORD, True)
    #             texts = []
    #             for i, (im, box, _, _) in enumerate(text):
    #                 delta = img.size[0] / 200
    #                 api.SetRectangle(box['x'] - delta, box['y'] - delta, box['w'] + 2 * delta, box['h'] + 2 * delta)
    #                 # widening the box with delta can greatly improve the text output
    #                 ocrResult = api.GetUTF8Text()
    #                 texts.append(ocrResult)


    #             wordsDf = text.loc[text['conf'] > 30, ['left', 'top', 'width', 'height', 'text', 'block_num', 'line_num', 'word_num', 'conf']]
    #             wordsDf['right'] = wordsDf['left'] + wordsDf['width']
    #             wordsDf['bottom'] = wordsDf['top'] + wordsDf['height']
    #             wordsDf = wordsDf.drop(columns=['width', 'height'])
    #             wordsDf.to_parquet(outPath)
    #         else:
    #             continue

    # start_tesseract_tesserocr = time()
    # # if PARALLEL:

    # #     results = Parallel(n_jobs=-1, backend='loky', verbose=9)(delayed(pdf_to_words_tesseract_tesserocr_fast)(labelFiles_byPdf_dict) for labelFiles_byPdf_dict in tabledetect_labelFiles_byPdf.items())
    # # else:
    # api = PyTessBaseAPI(path=r"C:\Program Files\Tesseract-OCR\tessdata_fast", lang=LANGUAGES, psm=11, oem=3)
    # with PyTessBaseAPI() as api:
    #     for labelFiles_byPdf_dict in tqdm(tabledetect_labelFiles_byPdf.items(), desc='Gathering words'):        # labelFiles_byPdf_dict = list(tabledetect_labelFiles_byPdf.items())[1]
    #         result = pdf_to_words_tesseract_tesserocr_fast(labelFiles_byPdf_dict, api)
    # end_tesseract_tesserocr = time()

if VISUALIZE_WORDS:
    pathImages = pathWords / 'annotated'
    replaceDirs(pathImages)
    ocrType_colors = [(25, 100, 126, TRANSPARANCY), (28, 155, 98, TRANSPARANCY), (244, 211, 94, TRANSPARANCY), (197, 123, 87, TRANSPARANCY)]

    def df_to_boxes(df, img, ocrType, tableName):
        ocrLabel = OCR_TYPES[ocrType]
        if 'tesseract_fitz' in ocrLabel:
            df[['left', 'right', 'top', 'bottom']] = df[['left', 'right', 'top', 'bottom']] * (DPI_OCR / DPI_PYMUPDF) + PADDING
        
        # df[['left', 'top']] = df[['left', 'top']] - 3
        # df[['right', 'bottom']] = df[['right', 'bottom']] + 3
        # Overlay
        img_overlay = Image.new('RGBA', img.size, (0,0,0,0))
        img_annot = ImageDraw.Draw(img_overlay, mode='RGBA')
        color = ocrType_colors[ocrType]
        
        for idx, row in df.iterrows():
            img_annot.rectangle(xy=row[['left', 'top', 'right', 'bottom']], fill=color, width=4, outline=(255,255,255,TRANSPARANCY))
        
        img_annot.text(xy=((img.size[0] // (len(OCR_TYPES)+1)) * (ocrType+1), img.size[1] // 12 * 11), text=ocrLabel, anchor='ld', fill=color, font=FONT_LABEL)

        # Individual
        img_type = Image.alpha_composite(img, img_overlay)
        img_type.save(pathImages / f'{tableName}_{ocrLabel}.png' )
        return img_overlay


    def pdf_and_words_to_images(labelFiles_byPdf_dict):
        # Parse dict
        pdfName, labelNames = labelFiles_byPdf_dict
        pageNumbers = [int(labelName.split('-p')[1].split('.')[0].replace('p', ''))-1 for labelName in labelNames]

        # Open pdf
        pdfPath = pathPdfs_local / pdfName
        doc:fitz.Document = fitz.open(pdfPath)

        # Get image and words from appropriate pages
        for pageIteration, _ in tqdm(enumerate(pageNumbers), position=1, leave=False, desc='Looping over pages', total=len(pageNumbers), disable=PARALLEL):     # pageIteration = 0
            # Load page
            pageNumber = pageNumbers[pageIteration]
            page:fitz.Page = doc.load_page(page_id=pageNumber)
            tableName = f'{os.path.splitext(pdfName)[0]}-p{pageNumber+1}'
            wordsPaths = [pathWords / ocrType / f'{tableName}.pq' for ocrType in OCR_TYPES]
            if not any([os.path.exists(wordsPath) for wordsPath in wordsPaths]):
                continue

            # Extract img from pdf
            page_image = page.get_pixmap(dpi=DPI_OCR, alpha=False, colorspace=fitz.csGRAY)
            img_tight = Image.frombytes(mode='L', size=(page_image.width, page_image.height), data=page_image.samples)
            img = Image.new(img_tight.mode, (img_tight.width+PADDING*2, img_tight.height+PADDING*2), 255)
            img.paste(img_tight, (PADDING, PADDING)) 
            img = img.convert('RGBA')

            # Fetch words
            img_annots = [df_to_boxes(df=pd.read_parquet(path=path), img=img, ocrType=ocrType, tableName=tableName) for ocrType, path in enumerate(wordsPaths)]    # ocrType = 0
            for img_annot in img_annots:
                img = Image.alpha_composite(img, img_annot)
            img.convert('RGB').save(pathImages / f'{tableName}_overlay.png')


    
    if PARALLEL:
        results = Parallel(n_jobs=-1, backend='loky', verbose=9)(delayed(pdf_and_words_to_images)(labelFiles_byPdf_dict) for labelFiles_byPdf_dict in tabledetect_labelFiles_byPdf.items())
    else:
        for labelFiles_byPdf_dict in tqdm(tabledetect_labelFiles_byPdf.items(), desc='Visualising words'):          # labelFiles_byPdf_dict = list(tabledetect_labelFiles_byPdf.items())[1]
            result = pdf_and_words_to_images(labelFiles_byPdf_dict)
    print('Finished visualising')

if COMBINE_WORDS:
    pathImagesCombined = pathWords / 'annotated_combined'
    pathWordsCombined = pathWords / 'words_combined'
    replaceDirs(pathImagesCombined)
    replaceDirs(pathWordsCombined)
    BoxIntersect = namedtuple('BoxIntersect', field_names=['left', 'right', 'top', 'bottom', 'intersect', 'Index'])

    def boxes_intersect(box, box_target):
        overlap_x = ((box.left >= box_target.left) & (box.left < box_target.right)) | ((box.right >= box_target.left) & (box.left < box_target.left))
        overlap_y = ((box.top >= box_target.top) & (box.top < box_target.bottom)) | ((box.bottom >= box_target.top) & (box.top < box_target.top))

        return all([overlap_x, overlap_y])

    def box_intersects_boxList(box, box_targets):
        overlap = any([boxes_intersect(box, box_target=box_target) for box_target in box_targets])
        return overlap
        

    def pdf_and_words_to_combined_words(labelFiles_byPdf_dict):
        # Parse dict
        pdfName, labelNames = labelFiles_byPdf_dict
        pageNumbers = [int(labelName.split('-p')[1].split('.')[0].replace('p', ''))-1 for labelName in labelNames]

        # Open pdf
        pdfPath = pathPdfs_local / pdfName
        doc:fitz.Document = fitz.open(pdfPath)

        # Get image and words from appropriate pages
        for pageIteration, _ in tqdm(enumerate(pageNumbers), position=1, leave=False, desc='Looping over pages', total=len(pageNumbers), disable=PARALLEL):     # pageIteration = 0
            # Load page
            pageNumber = pageNumbers[pageIteration]
            page:fitz.Page = doc.load_page(page_id=pageNumber)
            tableName = f'{os.path.splitext(pdfName)[0]}-p{pageNumber+1}'
            wordsPaths = [pathWords / ocrType / f'{tableName}.pq' for ocrType in OCR_TYPES]
            if not any([os.path.exists(wordsPath) for wordsPath in wordsPaths]):
                continue

            # Extract img from pdf
            page_image = page.get_pixmap(dpi=DPI_OCR, alpha=False, colorspace=fitz.csGRAY)
            img_tight = Image.frombytes(mode='L', size=(page_image.width, page_image.height), data=page_image.samples)
            img = Image.new(img_tight.mode, (img_tight.width+PADDING*2, img_tight.height+PADDING*2), 255)
            img.paste(img_tight, (PADDING, PADDING)) 
            img = img.convert('RGBA')

            # Fetch words
            df = pd.concat([pd.read_parquet(path=path).assign(ocrLabel=OCR_TYPES[ocrType]) for ocrType, path in enumerate(wordsPaths)])    # ocrType = 0
            df.loc[df['ocrLabel'] == 'tesseract_fitz', ['left', 'right', 'top', 'bottom']] = df.loc[df['ocrLabel'] == 'tesseract_fitz', ['left', 'right', 'top', 'bottom']] * (DPI_OCR / DPI_PYMUPDF) + PADDING
            df[['left', 'right', 'top', 'bottom']] = df[['left', 'right', 'top', 'bottom']].round(0).astype(int)
            df = df[['left', 'right', 'top', 'bottom', 'ocrLabel', 'text']]
            df['text_sparse'] = df['text'].str.replace('[^a-zA-Z0-9À-ÿ.\(\)_\-\+ ]', '', regex=True)
            df = df.loc[df['text_sparse'].str.len() >= 2].drop(columns='text_sparse').reset_index(drop=True)

            # Detect intersection
            easyocr_boxes = set(df.loc[df['ocrLabel'] == 'easyocr', ['left', 'top', 'right', 'bottom']].itertuples(name='Box'))
            other_boxes   = set(df.loc[df['ocrLabel'] != 'easyocr', ['left', 'top', 'right', 'bottom']].itertuples(name='Box'))

            other_boxes = [BoxIntersect(left=otherBox.left, right=otherBox.right, top=otherBox.top, bottom=otherBox.bottom, intersect=box_intersects_boxList(box=otherBox, box_targets=easyocr_boxes), Index=otherBox.Index) for otherBox in other_boxes]

            # Visualise intersection
            img_overlay = Image.new('RGBA', img.size, (0,0,0,0))
            img_annot = ImageDraw.Draw(img_overlay, mode='RGBA')
            colors = {
                'EasyOCR': (255,228,181, int(0.9*255)),
                'Overlap': (255, 0, 0, TRANSPARANCY),
                'Extra': (0, 128, 0, TRANSPARANCY)
            }
        
            for box in other_boxes:      # box = intersecting_boxes[0]
                color = colors['Overlap'] if box.intersect else colors['Extra']
                img_annot.rectangle(xy=(box.left, box.top, box.right, box.bottom), fill=color, width=4, outline=(255,255,255,TRANSPARANCY))
            for box in easyocr_boxes:      # box = intersecting_boxes[0]
                img_annot.rectangle(xy=(box.left, box.top, box.right, box.bottom), outline=colors['EasyOCR'], width=4)
        
            img_annot.text(xy=(img.size[0] // 4 * 1, img.size[1] // 12 * 11), text='EasyOcr', anchor='ld', fill=colors['EasyOCR'], font=FONT_LABEL)
            img_annot.text(xy=(img.size[0] // 4 * 2, img.size[1] // 12 * 11), text='Overlap', anchor='ld', fill=colors['Overlap'], font=FONT_LABEL)
            img_annot.text(xy=(img.size[0] // 4 * 3, img.size[1] // 12 * 11), text='Extra', anchor='ld', fill=colors['Extra'], font=FONT_LABEL)

            img_intersection = Image.alpha_composite(img, img_overlay)
            img_intersection.convert('RGB').save(pathImagesCombined / f'{tableName}_intersection.png')

            # Combine words
            extraBoxes = [box for box in other_boxes if not box.intersect ]
            indexesToKeep = [box.Index for box in easyocr_boxes.union(set(extraBoxes))]
            df = df.iloc[indexesToKeep]
            df.to_parquet(path=pathWordsCombined / f'{tableName}.pq')

            # Visualise new words
            img_overlay = Image.new('RGBA', img.size, (0,0,0,0))
            img_annot = ImageDraw.Draw(img_overlay, mode='RGBA')
            colors = {
                'EasyOCR': (255,228,181, int(0.6*255)),
                'Other': (0, 128, 0, int(0.5*255))
            }
        
            for box in df.itertuples('box'):      # box = intersecting_boxes[0]
                color = colors['EasyOCR'] if box.ocrLabel == 'easyocr' else colors['Other']
                img_annot.rectangle(xy=(box.left, box.top, box.right, box.bottom), fill=color, width=4, outline=(255,255,255,TRANSPARANCY))
        
            img_annot.text(xy=(img.size[0] // 4 * 1, img.size[1] // 12 * 11), text='EasyOcr', anchor='ld', fill=colors['EasyOCR'], font=FONT_LABEL)
            img_annot.text(xy=(img.size[0] // 4 * 2, img.size[1] // 12 * 11), text='Other', anchor='ld', fill=colors['Other'], font=FONT_LABEL)

            img = Image.alpha_composite(img, img_overlay)
            img.convert('RGB').save(pathImagesCombined / f'{tableName}_words.png')
   
    if PARALLEL:
        results = Parallel(n_jobs=-1, backend='loky', verbose=9)(delayed(pdf_and_words_to_combined_words)(labelFiles_byPdf_dict) for labelFiles_byPdf_dict in tabledetect_labelFiles_byPdf.items())
    else:
        for labelFiles_byPdf_dict in tqdm(tabledetect_labelFiles_byPdf.items(), desc='Visualising words'):          # labelFiles_byPdf_dict = list(tabledetect_labelFiles_byPdf.items())[1]
            result = pdf_and_words_to_combined_words(labelFiles_byPdf_dict)
    print('Finished combining words from multiple ranges')