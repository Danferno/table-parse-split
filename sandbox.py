def run():
    # Imports
    import shutil
    import os, sys
    from pathlib import Path
    import torch
    import cv2 as cv
    import numpy as np
    from collections import namedtuple, OrderedDict
    from PIL import Image, ImageDraw, ImageFont
    import pandas as pd
    import easyocr
    from collections import Counter
    from tqdm import tqdm
    import fitz        # type: ignore
    
    from model import TabliterModel
    from dataloader import get_dataloader
    from train import train
    from evaluate import evaluate
    from describe import describe_model
    
    # Constants
    # RUN_NAME = datetime.now().strftime("%Y_%m_%d__%H_%M")
    RUN_NAME = 'test'

    PATH_ROOT = Path(r"F:\ml-parsing-project\table-parse-split")
    TASKS = {'train': False, 'eval': True, 'postprocess': False}
    BEST_RUN = Path(r"F:\ml-parsing-project\table-parse-split\models\4_separator_loss_exponential\best.pt")

    PADDING = 40
    COLOR_CELL = (102, 153, 255, int(0.05*255))      # light blue
    COLOR_OUTLINE = (255, 255, 255, int(0.6*255))

    # Model parameters
    EPOCHS = 3
    MAX_LR = 0.08

    # Derived constants
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    

    # Paths
    def replaceDirs(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            shutil.rmtree(path)
            os.makedirs(path)
    path_data = Path(r"F:\ml-parsing-project\table-parse-split\data\real_narrow")
    path_model = Path(r"F:\ml-parsing-project\table-parse-split\models") / 'sandbox_model'

    # Define model
    model = TabliterModel().to(DEVICE)

    if TASKS['train']:
        # Describe model
        describe_model(model)     

        # Train
        train(epochs=EPOCHS, max_lr=MAX_LR,
            path_data_train=path_data / 'train', path_data_val=path_data / 'val', path_model=path_model, device=DEVICE)

    if TASKS['eval']:
        path_best_model = BEST_RUN if not TASKS['train'] else path_model / 'best.pt'
        evaluate(path_model=path_best_model, path_data=path_data / 'val', device=DEVICE)


    if TASKS['postprocess']:
        print(''*4, 'Post-processing', ''*4)
        dataloader_val = get_dataloader(dir_data=path_data / 'val')
        def preds_to_separators(predArray, paddingSeparator, threshold=0.8, setToMidpoint=False):
            # Tensor > np.array on cpu
            if isinstance(predArray, torch.Tensor):
                predArray = predArray.cpu().numpy().squeeze()
                
            is_separator = (predArray > threshold)
            diff_in_separator_modus = np.diff(is_separator.astype(np.int8))
            separators_start = np.where(diff_in_separator_modus == 1)[0]
            separators_end = np.where(diff_in_separator_modus == -1)[0]
            separators = np.stack([separators_start, separators_end], axis=1)
            separators = np.concatenate([paddingSeparator, separators], axis=0)
            
            # Convert wide separator to midpoint
            if setToMidpoint:
                separator_means = np.floor(separators.mean(axis=1)).astype(np.int32)
                separators = np.stack([separator_means, separator_means+1], axis=1)

            return separators
        def get_first_non_null_values(df):
            header_candidates = df.iloc[:5].fillna(method='bfill', axis=0).iloc[:1].fillna('empty').values.squeeze()
            if header_candidates.shape == ():
                header_candidates = np.expand_dims(header_candidates, 0)
            return header_candidates
        def number_duplicates(l):
            counter = Counter()

            for v in l:
                counter[v] += 1
                if counter[v]>1:
                    yield v+f'-{counter[v]}'
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
        
        # Load model
        modelRun = BEST_RUN or RUN_NAME
        pathModel = PATH_ROOT / 'models' / modelRun
        pathModelDict =  pathModel / 'best.pt'
        model.load_state_dict(torch.load(pathModelDict))
        model.eval()

        # Load OCR reader
        reader = easyocr.Reader(lang_list=['nl', 'fr', 'de', 'en'], gpu=True, quantize=True)

        # Prepare output folder
        outPath = pathModel / 'predictions_data'
        replaceDirs(outPath)

        # Define loop
        dataloader = dataloader_val
        outPath = outPath
        model = model

        # Padding separator
        paddingSeparator = np.array([[0, 40]])
        TableRect = namedtuple('tableRect', field_names=['x0', 'x1', 'y0', 'y1'])
        FONT_TEXT = ImageFont.truetype('arial.ttf', size=26)
        FONT_BIG = ImageFont.truetype('arial.ttf', size=48)

        with torch.no_grad():
            for batch in tqdm(dataloader):
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
                    wordsPath = PATH_ROOT / 'data' / 'words' / f"{name_words}"

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

                    # Cells
                    # Cells | Convert predictions to boundaries
                    separators_row = preds_to_separators(predArray=preds.row[sampleNumber], paddingSeparator=paddingSeparator, setToMidpoint=True)
                    separators_col = preds_to_separators(predArray=preds.col[sampleNumber], paddingSeparator=paddingSeparator, setToMidpoint=True)

                    # Cells | Convert boundaries to cells
                    cells = [dict(x0=separators_col[c][1]+1, y0=separators_row[r][1]+1, x1=separators_col[c+1][1], y1=separators_row[r+1][1], row=r, col=c)
                             for r in range(len(separators_row)-1) for c in range(len(separators_col)-1)]
                    cells = [scale_cell_to_dpi(cell, dpi_start=dpi_model, dpi_target=dpi_words) for cell in cells]

                    # Extract image from pdf
                    pdf = fitz.open(PATH_ROOT / 'data' / 'pdfs' / name_pdf)
                    page = pdf.load_page(pageNumber-1)
                    img = page.get_pixmap(dpi=dpi_words, clip=(tableRect.x0, tableRect.y0, tableRect.x1, tableRect.y1), colorspace=fitz.csGRAY)
                    img = np.frombuffer(img.samples, dtype=np.uint8).reshape(img.height, img.width, img.n)
                    _, img_array = cv.threshold(np.array(img), 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
                    img_tight = Image.fromarray(img_array)
                    scale_factor = dpi_words/dpi_model
                    img = Image.new(img_tight.mode, (int(img_tight.width+PADDING*2*scale_factor), int(img_tight.height+PADDING*2*scale_factor)), 255)
                    img.paste(img_tight, (int(PADDING*scale_factor), int(PADDING*scale_factor)))
                    img = img.rotate(angle, expand=True, fillcolor='white', resample=Image.Resampling.BICUBIC)     

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
                    df.to_parquet(outPath / f'{name_full}.pq')
                    
                    # Visualise
                    # Visualise | Cell annotations
                    overlay = Image.new('RGBA', img.size, (0,0,0,0))
                    img_annot = ImageDraw.Draw(overlay)
                    for cell in cells:
                        img_annot.rectangle(xy=(cell['x0'], cell['y0'], cell['x1'], cell['y1']), fill=COLOR_CELL, outline=COLOR_OUTLINE, width=2)
                        if cell['text']:
                            img_annot.rectangle(xy=img_annot.textbbox((cell['x0'], cell['y0']), text=cell['text'], font=FONT_TEXT, anchor='la'), fill=(255, 255, 255, 240), outline=(255, 255, 255, 240), width=2)
                            img_annot.text(xy=(cell['x0'], cell['y0']), text=cell['text'], fill=(0, 0, 0, 180), anchor='la', font=FONT_TEXT,)
                    img_annot.text(xy=(img.width // 2, img.height // 40 * 39), text=textSource, font=FONT_BIG, fill=(0, 0, 0, 230), anchor='md')
                    
                    # Visualise | Save image
                    img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
                    img.save(outPath / f'{name_full}.png')

if __name__ == '__main__':
    run()