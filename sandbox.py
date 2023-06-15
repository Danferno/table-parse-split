def run():
    # Imports
    import shutil
    import os, sys
    from pathlib import Path
    import json
    import torch
    from torch import nn
    from torchvision.io import read_image, ImageReadMode
    from torch.utils.data import Dataset, DataLoader
    from torch import Tensor
    from prettytable import PrettyTable
    import cv2 as cv
    import numpy as np
    from collections import namedtuple, OrderedDict
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    from time import perf_counter
    from functools import cache
    from PIL import Image, ImageDraw, ImageFont
    import pandas as pd
    import easyocr
    from collections import Counter
    from tqdm import tqdm
    import fitz        # type: ignore
    
    from model import TabliterModel, LOSS_ELEMENTS_COUNT
    from loss import WeightedBinaryCrossEntropyLoss, LogisticLoss, getLossFunctions, calculateLoss
    from dataloader import get_dataloader
    
    # Constants
    # RUN_NAME = datetime.now().strftime("%Y_%m_%d__%H_%M")
    RUN_NAME = 'test'

    PATH_ROOT = Path(r"F:\ml-parsing-project\table-parse-split")
    SUFFIX = 'narrow'
    DATA_TYPE = 'real'
    PROFILE = False
    TASKS = {'train': True, 'eval': True, 'postprocess': True}
    # BEST_RUN = '2023_06_09__15_01'
    # BEST_RUN = '2023_06_12__18_13'
    BEST_RUN = None  
    
    LUMINOSITY_GT_FEATURES_MAX = 240
    LUMINOSITY_FILLER = 255

    PADDING = 40
    COLOR_CELL = (102, 153, 255, int(0.05*255))      # light blue
    COLOR_OUTLINE = (255, 255, 255, int(0.6*255))

    # Model parameters
    EPOCHS = 3
    BATCH_SIZE = 1
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
    pathData = PATH_ROOT / 'data' / f'{DATA_TYPE}_{SUFFIX}'
    pathLogs = PATH_ROOT / 'torchlogs'
    pathModels = PATH_ROOT / 'models'

    # Timing
    timers = {}

    # Data   
    dataloader_train = get_dataloader(dir_data=pathData / 'train', shuffle=True)
    dataloader_val = get_dataloader(dir_data=pathData / 'val', shuffle=False)
    dataloader_test = get_dataloader(dir_data=pathData / 'test', shuffle=False)

    # Define model
    model = TabliterModel().to(DEVICE)

    # Loss function
    # Loss function | Calculate target ratio to avoid dominant focus

    # Train
    lossFunctions = getLossFunctions(dataloader=dataloader_train)
    optimizer = torch.optim.SGD(model.parameters(), lr=MAX_LR)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, steps_per_epoch=len(dataloader_train), epochs=EPOCHS)

    def train_loop(dataloader, model, lossFunctions, optimizer, report_frequency=4, device='cuda'):
        print('Train')
        start = perf_counter()
        size = len(dataloader.dataset)
        batch_size = dataloader.batch_size
        epoch_loss = 0
        for batchNumber, batch in enumerate(dataloader):     # batch, sample = next(enumerate(dataloader))
            # Compute prediction and loss
            preds = model(batch.features)
            loss = calculateLoss(batch.targets, preds, lossFunctions, device=device)
            epoch_loss += loss
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # Report intermediate losses
            report_batch_size = (size / batch_size) // report_frequency
            if (batchNumber+1) % report_batch_size == 0:
                epoch_loss, current = epoch_loss.item(), (batchNumber+1) * batch_size
                print(f'\tAvg epoch loss: {epoch_loss/current:.3f} [{current:>5d}/{size:>5d}]')
        
        print(f'\tEpoch duration: {perf_counter()-start:.0f}s')
        return epoch_loss / len(dataloader)

    def val_loop(dataloader, model, lossFunctions, device='cuda'):
        batchCount = len(dataloader)
        val_loss, correct, maxCorrect = torch.zeros(size=(LOSS_ELEMENTS_COUNT,1), device=DEVICE), torch.zeros(size=(LOSS_ELEMENTS_COUNT,1), device=device, dtype=torch.int64), torch.zeros(size=(LOSS_ELEMENTS_COUNT,1), device=DEVICE, dtype=torch.int64)
        with torch.no_grad():
            for batch in dataloader:     # batch = next(iter(dataloader))
                # Compute prediction and loss
                preds = model(batch.features)
                val_loss_batch, correct_batch, maxCorrect_batch = calculateLoss(batch.targets, preds, lossFunctions, calculateCorrect=True)
                val_loss += val_loss_batch
                correct  += correct_batch
                maxCorrect  += maxCorrect_batch

        val_loss = val_loss / batchCount
        shareCorrect = correct / maxCorrect

        print(f'''Validation
            Accuracy line-level: {(100*shareCorrect[0].item()):>0.1f}% (row) | {(100*shareCorrect[2].item()):>0.1f}% (col)
            Separator count (relative to truth): {(100*shareCorrect[1].item()):>0.1f}% (row) | {(100*shareCorrect[3].item()):>0.1f}% (col)
            Avg val loss: {val_loss.sum().item():.3f} (total) | {val_loss[0].item():.3f} (row-line) | {val_loss[2].item():.3f} (col-line) | {val_loss[1].item():.3f} (row-separator) | {val_loss[3].item():.3f} (col-separator)''')
        
        return val_loss

    if TASKS['train']:
        # Describe model
        # Model description | Count parameters
        def count_parameters(model):
            table = PrettyTable(["Modules", "Parameters"])
            table.align['Modules'] = 'l'
            table.align['Parameters'] = 'r'
            total_params = 0
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad: continue
                params = parameter.numel()
                table.add_row([name, params])
                total_params+=params
            print(table)
            print(f"Total Trainable Params: {total_params}")
            return total_params
        print(model)
        count_parameters(model=model)


        # Prepare folders
        pathModel = pathModels / RUN_NAME
        os.makedirs(pathModel, exist_ok=True)
        writer = SummaryWriter(f"torchlogs/{RUN_NAME}")
        
        start_train = perf_counter()
        with torch.autograd.profiler.profile(enabled=PROFILE) as prof:
            best_val_loss = 9e20
            for epoch in range(EPOCHS):
                learning_rate = scheduler.get_last_lr()[0]
                print(f"\nEpoch {epoch+1} of {EPOCHS}. Learning rate: {learning_rate:03f}")
                model.train()
                train_loss = train_loop(dataloader=dataloader_train, model=model, lossFunctions=lossFunctions, optimizer=optimizer, report_frequency=4, device=DEVICE)
                model.eval()
                val_loss = val_loop(dataloader=dataloader_val, model=model, lossFunctions=lossFunctions, device=DEVICE)

                writer.add_scalar('Train/Loss', scalar_value=train_loss, global_step=epoch)
                writer.add_scalar('Val/Loss/Total', scalar_value=val_loss.sum(), global_step=epoch)
                writer.add_scalars('Val/Loss/Components', tag_scalar_dict={
                    'row_line': val_loss[0],
                    'row_separator_count': val_loss[1],
                    'col_line': val_loss[2],
                    'col_separator_count': val_loss[3]
                }, global_step=epoch)
                writer.add_scalar('Learning rate', scalar_value=learning_rate, global_step=epoch)

                if val_loss.sum() < best_val_loss:
                    best_val_loss = val_loss.sum()
                    torch.save(model.state_dict(), pathModel / 'best.pt')

            torch.save(model.state_dict(), pathModel / f'last.pt')

        if PROFILE:
            print(prof.key_averages().table(sort_by="self_cpu_time_total"))

        # Finish model description
        sample = next(iter(dataloader_train))
        writer.add_graph(model, input_to_model=[sample.features])
        writer.add_hparams(hparam_dict={'epochs': EPOCHS,
                                        'batch_size': BATCH_SIZE,
                                        'max_lr': MAX_LR},
                        metric_dict={'val_loss': best_val_loss.sum().item()})
        writer.close()

        timers['train'] = perf_counter() - start_train

    if TASKS['eval']:
        modelRun = BEST_RUN or RUN_NAME
        pathModelDict = PATH_ROOT / 'models' / modelRun / 'best.pt'
        model.load_state_dict(torch.load(pathModelDict))
        model.eval()

        # Prepare folders
        path_annotations_raw_val = pathModel / 'val_annotated'
        replaceDirs(path_annotations_raw_val)  

        # Predict
        start_eval = perf_counter()
        def convert_01_array_to_visual(array, invert=False, width=40) -> np.array:
            luminosity = (1 - array) * LUMINOSITY_GT_FEATURES_MAX if invert else array * LUMINOSITY_GT_FEATURES_MAX
            luminosity = luminosity.round(0).astype(np.uint8)
            luminosity = np.expand_dims(luminosity, axis=1)
            luminosity = np.broadcast_to(luminosity, shape=(luminosity.shape[0], width))
            return luminosity

        def eval_loop(dataloader, model, lossFunctions, outPath=None):
            batchCount = len(dataloader)
            eval_loss, correct, maxCorrect = torch.zeros(size=(LOSS_ELEMENTS_COUNT,1), device=DEVICE), torch.zeros(size=(LOSS_ELEMENTS_COUNT,1), device=DEVICE, dtype=torch.int64), torch.zeros(size=(LOSS_ELEMENTS_COUNT,1), device=DEVICE, dtype=torch.int64)
            with torch.no_grad():
                for batchNumber, batch in enumerate(dataloader):
                    # Compute prediction and loss
                    preds = model(batch.features)
                    eval_loss_batch, correct_batch, maxCorrect_batch = calculateLoss(batch.targets, preds, lossFunctions, calculateCorrect=True)
                    eval_loss += eval_loss_batch
                    correct  += correct_batch
                    maxCorrect  += maxCorrect_batch

                    # Visualise
                    if outPath:
                        os.makedirs(outPath, exist_ok=True)
                        batch_size = dataloader.batch_size
                        for sampleNumber in range(batch_size):      # sampleNumber = 0
                            # Sample data
                            # Sample data | Image
                            pathImage = Path(batch.meta.path_image[sampleNumber])
                            img_annot = cv.imread(str(pathImage), flags=cv.IMREAD_GRAYSCALE)
                            img_initial_size = img_annot.shape
                            
                            # Sample data | Ground truth
                            gt = {}
                            gt['row'] = batch.targets.row_line[sampleNumber].squeeze().cpu().numpy()
                            gt['col'] = batch.targets.col_line[sampleNumber].squeeze().cpu().numpy()

                            predictions = {}
                            predictions['row'] = preds.row[sampleNumber].squeeze().cpu().numpy()
                            predictions['col'] = preds.col[sampleNumber].squeeze().cpu().numpy()
                            outName = f'{pathImage.stem}.png'

                            # Sample data | Features
                            pathFeatures = pathImage.parent.parent / 'features' / f'{pathImage.stem}.json'
                            with open(pathFeatures, 'r') as f:
                                features = json.load(f)
                            features = {key: np.array(value) for key, value in features.items()}

                            # Draw
                            row_annot = []
                            col_annot = []

                            # Draw | Ground truth
                            gt_row = convert_01_array_to_visual(gt['row'], width=40)
                            row_annot.append(gt_row)
                            gt_col = convert_01_array_to_visual(gt['col'], width=40)
                            col_annot.append(gt_col)
                        
                            # Draw | Features | Text is startlike
                            indicator_textline_like_rowstart = convert_01_array_to_visual(features['row_between_textlines_like_rowstart'], width=20)
                            row_annot.append(indicator_textline_like_rowstart)
                            indicator_nearest_right_is_startlike = convert_01_array_to_visual(features['col_nearest_right_is_startlike_share'], width=20)
                            col_annot.append(indicator_nearest_right_is_startlike)

                            # Draw | Features | Words crossed (lighter = fewer words crossed)
                            wc_row = convert_01_array_to_visual(features['row_wordsCrossed_relToMax'], invert=True, width=20)
                            row_annot.append(wc_row)
                            wc_col = convert_01_array_to_visual(features['col_wordsCrossed_relToMax'], invert=True, width=20)
                            col_annot.append(wc_col)

                            # Draw | Features | Add feature bars
                            row_annot = np.concatenate(row_annot, axis=1)
                            img_annot = np.concatenate([img_annot, row_annot], axis=1)

                            col_annot = np.concatenate(col_annot, axis=1).T
                            col_annot = np.concatenate([col_annot, np.full(shape=(col_annot.shape[0], row_annot.shape[1]), fill_value=LUMINOSITY_FILLER, dtype=np.uint8)], axis=1)
                            img_annot = np.concatenate([img_annot, col_annot], axis=0)

                            # Draw | Predictions
                            img_predictions_row = np.full(img_annot.shape, fill_value=255, dtype=np.uint8)
                            indicator_predictions_row = convert_01_array_to_visual(1-predictions['row'], width=img_initial_size[1])
                            img_predictions_row[:indicator_predictions_row.shape[0], :indicator_predictions_row.shape[1]] = indicator_predictions_row
                            img_predictions_row = cv.cvtColor(img_predictions_row, code=cv.COLOR_GRAY2RGB)
                            img_predictions_row[:, :, 0] = 255
                            img_predictions_row = Image.fromarray(img_predictions_row).convert('RGBA')
                            img_predictions_row.putalpha(int(0.1*255))

                            img_predictions_col = np.full(img_annot.shape, fill_value=255, dtype=np.uint8)
                            indicator_predictions_col = convert_01_array_to_visual(1-predictions['col'], width=img_initial_size[0]).T
                            img_predictions_col[:indicator_predictions_col.shape[0], :indicator_predictions_col.shape[1]] = indicator_predictions_col
                            img_predictions_col = cv.cvtColor(img_predictions_col, code=cv.COLOR_GRAY2RGB)
                            img_predictions_col[:, :, 0] = 255
                            img_predictions_col = Image.fromarray(img_predictions_col).convert('RGBA')
                            img_predictions_col.putalpha(int(0.1*255))

                            img_annot_color = Image.fromarray(cv.cvtColor(img_annot, code=cv.COLOR_GRAY2RGB)).convert('RGBA')
                            img_predictions = Image.alpha_composite(img_predictions_col, img_predictions_row)
                            img_complete = Image.alpha_composite(img_annot_color, img_predictions).convert('RGB')

                            img_complete.save(outPath / f'{outName}', format='png')

            eval_loss = eval_loss / batchCount
            shareCorrect = correct / maxCorrect

            print(f'''Evaluation on best model
                Accuracy line-level: {(100*shareCorrect[0].item()):>0.1f}% (row) | {(100*shareCorrect[2].item()):>0.1f}% (col)
                Separator count (relative to truth): {(100*shareCorrect[1].item()):>0.1f}% (row) | {(100*shareCorrect[3].item()):>0.1f}% (col)
                Avg val loss: {eval_loss.sum().item():.3f} (total) | {eval_loss[0].item():.3f} (row-line) | {eval_loss[2].item():.3f} (col-line) | {eval_loss[1].item():.3f} (row-separator) | {eval_loss[3].item():.3f} (col-separator)''')
            
            if outPath:
                return outPath

        # Visualize results
        eval_loop(dataloader=dataloader_val, model=model, lossFunctions=lossFunctions, outPath=path_annotations_raw_val)
        timers['eval'] = perf_counter() - start_eval

    if TASKS['postprocess']:
        print(''*4, 'Post-processing', ''*4)
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
        

    # Reporting timings
    for key, value in timers.items():
        print(f'{key}: {value:.1f}')

if __name__ == '__main__':
    run()