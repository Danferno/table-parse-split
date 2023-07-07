# Imports
import json
import cv2 as cv
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from pathlib import Path
import numpy as np
import warnings
import torch        # type : ignore

import utils
import dataloaders
from model import (TableLineModel, LOSS_ELEMENTS_LINELEVEL_COUNT,
                   TableSeparatorModel, LOSS_ELEMENTS_SEPARATORLEVEL_COUNT)
from loss import getLossFunctions, calculateLoss_lineLevel, calculateLoss_separatorLevel, getLossFunctions_separatorLevel

# Helper functions
def __convert_01_array_to_visual(array, max_luminosity_features, invert=False, width=40) -> np.array:
    luminosity = (1 - array) * max_luminosity_features if invert else array * max_luminosity_features
    luminosity = luminosity.round(0).astype(np.uint8)
    luminosity = np.expand_dims(luminosity, axis=1)
    luminosity = np.broadcast_to(luminosity, shape=(luminosity.shape[0], width))
    return luminosity

def __expand_dim(array, correct_dim_length=2):
    # This is a bug correction, need to fix something in preds_to_separators in process.py (something weird about torch tensors)
    return np.expand_dims(array, axis=0) if array.ndim < correct_dim_length else array

# Function
def evaluate_lineLevel(path_model_file, path_data, max_luminosity_features=240, luminosity_filler=255, device='cuda', replace_dirs='warn', draw_images=True, path_annotations_raw=None):
    # Parse parameters
    path_model_file = Path(path_model_file); path_data = Path(path_data)
    
    # Make folders
    path_annotations_raw = path_annotations_raw or path_model_file.parent / 'annotated'
    utils.makeDirs(path_annotations_raw, replaceDirs=replace_dirs)

    # Load model
    model = TableLineModel().to(device)
    model.load_state_dict(torch.load(path_model_file))
    model.eval()
    dataloader = dataloaders.get_dataloader_lineLevel(dir_data=path_data, ground_truth=True)
    lossFunctions = getLossFunctions(path_model_file=path_model_file)

    # Evaluate
    def eval_loop(dataloader, model, lossFunctions, max_luminosity_features, luminosity_filler, device, draw_images,path_annotations_raw, prediction_cutoff=0.5):
        batchCount = len(dataloader)
        eval_loss, correct, maxCorrect = torch.zeros(size=(LOSS_ELEMENTS_LINELEVEL_COUNT,1), device=device), torch.zeros(size=(LOSS_ELEMENTS_LINELEVEL_COUNT,1), device=device, dtype=torch.int64), torch.zeros(size=(LOSS_ELEMENTS_LINELEVEL_COUNT,1), device=device, dtype=torch.int64)
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Eval | Looping over batches'):
                # Compute prediction and loss
                preds = model(batch.features)
                eval_loss_batch, correct_batch, maxCorrect_batch = calculateLoss_lineLevel(batch.targets, preds, lossFunctions, calculateCorrect=True)
                eval_loss += eval_loss_batch
                correct  += correct_batch
                maxCorrect  += maxCorrect_batch

                # Visualise
                if draw_images:
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
                        pathFeatures = pathImage.parent.parent / 'features_lineLevel' / f'{pathImage.stem}.json'
                        with open(pathFeatures, 'r') as f:
                            features = json.load(f)
                        features = {key: np.array(value) for key, value in features.items()}

                        # Draw
                        row_annot = []
                        col_annot = []

                        # Draw | Ground truth
                        gt_row = __convert_01_array_to_visual(gt['row'], width=40, max_luminosity_features=max_luminosity_features)
                        row_annot.append(gt_row)
                        gt_col = __convert_01_array_to_visual(gt['col'], width=40, max_luminosity_features=max_luminosity_features)
                        col_annot.append(gt_col)
                    
                        # Draw | Features | Text is startlike
                        indicator_textline_like_rowstart = __convert_01_array_to_visual(features['row_between_textlines_like_rowstart'], width=20, max_luminosity_features=max_luminosity_features)
                        row_annot.append(indicator_textline_like_rowstart)
                        indicator_nearest_right_is_startlike = __convert_01_array_to_visual(features['col_nearest_right_is_startlike_share'], width=20, max_luminosity_features=max_luminosity_features)
                        col_annot.append(indicator_nearest_right_is_startlike)

                        # Draw | Features | Words crossed (lighter = fewer words crossed)
                        wc_row = __convert_01_array_to_visual(features['row_wordsCrossed_relToMax'], invert=True, width=20, max_luminosity_features=max_luminosity_features)
                        row_annot.append(wc_row)
                        wc_col = __convert_01_array_to_visual(features['col_wordsCrossed_relToMax'], invert=True, width=20, max_luminosity_features=max_luminosity_features)
                        col_annot.append(wc_col)

                        # Draw | Features | Add feature bars
                        row_annot = np.concatenate(row_annot, axis=1)
                        img_annot = np.concatenate([img_annot, row_annot], axis=1)

                        col_annot = np.concatenate(col_annot, axis=1).T
                        col_annot = np.concatenate([col_annot, np.full(shape=(col_annot.shape[0], row_annot.shape[1]), fill_value=luminosity_filler, dtype=np.uint8)], axis=1)
                        img_annot = np.concatenate([img_annot, col_annot], axis=0)

                        # Draw | Predictions
                        img_predictions_row = np.full(img_annot.shape, fill_value=255, dtype=np.uint8)
                        predictions['row'][predictions['row'] < prediction_cutoff] = predictions['row'][predictions['row'] < prediction_cutoff] / (prediction_cutoff * 10)
                        indicator_predictions_row = __convert_01_array_to_visual(1-predictions['row'], width=img_initial_size[1], max_luminosity_features=max_luminosity_features)
                        img_predictions_row[:indicator_predictions_row.shape[0], :indicator_predictions_row.shape[1]] = indicator_predictions_row
                        img_predictions_row = cv.cvtColor(img_predictions_row, code=cv.COLOR_GRAY2RGB)
                        img_predictions_row[:, :, 0] = 255
                        img_predictions_row = Image.fromarray(img_predictions_row).convert('RGBA')
                        img_predictions_row.putalpha(int(0.1*255))

                        img_predictions_col = np.full(img_annot.shape, fill_value=255, dtype=np.uint8)
                        predictions['col'][predictions['col'] < prediction_cutoff] = predictions['col'][predictions['col'] < prediction_cutoff] / (prediction_cutoff * 10)
                        indicator_predictions_col = __convert_01_array_to_visual(1-predictions['col'], width=img_initial_size[0], max_luminosity_features=max_luminosity_features).T
                        img_predictions_col[:indicator_predictions_col.shape[0], :indicator_predictions_col.shape[1]] = indicator_predictions_col
                        img_predictions_col = cv.cvtColor(img_predictions_col, code=cv.COLOR_GRAY2RGB)
                        img_predictions_col[:, :, 0] = 255
                        img_predictions_col = Image.fromarray(img_predictions_col).convert('RGBA')
                        img_predictions_col.putalpha(int(0.1*255))

                        img_annot_color = Image.fromarray(cv.cvtColor(img_annot, code=cv.COLOR_GRAY2RGB)).convert('RGBA')
                        img_predictions = Image.alpha_composite(img_predictions_col, img_predictions_row)
                        img_complete = Image.alpha_composite(img_annot_color, img_predictions).convert('RGB')

                        img_complete.save(path_annotations_raw / f'{outName}', format='png')

        eval_loss = eval_loss / batchCount
        shareCorrect = correct / maxCorrect

        print(f'''Eval | Model statistics
            Accuracy line-level: {(100*shareCorrect[0].item()):>0.1f}% (row) | {(100*shareCorrect[2].item()):>0.1f}% (col)
            Separator count (relative to truth): {(100*shareCorrect[1].item()):>0.1f}% (row) | {(100*shareCorrect[3].item()):>0.1f}% (col)
            Avg val loss: {eval_loss.sum().item():.3f} (total) | {eval_loss[0].item():.3f} (row-line) | {eval_loss[2].item():.3f} (col-line) | {eval_loss[1].item():.3f} (row-separator) | {eval_loss[3].item():.3f} (col-separator)''')
        
        if draw_images:
            return path_annotations_raw

    eval_loop(dataloader=dataloader, model=model, lossFunctions=lossFunctions, path_annotations_raw=path_annotations_raw, max_luminosity_features=max_luminosity_features, luminosity_filler=luminosity_filler, device=device, draw_images=draw_images)

def evaluate_separatorLevel(path_model_file, path_data, path_annotated_images=None, device='cuda', replace_dirs='warn', draw_images=True, draw_text_scale=1):
    # Parameters
    path_model_file = Path(path_model_file); path_data = Path(path_data)
    path_annotated_images = path_annotated_images or path_model_file.parent / 'annotated'
    font_text = ImageFont.truetype('cour.ttf', size=int(16*draw_text_scale))

    # Make folders
    utils.makeDirs(path_annotated_images, replaceDirs=replace_dirs)

    # Load model
    model = TableSeparatorModel().to(device)
    model.load_state_dict(torch.load(path_model_file))
    model.eval()
    dataloader = dataloaders.get_dataloader_separatorLevel(dir_data=path_data, ground_truth=True)
    lossFunctions = getLossFunctions_separatorLevel(path_model_file=path_model_file)

    # Evaluate
    def eval_loop(dataloader, model, lossFunctions, draw_images, path_annotated_images, device='cuda'):
        batchCount = len(dataloader)
        eval_loss, correct, maxCorrect = torch.zeros(size=(LOSS_ELEMENTS_SEPARATORLEVEL_COUNT,1), device=device), torch.zeros(size=(LOSS_ELEMENTS_SEPARATORLEVEL_COUNT,1), device=device, dtype=torch.int64), torch.zeros(size=(LOSS_ELEMENTS_SEPARATORLEVEL_COUNT,1), device=device, dtype=torch.int64)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Eval | Looping over batches'):
                # Compute prediction and loss
                preds = model(batch.features)
                eval_loss_batch, correct_batch, maxCorrect_batch = calculateLoss_separatorLevel(batch.targets, preds, lossFunctions, calculateCorrect=True)
                eval_loss += eval_loss_batch
                correct  += correct_batch
                maxCorrect  += maxCorrect_batch

                # Visualise
                if draw_images:
                    batch_size = dataloader.batch_size
                    for sampleNumber in range(batch_size):
                        # Sample data
                        # Sample data | Image
                        path_image = Path(batch.meta.path_image[sampleNumber])

                        # Sample data | Targets
                        targets = {}
                        targets['row'] = __expand_dim(batch.targets.row[sampleNumber].squeeze().cpu().numpy(), correct_dim_length=1)
                        targets['col'] = __expand_dim(batch.targets.col[sampleNumber].squeeze().cpu().numpy(), correct_dim_length=1)

                        # Sample data | Predictions
                        predictions = {}
                        predictions['row'] = __expand_dim((preds.row[sampleNumber].squeeze().cpu().numpy()), correct_dim_length=1) > 0.5
                        predictions['col'] = __expand_dim((preds.col[sampleNumber].squeeze().cpu().numpy()), correct_dim_length=1) > 0.5

                        # Sample data | Separator locations
                        locations = {}
                        locations['row'] = __expand_dim(batch.features.proposedSeparators_row[sampleNumber].squeeze(0).cpu().numpy())
                        locations['col'] = __expand_dim(batch.features.proposedSeparators_col[sampleNumber].squeeze(0).cpu().numpy())

                        if draw_images:
                            # Load image
                            img = Image.open(path_image).convert('RGBA')

                            # Rows
                            overlay_row = Image.new('RGBA', img.size, (0,0,0,0))
                            img_annot_row = ImageDraw.Draw(overlay_row)

                            for idx, separator in enumerate(locations['row']):
                                shape = [0, separator[0], img.width, separator[1]]
                                color_fill = None
                                if (predictions['row'][idx] == targets['row'][idx]):
                                    color_fill = utils.COLOR_CORRECT if predictions['row'][idx] else utils.COLOR_MIDDLE
                                else:
                                    color_fill = utils.COLOR_WRONG

                                img_annot_row.rectangle(xy=shape, fill=color_fill, width=1)
                            
                            # Cols
                            overlay_col = Image.new('RGBA', img.size, (0,0,0,0))
                            img_annot_col = ImageDraw.Draw(overlay_col)

                            for idx, separator in enumerate(locations['col']):
                                shape = [separator[0], 0, separator[1], img.height]
                                color_fill = None
                                if (predictions['col'][idx] == targets['col'][idx]):
                                    color_fill = utils.COLOR_CORRECT if predictions['col'][idx] else utils.COLOR_MIDDLE
                                else:
                                    color_fill = utils.COLOR_WRONG

                                img_annot_col.rectangle(xy=shape, fill=color_fill, width=1)

                            # Legend
                            img_annot_row.text(xy=(img.width-10, img.height-10), anchor='rd', text='Green: correct retention, orange: correct removal, red: incorrect', fill='black', font=font_text)

                            # Statistics
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore')
                                row_share_correct_retained = (predictions['row'] == targets['row'])[targets['row'] == True].mean()
                                row_share_correct_removed = (predictions['row'] == targets['row'])[targets['row'] == False].mean()
                                row_share_wrong = (predictions['row'] != targets['row']).mean()
                                row_count_wrong = (predictions['row'] != targets['row']).sum()
                                col_share_correct_retained = (predictions['col'] == targets['col'])[targets['col'] == True].mean()
                                col_share_correct_removed = (predictions['col'] == targets['col'])[targets['col'] == False].mean()
                                col_share_wrong = (predictions['col'] != targets['col']).mean()
                                col_count_wrong = (predictions['col'] != targets['col']).sum()
                            text = f'Row | True positive ({row_share_correct_retained:.0%}) - True negative ({row_share_correct_removed:.0%}) - Wrong ({row_share_wrong:.0%})[{row_count_wrong}]\nCol | True positive ({col_share_correct_retained:.0%}) - True negative ({col_share_correct_removed:.0%}) - Wrong ({col_share_wrong:.0%})[{col_count_wrong}]'
                            img_annot_row.multiline_text(xy=(0, 0), text=text, fill='black', font=font_text)

                            # Save image
                            img = Image.alpha_composite(img, overlay_row)
                            img = Image.alpha_composite(img, overlay_col).convert('RGB')
                            img.save(path_annotated_images / f'{path_image.stem}.png')
                        
        eval_loss = eval_loss / batchCount
        shareCorrect = correct / maxCorrect

        print(f'''Eval | Model statistics
            Accuracy separator-level: {(100*shareCorrect[0].item()):>0.1f}% (row) | {(100*shareCorrect[1].item()):>0.1f}% (col)
            Avg val loss: {eval_loss.sum().item():.3f} (total) | {eval_loss[0].item():.3f} (row) | {eval_loss[1].item():.3f} (col)''')
        
    eval_loop(dataloader=dataloader, model=model, lossFunctions=lossFunctions, path_annotated_images=path_annotated_images, device=device, draw_images=draw_images)
        


# Individual file run for testing
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    path_model = Path(r"F:\ml-parsing-project\table-parse-split\models\test\model_best.pt") 
    path_data = Path(r"F:\ml-parsing-project\table-parse-split\data\real_narrow")

    evaluate_lineLevel(path_model_file=path_model, path_data=path_data / 'val', device=device)