# Imports
import json
import cv2 as cv
from PIL import Image
from tqdm import tqdm

from pathlib import Path
import numpy as np
import torch        # type : ignore

import utils
from dataloader import get_dataloader
from model import TableLineModel, LOSS_ELEMENTS_COUNT
from loss import getLossFunctions, calculateLoss

# Helper functions
def __convert_01_array_to_visual(array, max_luminosity_features, invert=False, width=40) -> np.array:
    luminosity = (1 - array) * max_luminosity_features if invert else array * max_luminosity_features
    luminosity = luminosity.round(0).astype(np.uint8)
    luminosity = np.expand_dims(luminosity, axis=1)
    luminosity = np.broadcast_to(luminosity, shape=(luminosity.shape[0], width))
    return luminosity

def eval_loop(dataloader, model, lossFunctions, max_luminosity_features, luminosity_filler, device, draw_images,path_annotations_raw, prediction_cutoff=0.5):
    batchCount = len(dataloader)
    eval_loss, correct, maxCorrect = torch.zeros(size=(LOSS_ELEMENTS_COUNT,1), device=device), torch.zeros(size=(LOSS_ELEMENTS_COUNT,1), device=device, dtype=torch.int64), torch.zeros(size=(LOSS_ELEMENTS_COUNT,1), device=device, dtype=torch.int64)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Eval | Looping over batches'):
            # Compute prediction and loss
            preds = model(batch.features)
            eval_loss_batch, correct_batch, maxCorrect_batch = calculateLoss(batch.targets, preds, lossFunctions, calculateCorrect=True)
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
                    pathFeatures = pathImage.parent.parent / 'features' / f'{pathImage.stem}.json'
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

# Function
def evaluate(path_model_file, path_data, max_luminosity_features=240, luminosity_filler=255, device='cuda', replace_dirs='warn', draw_images=True, path_annotations_raw=None):
    # Parse parameters
    path_model_file = Path(path_model_file); path_data = Path(path_data)
    
    # Make folders
    path_annotations_raw = path_annotations_raw or path_model_file.parent / 'annotated'
    utils.makeDirs(path_annotations_raw, replaceDirs=replace_dirs)

    # Load model
    model = TableLineModel().to(device)
    model.load_state_dict(torch.load(path_model_file))
    model.eval()
    dataloader = get_dataloader(dir_data=path_data)
    lossFunctions = getLossFunctions(path_model_file=path_model_file)

    # Predict
    # Visualize results
    eval_loop(dataloader=dataloader, model=model, lossFunctions=lossFunctions, path_annotations_raw=path_annotations_raw, max_luminosity_features=max_luminosity_features, luminosity_filler=luminosity_filler, device=device, draw_images=draw_images)

# Individual file run for testing
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    path_model = Path(r"F:\ml-parsing-project\table-parse-split\models\test\model_best.pt") 
    path_data = Path(r"F:\ml-parsing-project\table-parse-split\data\real_narrow")

    evaluate(path_model_file=path_model, path_data=path_data / 'val', device=device)