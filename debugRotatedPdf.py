# Imports
from pathlib import Path
from utils import makeDirs, yolo_to_pilbox
import process
from PIL import Image
import tabledetect
import shutil

# Parameters
PATH_ROOT = Path(r"F:\datatog-junkyard\debugRotatedPdf")
path_models = Path(r'F:\ml-parsing-project\models')
replace_dirs = True
image_format = '.png'
padding = 40
threshold_detect=0.7

path_model_detect=path_models / 'codamo-tabledetect-best.pt'
path_model_parse_line=path_models / 'codamo-tableparse-line-best.pt'
path_model_parse_separator=path_models / 'codamo-tableparse-separator-best.pt'

path_out = PATH_ROOT
path_out_tables_images = path_out / 'tables_images'
path_meta = path_out / 'meta'
makeDirs(path_meta, replaceDirs='overwrite')
makeDirs(path_out_tables_images, replaceDirs=replace_dirs)
path_images = path_out / 'pages_images'
path_labels = path_out / 'tables_bboxes'

# Table detect
tabledetect.detect_table(path_weights=path_model_detect, path_input=path_out / 'pages_images', path_output=path_out / 'temp', image_format='.png', threshold_confidence=threshold_detect, save_visual_output=True)
shutil.copytree(src=path_out / 'temp' / 'out' / 'table-detect' / 'labels', dst=path_out / 'tables_bboxes', dirs_exist_ok=True)
shutil.rmtree(path_out / 'temp')

# Detect to cropped tables > 1385 x 518
pageName = '2021-30300200-p12'
img = Image.open(path_images / f'{pageName}{image_format}')
table_bboxes = yolo_to_pilbox(yoloPath=path_labels / f'{pageName}.txt', targetImage=img)        # x0 y0 x1 y1

for idx, bbox in enumerate(table_bboxes):       # idx = 0; bbox = table_bboxes[idx]
    img_cropped = img.crop(bbox)
    img_padded = Image.new(img_cropped.mode, (img_cropped.width+padding*2, img_cropped.height+padding*2), 255)
    img_padded.paste(img_cropped, (padding, padding))
    img_padded.save(path_out_tables_images / f'{pageName}_t{idx}{image_format}')

# Predict and process
process.preprocess_lineLevel(path_images=path_out / 'tables_images', path_pdfs=path_out / 'pdfs', path_out=path_out, path_data_skew=path_out / 'meta' / 'skewAngles', replace_dirs=replace_dirs, n_workers=1)

# Apply line-level model and preprocess separator-level features
process.preprocess_separatorLevel(path_model_line=path_model_parse_line, path_data=path_out, replace_dirs=replace_dirs)

# # Apply separator-level model and process out
process.predict_and_process(path_model_file=path_model_parse_separator, path_data=path_out, replace_dirs=replace_dirs, out_data=True, out_images=True, out_labels_rows=True)