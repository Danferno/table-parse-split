# Imports
import os
from pathlib import Path
from lxml import etree
from math import floor, ceil
from copy import deepcopy
from tqdm import tqdm
import tabledetect
import numpy as np
import cv2 as cv

# Constants
PATH_ROOT = Path(r"F:\ml-parsing-project\table-parse-split")
PATH_DATA = PATH_ROOT / 'data'

PATH_IN = Path(r"F:\ml-parsing-project\data")
PROJECT = "parse_activelearning1_jpg"

IMAGE_FORMAT = '.jpg'
THRESHOLD_EXPANSION = 50
THRESHOLD_PATTERN = 3
THRESHOLD_WHITES = 3

# Path things
pathLabels = PATH_IN / PROJECT / 'labels'
pathLabels_separators_narrow = PATH_DATA / 'labels' / 'narrow'
pathLabels_separators_wide = PATH_DATA / 'labels' / 'wide'
os.makedirs(pathLabels_separators_narrow, exist_ok=True)
os.makedirs(pathLabels_separators_wide, exist_ok=True)

# Convert row/column annotations to separator annotations
labelFiles = list(os.scandir(pathLabels))
for labelFileEntry in tqdm(labelFiles):       # labelFileEntry = labelFiles[0]
    # Image | Convert to monochrome B/W
    filename = os.path.splitext(labelFileEntry.name)[0]
    pathImg = PATH_IN / PROJECT / 'selected' / f'{filename}{IMAGE_FORMAT}'
    img = cv.imread(str(pathImg), cv.IMREAD_GRAYSCALE)
    thres, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    img01 = np.divide(img, 255).astype('float32')
    # cv.imshow('window', img); cv.waitKey(0); cv.destroyAllWindows

    # XML | Parse annotation
    root = etree.parse(labelFileEntry.path)
    tableEl = root.find('object[name="table"]')
    rows = root.findall('object[name="table row"]')
    cols = root.findall('object[name="table column"]')

    # XML | Table
    table = {el.tag:float(el.text) for el in tableEl.find('bndbox').getchildren()}
    for key, val in table.items():
        table[key] = floor(val) if 'min' in key else ceil(val)
    
    # XML | Rows 
    # XML | Rows | Get separation location
    row_edges = [(float(row.find('.//ymin').text), float(row.find('.//ymax').text)) for row in rows]
    row_edges = sorted(row_edges, key= lambda x: x[0])

    row_separators = [(row_edges[i][1], row_edges[i+1][0]) for i in range(len(row_edges)-1)]
    row_separators = [(floor(min(tuple)), floor(max(tuple))) for tuple in row_separators]

    row_bboxes_narrow = [{'xmin': table['xmin'], 'ymin': row_separator[0], 'xmax': table['xmax'], 'ymax': row_separator[1]} for row_separator in row_separators]

    # XML | Rows | Widen
    import matplotlib.pyplot as plt
    row_patterns = np.asarray([np.absolute(np.diff(row)).mean()*100 for row in img01])
    row_whites = np.asarray([np.mean(row)*100 for row in img01])

    row_bboxes_wide = []
    for rowbbox in row_bboxes_narrow:
        ymin = rowbbox['ymin']
        ymax = rowbbox['ymax']+1

        crop = img[ymin-THRESHOLD_EXPANSION:ymax+THRESHOLD_EXPANSION]
        # crop = img[ymin-2:ymax+2]
        # cv.imshow("cropped", crop)

        pattern_this = row_patterns[ymin:ymax]
        whites_this = row_whites[ymin:ymax]

        patterns_before = row_patterns[ymin-THRESHOLD_EXPANSION:ymin]
        patterns_before_difference = np.abs(patterns_before - pattern_this.reshape((-1, 1)))
        patterns_before_difference = np.min(patterns_before_difference, axis=0)
        patterns_before_similar = patterns_before_difference <= THRESHOLD_PATTERN
        try:
            patterns_before_furthestsimilar = np.where(np.flip(patterns_before_similar) == False)[0][0]
        except IndexError:
            patterns_before_furthestsimilar = THRESHOLD_EXPANSION

        patterns_after = row_patterns[ymax+1:ymax+1+THRESHOLD_EXPANSION]
        patterns_after_difference = np.abs(patterns_after - pattern_this.reshape((-1, 1)))
        patterns_after_difference = np.min(patterns_after_difference, axis=0)
        patterns_after_similar = patterns_after_difference <= THRESHOLD_PATTERN
        try:
            patterns_after_furthestsimilar = np.where(patterns_after_similar == False)[0][0]
        except IndexError:
            patterns_after_furthestsimilar = THRESHOLD_EXPANSION

        whites_before = row_whites[ymin-THRESHOLD_EXPANSION:ymin]
        whites_before_difference = np.abs(whites_before - whites_this.reshape((-1, 1)))
        whites_before_difference = np.min(whites_before_difference, axis=0)
        whites_before_similar = whites_before_difference <= THRESHOLD_WHITES
        try:
            whites_before_furthestsimilar = np.where(np.flip(whites_before_similar) == False)[0][0]
        except IndexError:
            whites_before_furthestsimilar = THRESHOLD_EXPANSION

        whites_after = row_whites[ymax+1:ymax+1+THRESHOLD_EXPANSION]
        whites_after_difference = np.abs(whites_after - whites_this.reshape((-1, 1)))
        whites_after_difference = np.min(whites_after_difference, axis=0)
        whites_after_similar = whites_after_difference <= THRESHOLD_WHITES
        try:
            whites_after_furthestsimilar = np.where(whites_after_similar == False)[0][0]
        except IndexError:
            whites_after_furthestsimilar = THRESHOLD_EXPANSION

        rowbbox_wide = deepcopy(rowbbox)
        rowbbox_wide['ymin'] = rowbbox_wide['ymin'] - min(patterns_before_furthestsimilar, whites_before_furthestsimilar)
        rowbbox_wide['ymax'] = rowbbox_wide['ymax'] + min(patterns_after_furthestsimilar, whites_after_furthestsimilar)

        if rowbbox_wide['ymin'] < 0:
            rowbbox_wide['ymin'] = 0
        if rowbbox_wide['ymax'] > img.shape[0]:
            rowbbox_wide['ymax'] = img.shape[0]

        row_bboxes_wide.append(rowbbox_wide)

    # XML | Columns
    # XML | Columns | Get separation location
    col_edges = [(float(col.find('.//xmin').text), float(col.find('.//xmax').text)) for col in cols]
    col_edges = sorted(col_edges, key= lambda x: x[0])

    col_separators = [(col_edges[i][1], col_edges[i+1][0]) for i in range(len(col_edges)-1)]
    col_separators = [(floor(min(tuple)), ceil(max(tuple))) for tuple in col_separators]

    col_bbox_narrow = [{'ymin': table['ymin'], 'xmin': col_separator[0], 'ymax': table['ymax'], 'xmax': col_separator[1]} for col_separator in col_separators]


    # XML | Write separator annotation
    # XML | Write separator annotation | Narrow
    separatorXml = etree.Element('annotation')
    for row_bbox in row_bboxes_narrow:        # object = extractedTable['objects'][0]
        xml_obj = etree.SubElement(separatorXml, 'object')
        xml_bbox = etree.SubElement(xml_obj, 'bndbox')
        for edge in ['xmin', 'ymin', 'xmax', 'ymax']:
            _ = etree.SubElement(xml_bbox, edge)
            _.text = str(row_bbox[edge])
        label = etree.SubElement(xml_obj, 'name'); label.text = 'row separator'
    for col_bbox in col_bbox_narrow:        # object = extractedTable['objects'][0]
        xml_obj = etree.SubElement(separatorXml, 'object')
        xml_bbox = etree.SubElement(xml_obj, 'bndbox')
        for edge in ['xmin', 'ymin', 'xmax', 'ymax']:
            _ = etree.SubElement(xml_bbox, edge)
            _.text = str(col_bbox[edge])
        label = etree.SubElement(xml_obj, 'name'); label.text = 'column separator'

    tree = etree.ElementTree(separatorXml)
    tree.write(pathLabels_separators_narrow / labelFileEntry.name, pretty_print=True, xml_declaration=False, encoding="utf-8") 

    # XML | Write separator annotation | Wide
    separatorXml = etree.Element('annotation')
    for row_bbox in row_bboxes_wide:        # object = extractedTable['objects'][0]
        xml_obj = etree.SubElement(separatorXml, 'object')
        xml_bbox = etree.SubElement(xml_obj, 'bndbox')
        for edge in ['xmin', 'ymin', 'xmax', 'ymax']:
            _ = etree.SubElement(xml_bbox, edge)
            _.text = str(row_bbox[edge])
        label = etree.SubElement(xml_obj, 'name'); label.text = 'row separator'
    # for col_bbox in col_bbox_narrow:        # object = extractedTable['objects'][0]
    #     xml_obj = etree.SubElement(separatorXml, 'object')
    #     xml_bbox = etree.SubElement(xml_obj, 'bndbox')
    #     for edge in ['xmin', 'ymin', 'xmax', 'ymax']:
    #         _ = etree.SubElement(xml_bbox, edge)
    #         _.text = str(col_bbox[edge])
    #     label = etree.SubElement(xml_obj, 'name'); label.text = 'column separator'

    tree = etree.ElementTree(separatorXml)
    tree.write(pathLabels_separators_wide / labelFileEntry.name, pretty_print=True, xml_declaration=False, encoding="utf-8") 

# Plot
tabledetect.utils.visualise_annotation(path_images=PATH_IN / PROJECT / 'selected', path_labels=pathLabels_separators_narrow, path_output=PATH_DATA / 'images_annotated' / 'narrow', annotation_type=None, annotation_format={'labelFormat': 'voc', 'labels': ['row separator', 'column separator'], 'classMap': None, 'split_annotation_types': False, 'show_labels': False})
tabledetect.utils.visualise_annotation(path_images=PATH_IN / PROJECT / 'selected', path_labels=pathLabels_separators_wide, path_output=PATH_DATA / 'images_annotated' / 'wide', annotation_type=None, annotation_format={'labelFormat': 'voc', 'labels': ['row separator', 'column separator'], 'classMap': None, 'split_annotation_types': False, 'show_labels': False, 'as_area': True})