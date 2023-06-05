# Run collectTableparsedata script first (this script requires all albels in 'ml/data/labels' folder)
# Imports
import os
from pathlib import Path
from lxml import etree
from math import floor, ceil
from copy import deepcopy
from tqdm import tqdm
import tabledetect
import shutil
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from typing import Literal
from datetime import datetime
import json
import pandas as pd

# Constants
PATH_ROOT = Path(r"F:\ml-parsing-project\table-parse-split")
PATH_DATA = PATH_ROOT / 'data'

PATH_IN = Path(r"F:\ml-parsing-project\data")
PROJECT = "parse_activelearning1_jpg"

IMAGE_FORMAT = '.jpg'
THRESHOLDS_ROWS = {'expansion': 50, 'pattern': 2.5, 'whites': 2.5}
THRESHOLDS_COLUMNS = {'expansion': 50, 'pattern': 0.5, 'whites': 0.5}
THRESHOLDS_SPAN_OVERFLOW_PIXEL = {'row': 8, 'column': 6}

# Path things
pathLabels = PATH_IN / PROJECT / 'labels'
pathLabels_separators_narrow = PATH_DATA / 'labels' / 'narrow'
pathLabels_separators_wide = PATH_DATA / 'labels' / 'wide'
pathLabels_yolo = PATH_DATA / 'labels_yolo'
pathErrors = PATH_DATA / 'errors' / f"{datetime.now().strftime('%Y_%m_%d-%H_%M')}.tsv"

def replaceDirs(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        shutil.rmtree(path)
        os.makedirs(path)

os.makedirs(PATH_DATA, exist_ok=True)
replaceDirs(pathLabels_separators_narrow)
replaceDirs(pathLabels_separators_wide)
replaceDirs(pathLabels_yolo)
replaceDirs(pathErrors.parent)


# Errors
class MinMaxError(ValueError):
    ...

# Helper functions
def tryCatch(func):
    def wrapper_tryCatch(bbox, **kwargs):
        try:
            return func(bbox, **kwargs)

        except ValueError:
            return bbox
    return wrapper_tryCatch

@tryCatch
def widenBbox(bbox, minCoord:Literal['ymin', 'xmin'], maxCoord:Literal['ymax', 'xmax'], shapeIndex, patterns, whites, thresholds):
    # Get edges of narrow bbox
    minEdge = bbox[minCoord]
    maxEdge = bbox[maxCoord]+1

    # crop = img[ymin-THRESHOLD_EXPANSION:ymax+THRESHOLD_EXPANSION]
    # crop = img[ymin-2:ymax+2]
    # cv.imshow("cropped", crop)

    pattern_this = patterns[minEdge:maxEdge]
    whites_this = whites[minEdge:maxEdge]

    patterns_before = patterns[minEdge-thresholds['expansion']:minEdge]
    patterns_before_difference = np.abs(patterns_before - pattern_this.reshape((-1, 1)))
    patterns_before_difference = np.min(patterns_before_difference, axis=0)
    patterns_before_similar = patterns_before_difference <= thresholds['pattern']
    try:
        patterns_before_furthestsimilar = np.where(np.flip(patterns_before_similar) == False)[0][0]
    except IndexError:
        patterns_before_furthestsimilar = thresholds['expansion']

    patterns_after = patterns[maxEdge+1:maxEdge+1+thresholds['expansion']]
    patterns_after_difference = np.abs(patterns_after - pattern_this.reshape((-1, 1)))
    patterns_after_difference = np.min(patterns_after_difference, axis=0)
    patterns_after_similar = patterns_after_difference <= thresholds['pattern']
    try:
        patterns_after_furthestsimilar = np.where(patterns_after_similar == False)[0][0]
    except IndexError:
        patterns_after_furthestsimilar = thresholds['expansion']

    whites_before = whites[minEdge-thresholds['expansion']:minEdge]
    whites_before_difference = np.abs(whites_before - whites_this.reshape((-1, 1)))
    whites_before_difference = np.min(whites_before_difference, axis=0)
    whites_before_similar = whites_before_difference <= thresholds['whites']
    try:
        whites_before_furthestsimilar = np.where(np.flip(whites_before_similar) == False)[0][0]
    except IndexError:
        whites_before_furthestsimilar = thresholds['expansion']

    whites_after = whites[maxEdge+1:maxEdge+1+thresholds['expansion']]
    whites_after_difference = np.abs(whites_after - whites_this.reshape((-1, 1)))
    whites_after_difference = np.min(whites_after_difference, axis=0)
    whites_after_similar = whites_after_difference <= thresholds['whites']
    try:
        whites_after_furthestsimilar = np.where(whites_after_similar == False)[0][0]
    except IndexError:
        whites_after_furthestsimilar = thresholds['expansion']

    bbox_wide = deepcopy(bbox)
    bbox_wide[minCoord] = bbox_wide[minCoord] - min(patterns_before_furthestsimilar, whites_before_furthestsimilar)
    bbox_wide[maxCoord] = bbox_wide[maxCoord] + min(patterns_after_furthestsimilar, whites_after_furthestsimilar)

    if bbox_wide[minCoord] < 0:
        bbox_wide[minCoord] = 0
    if bbox_wide[maxCoord] > img.shape[shapeIndex]:
        bbox_wide[maxCoord] = img.shape[shapeIndex]

    return bbox_wide

def constrainToShape(bbox, shape):
    constrainedBbox = {}
    constrainedBbox['xmin'] = max(0, bbox['xmin'])
    constrainedBbox['ymin'] = max(0, bbox['ymin'])
    constrainedBbox['xmax'] = min(shape[1], bbox['xmax'])
    constrainedBbox['ymax'] = min(shape[0], bbox['ymax'])
    return constrainedBbox

def findClosestSeparators(rowSeparatorBboxes, colSeparatorBboxes, cellBbox):
    def findByOrientation(separatorBboxes, orientation:Literal['row', 'column']):
        minCoord = 'ymin' if orientation == 'row' else 'xmin'
        maxCoord = 'ymax' if orientation == 'row' else 'xmax'
        
        # First edge
        minCoord_value = cellBbox[minCoord]
        smallestValue = min([separatorBbox[minCoord] for separatorBbox in separatorBboxes])

        if minCoord_value <= smallestValue - THRESHOLDS_SPAN_OVERFLOW_PIXEL[orientation]:
            minCoord_new = minCoord_value
        else:
            borderDistances = [abs(separatorBbox[minCoord]-minCoord_value) for separatorBbox in separatorBboxes]
            closestSeparator = separatorBboxes[borderDistances.index(min(borderDistances))]
            if closestSeparator[minCoord] >= minCoord_value:
                minCoord_new = closestSeparator[minCoord] + 1
            else:
                minCoord_new = minCoord_value

        # Last edge
        maxCoord_value = cellBbox[maxCoord]
        largestValue = max([separatorBbox[maxCoord] for separatorBbox in separatorBboxes])

        if maxCoord_value >= largestValue + THRESHOLDS_SPAN_OVERFLOW_PIXEL[orientation]:
            maxCoord_new = maxCoord_value
        else:
            borderDistances = [abs(separatorBbox[maxCoord]-maxCoord_value) for separatorBbox in separatorBboxes]
            closestSeparator = separatorBboxes[borderDistances.index(min(borderDistances))]
            if closestSeparator[maxCoord] <= maxCoord_value:
                maxCoord_new = closestSeparator[maxCoord] - 1
            else:
                maxCoord_new = maxCoord_value
        
        return {minCoord: minCoord_new, maxCoord: maxCoord_new}
    rowCoords_new = findByOrientation(separatorBboxes=rowSeparatorBboxes, orientation='row')
    colCoords_new = findByOrientation(separatorBboxes=colSeparatorBboxes, orientation='column')
    cellBbox_new = colCoords_new | rowCoords_new

    return cellBbox_new

def isCellSeparatorEdge(separatorBbox, cellBbox, orientation:Literal['row', 'column']) -> bool:
    minCoord = 'ymin' if orientation == 'row' else 'xmin'
    maxCoord = 'ymax' if orientation == 'row' else 'xmax'

    # Check first edge
    edge1_someOverlap       = (separatorBbox[maxCoord] >= cellBbox[minCoord])
    edge1_incompleteOverlap = (separatorBbox[minCoord] <= cellBbox[minCoord])
    if (edge1_someOverlap) and (edge1_incompleteOverlap):
        return True

    # Check second edge
    edge2_someOverlap       = (separatorBbox[minCoord] <= cellBbox[maxCoord])
    edge2_incompleteOverlap = (separatorBbox[maxCoord] >= cellBbox[maxCoord])
    if (edge2_someOverlap) and (edge2_incompleteOverlap):
        return True
    
    # Neither are edge
    return False

def cellToInterior(rowSeparatorBboxes:list[dict], colSeparatorBboxes:list[dict], cellBbox, bboxConstraint:tuple):
    # Check which separators will slice
    borderEdges = {}
    borderEdges['xmin'] = [separatorBbox['xmax'] for separatorBbox in colSeparatorBboxes if (separatorBbox['xmin'] <= cellBbox['xmin']) and (separatorBbox['xmax'] >= cellBbox['xmin'])]
    borderEdges['xmax'] = [separatorBbox['xmin'] for separatorBbox in colSeparatorBboxes if (separatorBbox['xmax'] >= cellBbox['xmax']) and (separatorBbox['xmin'] <= cellBbox['xmax'])]
    borderEdges['ymin'] = [separatorBbox['ymax'] for separatorBbox in rowSeparatorBboxes if (separatorBbox['ymin'] <= cellBbox['ymin']) and (separatorBbox['ymax'] >= cellBbox['ymin'])]
    borderEdges['ymax'] = [separatorBbox['ymin'] for separatorBbox in rowSeparatorBboxes if (separatorBbox['ymax'] >= cellBbox['ymax']) and (separatorBbox['ymin'] <= cellBbox['ymax'])]

    # Set border to separator edge +1 if present, otherwise retain original
    interiorCell = {}
    for edgeName in borderEdges:
        adjustmentFactor = 1 if 'min' in edgeName else -1
        interiorCell[edgeName] = cellBbox[edgeName] if not borderEdges[edgeName] else borderEdges[edgeName][0] + adjustmentFactor
    interiorCell = constrainToShape(bbox=interiorCell, shape=bboxConstraint)

    # Check cell dimensions for errors (normally caused by separator eclipsing cells)
    errorMessage = 'Faulty annotation ({}): max exceeds min.'
    if (interiorCell['xmin'] >= interiorCell['xmax']):
        raise MinMaxError(errorMessage.format('col'))
    if (interiorCell['ymin'] >= interiorCell['ymax']):
        raise MinMaxError(errorMessage.format('row'))
        
    return interiorCell

def writeBboxToXml(bboxes:list, xmlRoot:etree.Element, label:str):
    if not isinstance(bboxes, list):
        bboxes = [bboxes]
    for bbox in bboxes:
        xml_obj = etree.SubElement(xmlRoot, 'object')
        xml_bbox = etree.SubElement(xml_obj, 'bndbox')
        for edge in ['xmin', 'ymin', 'xmax', 'ymax']:
            _ = etree.SubElement(xml_bbox, edge)
            _.text = str(bbox[edge])
        xml_label = etree.SubElement(xml_obj, 'name'); xml_label.text = label

# Convert row/column annotations to separator annotations
labelFiles = list(os.scandir(pathLabels))
with open(PATH_IN / PROJECT / 'rotated_images.txt', 'r') as file:
    rotatedFiles = [line.strip() for line in file.readlines()]

for labelFileEntry in tqdm(labelFiles):       # labelFileEntry = labelFiles[0]
    # Skip rotated images
    rotated = any([rotatedFilename in labelFileEntry.name for rotatedFilename in rotatedFiles])
    if rotated:
        print('Skipping rotated file')
        continue
    
    faultyAnnotation = []

    # Image | Convert to monochrome B/W
    filename = os.path.splitext(labelFileEntry.name)[0]
    pathImg = PATH_IN / PROJECT / 'selected' / f'{filename}{IMAGE_FORMAT}'
    img = cv.imread(str(pathImg), cv.IMREAD_GRAYSCALE)
    thres, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    img01 = np.divide(img, 255).astype('float32')

    # XML | Parse annotation
    root = etree.parse(labelFileEntry.path)
    objectCount = len(root.findall('.//object'))
    if objectCount != 0:
        tableEl = root.find('object[name="table"]')
        rows = root.findall('object[name="table row"]')
        cols = root.findall('object[name="table column"]')
        spanners = root.findall('object[name="table spanning cell"]')

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
        row_patterns = np.asarray([np.absolute(np.diff(row)).mean()*100 for row in img01])
        row_whites = np.asarray([np.mean(row)*100 for row in img01])

        row_bboxes_wide = [widenBbox(bbox=rowbbox, minCoord='ymin', maxCoord='ymax', shapeIndex=0, patterns=row_patterns, whites=row_whites, thresholds=THRESHOLDS_ROWS) for rowbbox in row_bboxes_narrow]
        if not len(row_bboxes_wide) == len([dict(t) for t in {tuple(d.items()) for d in row_bboxes_wide}]):
            faultyAnnotation.append('wide')     # duplicate bboxes
        
        # XML | Columns
        # XML | Columns | Get separation location
        col_edges = [(float(col.find('.//xmin').text), float(col.find('.//xmax').text)) for col in cols]
        col_edges = sorted(col_edges, key= lambda x: x[0])

        col_separators = [(col_edges[i][1], col_edges[i+1][0]) for i in range(len(col_edges)-1)]
        col_separators = [(floor(min(tuple))-1, ceil(max(tuple))+1) for tuple in col_separators]

        col_bboxes_narrow = [{'ymin': table['ymin'], 'xmin': col_separator[0], 'ymax': table['ymax'], 'xmax': col_separator[1]} for col_separator in col_separators]

        # XML | Columns | Widen
        col_patterns = np.asarray([np.absolute(np.diff(img01[:, colIndex])).mean()*100 for colIndex in range(img01.shape[1])])
        col_whites = np.asarray([np.mean(img01[:, colIndex])*100 for colIndex in range(img01.shape[1])])

        col_bboxes_wide = [widenBbox(bbox=colbbox, minCoord='xmin', maxCoord='xmax', shapeIndex=1, patterns=col_patterns, whites=col_whites, thresholds=THRESHOLDS_COLUMNS) for colbbox in col_bboxes_narrow]
        if not len(col_bboxes_wide) == len([dict(t) for t in {tuple(d.items()) for d in col_bboxes_wide}]):
            faultyAnnotation.append('wide')     # duplicate bboxes


        # XML | Spanning cells
        # XML | Spanning cells | Get location
        spans = [spanElement.find('.//bndbox') for spanElement in spanners]
        spans = map(lambda el: {child.tag: float(child.text) for child in el.getchildren()}, spans)
        spans = list(map(lambda edges: {key: floor(value) if 'min' in key else ceil(value) for key, value in edges.items()}, spans))

        # XML | Spanning cells | Align to closest separators (remove overflox from labeling inaccuracies)
        spans = [findClosestSeparators(rowSeparatorBboxes=row_bboxes_narrow, colSeparatorBboxes=col_bboxes_narrow, cellBbox=span) for span in spans]        # span = spans[0]

        # XML | Convert to interior (remove borders)
        spanInteriors_narrow = []
        spanInteriors_wide = []
        for span in spans:      # span = spans[0]
            span_narrowSeparators_rows = list(filter(lambda bbox: isCellSeparatorEdge(separatorBbox=bbox, cellBbox=span, orientation='row'), row_bboxes_narrow))
            span_narrowSeparators_cols = list(filter(lambda bbox: isCellSeparatorEdge(separatorBbox=bbox, cellBbox=span, orientation='col'), col_bboxes_narrow))
            try:
                span_narrowInterior = cellToInterior(rowSeparatorBboxes=span_narrowSeparators_rows, colSeparatorBboxes=span_narrowSeparators_cols, cellBbox=span, bboxConstraint=img.shape)
                spanInteriors_narrow.append(span_narrowInterior)
            except MinMaxError as e:
                with open(pathErrors, 'a+') as f:
                    f.write(f'{labelFileEntry.name}\tnarrow span\n')
                tqdm.write(str(e))
                faultyAnnotation.append('narrow')
            del span_narrowSeparators_cols, span_narrowSeparators_rows

            span_wideSeparators_rows = list(filter(lambda bbox: isCellSeparatorEdge(separatorBbox=bbox, cellBbox=span, orientation='row'), row_bboxes_wide))
            span_wideSeparators_cols = list(filter(lambda bbox: isCellSeparatorEdge(separatorBbox=bbox, cellBbox=span, orientation='col'), col_bboxes_wide))
            try:
                span_wideInterior = cellToInterior(rowSeparatorBboxes=span_wideSeparators_rows, colSeparatorBboxes=span_wideSeparators_cols, cellBbox=span, bboxConstraint=img.shape)
                spanInteriors_wide.append(span_wideInterior)
            except MinMaxError as e:
                with open(pathErrors, 'a+') as f:
                    f.write(f'{labelFileEntry.name}\twide span\n')
                tqdm.write(str(e))
                faultyAnnotation.append('wide')
            del span_wideSeparators_cols, span_wideSeparators_rows
    else:
        row_bboxes_narrow = []
        row_bboxes_wide = []
        col_bboxes_narrow = []
        col_bboxes_wide = []
        spanInteriors_narrow = []
        spanInteriors_wide = []
        

    # XML | Write separator annotation
    # XML | Write separator annotation | Narrow
    if 'narrow' not in faultyAnnotation:
        separatorXml = etree.Element('annotation')
        xml_size = etree.SubElement(separatorXml, 'size')
        xml_width = etree.SubElement(xml_size, 'width'); xml_width.text = str(img.shape[1])
        xml_height = etree.SubElement(xml_size, 'height'); xml_height.text = str(img.shape[0])
        writeBboxToXml(bboxes=row_bboxes_narrow, xmlRoot=separatorXml, label='row separator')
        writeBboxToXml(bboxes=col_bboxes_narrow, xmlRoot=separatorXml, label='column separator')
        writeBboxToXml(bboxes=spanInteriors_narrow, xmlRoot=separatorXml, label='spanning cell interior')
        tree = etree.ElementTree(separatorXml)
        tree.write(pathLabels_separators_narrow / labelFileEntry.name, pretty_print=True, xml_declaration=False, encoding="utf-8") 

    # XML | Write separator annotation | Wide
    # # XML | Write separator annotation | Wide | Check for duplicates
    # df = pd.DataFrame.from_records(row_bboxes_wide + col_bboxes_wide + spanInteriors_wide)
    # if df.duplicated().sum():
    #     df = df.sort_values(by=df.columns.tolist())
    #     df.to_csv('temp.csv')
    #     raise Exception('duplicates spotted')

    if 'wide' not in faultyAnnotation:
        separatorXml = etree.Element('annotation')
        xml_size = etree.SubElement(separatorXml, 'size')
        xml_width = etree.SubElement(xml_size, 'width'); xml_width.text = str(img.shape[1])
        xml_height = etree.SubElement(xml_size, 'height'); xml_height.text = str(img.shape[0])
        writeBboxToXml(bboxes=row_bboxes_wide, xmlRoot=separatorXml, label='row separator')
        writeBboxToXml(bboxes=col_bboxes_wide, xmlRoot=separatorXml, label='column separator')
        writeBboxToXml(bboxes=spanInteriors_wide, xmlRoot=separatorXml, label='spanning cell interior')
        tree = etree.ElementTree(separatorXml)
        tree.write(pathLabels_separators_wide / labelFileEntry.name, pretty_print=True, xml_declaration=False, encoding="utf-8") 

# Plot
tabledetect.utils.visualise_annotation(path_images=PATH_IN / PROJECT / 'selected', path_labels=pathLabels_separators_narrow, path_output=PATH_DATA / 'images_annotated' / 'narrow', annotation_type=None, annotation_format={'labelFormat': 'voc', 'labels': ['row separator', 'column separator', 'spanning cell interior'], 'classMap': None, 'split_annotation_types': False, 'show_labels': False, 'as_area': True})
tabledetect.utils.visualise_annotation(path_images=PATH_IN / PROJECT / 'selected', path_labels=pathLabels_separators_wide, path_output=PATH_DATA / 'images_annotated' / 'wide', annotation_type=None, annotation_format={'labelFormat': 'voc', 'labels': ['row separator', 'column separator', 'spanning cell interior'], 'classMap': None, 'split_annotation_types': False, 'show_labels': False, 'as_area': True})

# Convert to YOLO
def voc_to_yolo(vocPath, outPath, classMap:str):
    filePaths = [entry.path for entry in os.scandir(vocPath)]
    outPath = Path(outPath)

    for filePath in tqdm(filePaths, desc='Converting voc to yolo'):      # filePath = filePaths[0]
        bboxes = []
        root = etree.parse(filePath).getroot()
        width = int(root.find(".//width").text); height = int(root.find(".//height").text)

        for obj in root.findall('object'):          # obj = root.findall('object')[0]
            label = obj.find('name').text
            labelIndex = classMap[label]
            bbox = {el.tag: int(el.text) for el in obj.find('bndbox').getchildren()}
        
            x_center = (bbox['xmin'] + bbox['xmax']) / 2 / width
            box_width = (bbox['xmax'] - bbox['xmin']) / width
            y_center = (bbox['ymin'] + bbox['ymax']) / 2 / height
            box_height = (bbox['ymax'] - bbox['ymin']) / height
            bboxes.append(f'{labelIndex} {x_center} {y_center} {box_width} {box_height}')

        yoloName = os.path.splitext(os.path.basename(filePath))[0] + '.txt'
        with open(outPath / yoloName, 'w') as outFile:
            outFile.write('\n'.join(bboxes))

    with open(outPath.parent / 'yolo_classes.json', 'w') as f:
        json.dump(classMap, f, indent=1)

# CLASS_MAP = {'row separator': 0, 'column separator': 1, 'spanning cell interior': 2}
# voc_to_yolo(pathLabels_separators_wide, outPath=pathLabels_yolo, classMap=CLASS_MAP)