# Imports
import os
from pathlib import Path
from lxml import etree
from math import floor, ceil
from tqdm import tqdm
import tabledetect

# Constants
PATH_ROOT = Path(r"F:\ml-parsing-project\table-parse-split")
PATH_DATA = PATH_ROOT / 'data'

PATH_IN = Path(r"F:\ml-parsing-project\data")
PROJECT = "parse_activelearning1_jpg"

# Path things
pathLabels = PATH_IN / PROJECT / 'labels'
pathLabels_sepatators = PATH_DATA / 'labels' / 'narrow'
os.makedirs(pathLabels_sepatators, exist_ok=True)

# Convert row/column annotations to separator annotations
labelFiles = list(os.scandir(pathLabels))
for labelFileEntry in tqdm(labelFiles):       # labelFileEntry = labelFiles[0]
    # XML | Parse annotation
    root = etree.parse(labelFileEntry.path)
    tableEl = root.find('object[name="table"]')
    rows = root.findall('object[name="table row"]')
    columns = root.findall('object[name="table column"]')

    # XML | Table
    table = {el.tag:float(el.text) for el in tableEl.find('bndbox').getchildren()}
    for key, val in table.items():
        table[key] = floor(val) if 'min' in key else ceil(val)
    
    # XML | Rows
    # XML | Rows | Get separation location
    row_edges = [(float(row.find('.//ymin').text), float(row.find('.//ymax').text)) for row in rows]
    row_edges = sorted(row_edges, key= lambda x: x[0])

    row_separators = [(row_edges[i][1], row_edges[i+1][0]) for i in range(len(row_edges)-1)]
    row_separators = [(floor(min(tuple)), ceil(max(tuple))) for tuple in row_separators]

    row_bbox_narrow = [{'xmin': table['xmin'], 'ymin': row_separator[0], 'xmax': table['xmax'], 'ymax': row_separator[1]} for row_separator in row_separators]


    # XML | Write separator annotation
    separatorXml = etree.Element('annotation')
    for row_bbox in row_bbox_narrow:        # object = extractedTable['objects'][0]
        xml_obj = etree.SubElement(separatorXml, 'object')
        xml_bbox = etree.SubElement(xml_obj, 'bndbox')
        for edge in ['xmin', 'ymin', 'xmax', 'ymax']:
            _ = etree.SubElement(xml_bbox, edge)
            _.text = str(row_bbox[edge])
        label = etree.SubElement(xml_obj, 'name'); label.text = 'row separator'
    tree = etree.ElementTree(separatorXml)
    tree.write(pathLabels_sepatators / labelFileEntry.name, pretty_print=True, xml_declaration=False, encoding="utf-8") 

