import os, shutil
import click
import pandas as pd

# Constants
COLOR_CORRECT = (0, 255, 0, int(0.25*255))
COLOR_WRONG = (255, 0, 0, int(0.25*255))

# Functions
def replaceDir(path):
    shutil.rmtree(path)
    os.makedirs(path)

def makeDirs(path, replaceDirs='warn'):
    try:
        os.makedirs(path)
    except FileExistsError:
        if replaceDirs == 'warn':
            if click.confirm(f'This will remove folder {path}. Are you sure you are okay with this?'):
                replaceDir(path)
            else:
                raise InterruptedError(f'User was not okay with removing {path}')
        elif replaceDirs:
            replaceDir(path)
        else:
            raise FileExistsError
        
def pageWords_to_tableWords(path_words, tableName, metaInfo, tableSplitString='_t'):
    # Technicalities
    pd.set_option('mode.copy_on_write', True)
    coordinateColumns = ['left', 'right', 'top', 'bottom']
    tableCoords = metaInfo['table_coords']
    
    # Load words
    pageName = tableName.split(tableSplitString)[0]
    wordsDf_page = pd.read_parquet(path_words / f'{pageName}.pq').drop_duplicates()

    # Rescale OCR coordinates to PDF coordinates
    wordsDf_page[coordinateColumns] = wordsDf_page[coordinateColumns] * (metaInfo['dpi_pdf'] / metaInfo['dpi_words'])

    # Reduce to table bbox
    wordsDf_table = wordsDf_page.loc[(wordsDf_page['top'] >= tableCoords['y0']) & (wordsDf_page['left'] >= tableCoords['x0']) & (wordsDf_page['bottom'] <= tableCoords['y1']) & (wordsDf_page['right'] <= tableCoords['x1'])]

    # Rescale PDF coordinates to padded table/model coordinates
    wordsDf_table[['left', 'right']] = wordsDf_table[['left', 'right']] - tableCoords['x0']
    wordsDf_table[['top', 'bottom']] = wordsDf_table[['top', 'bottom']] - tableCoords['y0']
    wordsDf_table[coordinateColumns] = wordsDf_table[coordinateColumns] * (metaInfo['dpi_model'] / metaInfo['dpi_pdf']) + metaInfo['padding_model']
    wordsDf_table[coordinateColumns] = wordsDf_table[coordinateColumns].round(0).astype(int)

    # Return df
    return wordsDf_table
