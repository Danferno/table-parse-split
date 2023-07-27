# Imports
import sklearn.cluster as cluster
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
from pathlib import Path
import numpy as np
import pandas as pd
import os
import seaborn as sns
from math import ceil, sqrt
from random import randrange

# Parameters
PATH_ROOT = Path(r"F:\ml-parsing-project\table-parse-split\data\tableparse_round2\splits\train\predictions_lineLevel")
batch_size = 5

# Functions
def visualize_bins(df, cols_to_bin, outpath):
    vizdf = df.copy()
    vizdf['withinID'] = vizdf.groupby('batch_bucket_id').cumcount()
    vizdf[f'{cols_to_bin[0]}_jit'] = vizdf[cols_to_bin[0]] + np.random.uniform(low=0, high=0.8, size=len(vizdf))
    vizdf[f'{cols_to_bin[1]}_jit'] = vizdf[cols_to_bin[1]] + np.random.uniform(low=0, high=0.8, size=len(vizdf))

    rectangles = vizdf.groupby('batch_bucket_id').agg({cols_to_bin[0]: ['min', 'max'], cols_to_bin[1]: ['min', 'max']})
    rectangles.columns = rectangles.columns.map('_'.join)
    rectangles['x0'] = rectangles[f'{cols_to_bin[0]}_min'] - 0.1
    rectangles['y0'] = rectangles[f'{cols_to_bin[1]}_min'] - 0.1
    rectangles['x1'] = rectangles[f'{cols_to_bin[0]}_max'] + 0.9
    rectangles['y1'] = rectangles[f'{cols_to_bin[1]}_max'] + 0.9
    rectangles['width']  = rectangles['x1'] - rectangles['x0']
    rectangles['height'] = rectangles['y1'] - rectangles['y0']

    sns.scatterplot(x='count_row_jit', y='count_col_jit', data=vizdf, marker='o', s=15, hue='batch_bucket_id', palette=sns.color_palette('hls'), legend=False)
    ax = plt.gca()
    for _, rectangle in rectangles.iterrows():
        xy, width, height = (rectangle['x0'], rectangle['y0']), rectangle['width'], rectangle['height']
        _ = ax.add_patch(Rectangle(xy=xy, width=width, height=height, fill=None, alpha=0.2))
    plt.savefig(outpath, dpi=300, bbox_inches='tight', format='png')

# Load data
predictionFiles = os.scandir(PATH_ROOT)
counts = []

for entry in predictionFiles:       # entry = next(iter(predictionFiles))
    with open(entry.path) as f:
        preds = json.load(f)
    counts.append({'filename': entry.name,
                   'count_row': len(preds['row_separator_predictions']),
                   'count_col': len(preds['col_separator_predictions'])})

df = pd.DataFrame.from_records(counts).set_index('filename', drop=True).sort_values(by=['count_row', 'count_col'])
df = df.loc[(df['count_row'] > 0) & (df['count_col'] > 0)]
cols_to_bin = ['count_row', 'count_col']

# Greedy binner
def greedy_binner(df, cols_to_bin, out_list=True, path_out_plot='greedy.png'):
    def greedy_binner_innerloop(df_notok, batch_size, cols_to_bin, df_ok=pd.DataFrame()):
        # Split into buckets
        desired_bins = ceil(len(df_notok) / batch_size)

        if desired_bins > 2:
            bin_count = ceil(sqrt(desired_bins))
            _, x_edges, y_edges = np.histogram2d(x=df_notok[cols_to_bin[0]], y=df_notok[cols_to_bin[1]], bins=[bin_count, bin_count])

            # Assign obs to large buckets
            df_notok['bucket_1'] = np.digitize(df_notok[cols_to_bin[0]], x_edges) - 1
            df_notok['bucket_2'] = np.digitize(df_notok[cols_to_bin[1]], y_edges) - 1
            df_notok['bucket_id'] = df_notok.groupby(['bucket_1', 'bucket_2']).ngroup()
            df_notok = df_notok.drop(columns=['bucket_1', 'bucket_2'])

            # Split large buckets into batch_sized buckets
            df_notok['within_bucket_n'] = df_notok.groupby('bucket_id').cumcount()
            df_notok['batch_bucket_id_within'] = df_notok['within_bucket_n'] // batch_size
            df_notok['batch_bucket_id'] = df_notok.groupby(['bucket_id', 'batch_bucket_id_within']).ngroup()
            df_notok['batch_bucket_size'] = df_notok.groupby(['batch_bucket_id'])[df_notok.columns[0]].transform('size')

            # Split OK and too-small buckets
            df_ok_new = df_notok.loc[(df_notok['batch_bucket_size'] == batch_size), ['batch_bucket_id']]
            df_notok = df_notok.loc[(~df_notok.index.isin(df_ok_new.index)), [cols_to_bin[0], cols_to_bin[1]]]
        
        else:
            df_notok['batch_bucket_id'] = (df_notok.reset_index().index < 5).astype(np.int8)
            df_ok_new = df_notok[['batch_bucket_id']]
            df_notok = df_notok.loc[(~df_notok.index.isin(df_ok_new.index)), [cols_to_bin[0], cols_to_bin[1]]]

        # Merge
        df_ok = pd.merge(left=df_ok, right=df_ok_new, how='outer', left_index=True, right_index=True)
        if 'batch_bucket_id_x' in df_ok.columns:
            df_ok['batch_bucket_id'] = df_ok.groupby(['batch_bucket_id_x', 'batch_bucket_id_y'], dropna=False).ngroup()
            df_ok = df_ok.drop(columns=['batch_bucket_id_x', 'batch_bucket_id_y'])

        print(f'Attempted to make {desired_bins:>4d} bins. Left with {len(df_notok):>5d} obs to distribute.')
        return df_ok, df_notok

    if len(cols_to_bin) == 1:
        raise ValueError('Single column binner not yet implemented')
    elif len(cols_to_bin) == 2:
        # Initialize
        df_notok = df.copy()
        df_ok = pd.DataFrame()
        delta_obs = len(df_notok)

        # Bin greedily
        while len(df_notok) & (delta_obs):
            obs_to_distribute_initial = len(df_notok)
            df_ok, df_notok = greedy_binner_innerloop(df_notok=df_notok, df_ok=df_ok, batch_size=batch_size, cols_to_bin=cols_to_bin)
            delta_obs = obs_to_distribute_initial - len(df_notok)
    else:
        raise ValueError('Dims > 2 not yet implemented')
    
    if path_out_plot:
        visualize_bins(df=df_naive, cols_to_bin=cols_to_bin, outpath=path_out_plot)

    return df_ok

df_greedy = greedy_binner(df=df, cols_to_bin=cols_to_bin)

# Compare to naive binner
df_naive = df.copy()
df_naive['batch_bucket_id'] = df_naive.reset_index().index // 5
visualize_bins(df_naive, cols_to_bin=cols_to_bin, outpath='naive.png')

