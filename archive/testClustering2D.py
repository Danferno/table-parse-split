# Imports
import sklearn.cluster as cluster
import json
import matplotlib.pyplot as plt
import time
from pathlib import Path
import numpy as np
import pandas as pd
import os
import seaborn as sns
from math import ceil, sqrt

# Parameters
PATH_ROOT = Path(r"F:\ml-parsing-project\table-parse-split\data\tableparse_round2\splits\train\predictions_lineLevel")
batch_size = 5

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

# Greedy binner
def greedy_batcher(df_notok, batch_size, df_ok=pd.DataFrame(), draw_images=False):
    # Split into buckets
    desired_bins = ceil(len(df_notok) / batch_size)

    if desired_bins > 2:
        bin_count = ceil(sqrt(desired_bins))
        hist, x_edges, y_edges = np.histogram2d(x=df_notok['count_row'], y=df_notok['count_col'], bins=[bin_count, bin_count])

        if draw_images:
            plt.figure(figsize=(6,6))
            plt.pcolormesh(*np.meshgrid(x_edges, y_edges), hist.T, cmap='Blues')
            plt.colorbar(label="Frequency")
            plt.xlabel('Row'); plt.ylabel('Col')
            plt.grid(True)
            plt.show()

        # Assign obs to large buckets
        df_notok['bucket_x'] = np.digitize(df_notok['count_row'], x_edges) - 1
        df_notok['bucket_y'] = np.digitize(df_notok['count_col'], y_edges) - 1
        df_notok['bucket_id'] = df_notok.groupby(['bucket_x', 'bucket_y']).ngroup()
        df_notok = df_notok.drop(columns=['bucket_x', 'bucket_y'])

        # Split large buckets into batch_sized buckets
        df_notok['within_bucket_n'] = df_notok.groupby('bucket_id').cumcount()
        df_notok['batch_bucket_id_within'] = df_notok['within_bucket_n'] // batch_size
        df_notok['batch_bucket_id'] = df_notok.groupby(['bucket_id', 'batch_bucket_id_within']).ngroup()
        df_notok['batch_bucket_size'] = df_notok.groupby(['batch_bucket_id'])[df_notok.columns[0]].transform('size')

        # Split OK and too-small buckets
        df_ok_new = df_notok.loc[(df_notok['batch_bucket_size'] == batch_size), ['batch_bucket_id']]
        df_notok = df_notok.loc[(~df_notok.index.isin(df_ok_new.index)), ['count_row', 'count_col']]
    
    else:
        df_notok['within_bucket_n'] = df_notok.cumcount()
        df_notok['batch_bucket_id_within'] = df_notok['within_bucket_n'] // batch_size
        df_notok['batch_bucket_id'] = df_notok.groupby(['bucket_id', 'batch_bucket_id_within']).ngroup()

    # Merge
    df_ok = pd.merge(left=df_ok, right=df_ok_new, how='outer', left_index=True, right_index=True)
    if 'batch_bucket_id_x' in df_ok.columns:
        df_ok['batch_bucket_id'] = df_ok.groupby(['batch_bucket_id_x', 'batch_bucket_id_y'], dropna=False).ngroup()
        df_ok = df_ok.drop(columns=['batch_bucket_id_x', 'batch_bucket_id_y'])

    print(f'Attempted to make {desired_bins:>4d} bins. Left with {len(df_notok):>5d} obs to distribute.')
    return df_ok, df_notok

df_notok = df.copy()
df_ok = pd.DataFrame()
df_ok, df_notok = greedy_batcher(df_notok=df_notok, df_ok=df_ok, batch_size=batch_size)
df_ok, df_notok = greedy_batcher(df_notok=df_notok, df_ok=df_ok, batch_size=batch_size)
df_ok, df_notok = greedy_batcher(df_notok=df_notok, df_ok=df_ok, batch_size=batch_size)
df_ok, df_notok = greedy_batcher(df_notok=df_notok, df_ok=df_ok, batch_size=batch_size)


# Greedy binner | Round 1 | Get large buckets
desired_bins = ceil(len(df) / batch_size)
bin_count = ceil(sqrt(desired_bins))
hist, x_edges, y_edges = np.histogram2d(x=df['count_row'], y=df['count_col'], bins=[bin_count, bin_count])

# # Greedy binner | Round 1 | Plot 
# plt.figure(figsize=(6,6))
# plt.pcolormesh(*np.meshgrid(x_edges, y_edges), hist.T, cmap='Blues')
# plt.colorbar(label="Frequency")
# plt.xlabel('Row'); plt.ylabel('Col')
# plt.grid(True)
# plt.show()

# Greedy binner | Round 1 | Assign obs to large buckets
df['bucket_x'] = np.digitize(df['count_row'], x_edges) - 1
df['bucket_y'] = np.digitize(df['count_col'], y_edges) - 1
df['bucket_id'] = df.groupby(['bucket_x', 'bucket_y']).ngroup()
df = df.drop(columns=['bucket_x', 'bucket_y'])

# Greedy binner | Round 1 | Split large buckets into batch_sized buckets
df['within_bucket_n'] = df.groupby('bucket_id').cumcount()
df['batch_bucket_id_within'] = df['within_bucket_n'] // batch_size
df['batch_bucket_id'] = df.groupby(['bucket_id', 'batch_bucket_id_within']).ngroup()
df['batch_bucket_size'] = df.groupby(['batch_bucket_id'])[df.columns[0]].transform('size')

# Greedy binner | Round 1 | Split OK and too-small buckets
bucket_r1 = df.loc[(df['batch_bucket_size'] == batch_size), 'batch_bucket_id']
df = df.loc[(~df.index.isin(bucket_r1.index)), ['count_row', 'count_col']]
