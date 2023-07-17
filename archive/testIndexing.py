import torch
from einops import rearrange

batch_size = 2
row_count = 6
feat_count = 2
length = 3
separator_count = 2

# src
rows = torch.arange(row_count)
feats = torch.arange(feat_count)
batches = torch.arange(batch_size)

batches, rows, cols,  = torch.meshgrid(batches, rows, feats)

src = batches * 100 + cols * 10 + rows

# indices
separatorstarts = torch.tensor([[0, 1], [0, 1]]).unsqueeze(-1)
indices = separatorstarts + torch.arange(length)
indices = rearrange(indices, pattern='batch sep incr -> batch (sep incr)')

i = torch.arange(batch_size).reshape(batch_size, 1, 1)
j = indices.unsqueeze(-1)
k = torch.arange(feat_count)

values = src[i, j, k]
reshaped = rearrange(values, pattern='batch (sep incr) feats -> batch sep incr feats', incr=length)

labels = torch.randint(0, row_count, size=[batch_size, 2])