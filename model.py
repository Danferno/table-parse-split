# Imports
import os, sys
from pathlib import Path
import json
import torch
from torch import nn
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, DataLoader
from torch import ByteTensor, Tensor, as_tensor
import cv2 as cv
import numpy as np

# Constants
PATH_ROOT = Path(r"F:\ml-parsing-project\table-parse-split")
COMPLEXITY = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 2

TARGET_MAX_LUMINOSITY = 240
PREDS_MAX_LUMINOSITY = 200

# Paths
pathData = PATH_ROOT / 'data' / f'fake_{COMPLEXITY}'
os.makedirs(pathData / 'val_annotated', exist_ok=True)

# Data
class TableDataset(Dataset):
    def __init__(self, dir_data,
                    image_format='.jpg',
                    transform=None, target_transform=None):
        # class A:
        #  pass
        # self = A()

        # Store in self 
        dir_data = Path(dir_data)
        self.dir_images = dir_data / 'images'
        self.dir_features = dir_data / 'features'
        self.dir_labels = dir_data / 'labels'
        self.transform = transform
        self.target_transform = target_transform
        self.image_format = image_format
        self.image_size = (600, 400)

        # Get filepaths
        self.image_fileEntries = list(os.scandir(self.dir_images))
        self.feature_fileEntries = list(os.scandir(self.dir_features))
        self.label_fileEntries = list(os.scandir(self.dir_labels))

        items_images   = set([os.path.splitext(file.name)[0] for file in self.image_fileEntries])
        items_features = set([os.path.splitext(file.name)[0] for file in self.feature_fileEntries])
        items_labels   = set([os.path.splitext(file.name)[0] for file in self.label_fileEntries])

        # Verify consistency between image/feature/label
        self.items = sorted(list(items_images.intersection(items_features).intersection(items_labels)))
        assert len(self.items) == len(items_images) == len(items_features) == len(items_labels), 'Set of images, features and labels do not match.'
    def __len__(self):
        return len(self.items)  
    def __getitem__(self, idx):     # idx = 0
        # Generate paths for a specific item
        pathImage = str(self.dir_images / f'{self.items[idx]}{self.image_format}')
        pathLabel = str(self.dir_labels / f'{self.items[idx]}.json')
        pathFeatures = str(self.dir_features / f'{self.items[idx]}.json')

        # Load sample
        sample = {}

        # Load sample | Image (not sure if this is working properly)
        sample['image'] = read_image(pathImage, mode=ImageReadMode.GRAY)
        
        # Load sample | Label
        with open(pathLabel, 'r') as f:
            labelData = json.load(f)
        sample['label'] = {}
        sample['label']['row'] = Tensor(labelData['row']).unsqueeze(-1)
    
        # Load sample | Features
        with open(pathFeatures, 'r') as f:
            featuresData = json.load(f)
        sample['features'] = {}
        sample['features']['row_absDiff'] = Tensor(featuresData['row_absDiff']).unsqueeze(-1)
        sample['features']['row_avg'] = Tensor(featuresData['row_avg']).unsqueeze(-1)

        # Optional transform
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        if self.target_transform:
            for labelType in sample['label']:
                sample['label'][labelType] = self.target_transform(sample['label'][labelType])

        # Return sample
        return sample    

dataset_train = TableDataset(dir_data=pathData / 'train')
dataloader_train = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True)

dataset_val = TableDataset(dir_data=pathData / 'val')
dataloader_val = DataLoader(dataset=dataset_val, batch_size=BATCH_SIZE, shuffle=False)

dataset_test = TableDataset(dir_data=pathData / 'test')
dataloader_test = DataLoader(dataset=dataset_val, batch_size=BATCH_SIZE, shuffle=False)

sample = next(iter(dataloader_train))
# img = sample['image'][0].squeeze().numpy()
# # cv.imshow('img', img); cv.waitKey(0)
# label = sample['label']['row'][0]
# features = sample['features']
# feature_row_absDiff = features['row_absDiff'][0]
# feature_row_avg = features['row_avg'][0]

# Model
class TabliterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_row_avg = nn.Linear(in_features=2, out_features=1)
        self.layer_logit = nn.Sigmoid()
    
    def forward(self, sample):
        # row_avg_inputs = [row.view(1,) for row in sample['features']['row_avg'].squeeze()]
        # row_avgs = [self.layer_row_avg(row_avg_input) for row_avg_input in row_avg_inputs]
        # intermediate_inputs = [Tensor(row_avgs[i]) for i in range(len(row_avgs))]
        # logits = torch.cat([self.layer_logit(intermediate_input) for intermediate_input in intermediate_inputs])

        row_avg_inputs = sample['features']['row_avg']
        row_absDiff_inputs = sample['features']['row_absDiff']
        row_inputs = torch.cat([row_avg_inputs, row_absDiff_inputs], dim=-1)
        row_avgs = self.layer_row_avg(row_inputs)
        
        intermediate_inputs = row_avgs
        logits = self.layer_logit(intermediate_inputs)
        
        return logits

model = TabliterModel()
logits = model(sample)

# Loss function
# Loss function | Calculate target ratio to avoid dominant focus
targets = [(sample['label']['row'].sum().item(), sample['label']['row'].shape[0]) for sample in iter(dataset_train)]
ones = sum([sample[0] for sample in targets])
total = sum([sample[1] for sample in targets])

shareOnes = ones / total
shareZeros = 1-shareOnes
classWeights = Tensor([1.0/shareZeros, 1.0/shareOnes])
classWeights = classWeights / classWeights.sum()

# Loss function | Define weighted loss function
class WeightedBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, weights=[]):
        super(WeightedBinaryCrossEntropyLoss, self).__init__()
        self.weights = weights
    def forward(self, input, target):
        input_clamped = torch.clamp(input, min=1e-8, max=1-1e-8)
        if self.weights is not None:
            assert len(self.weights) == 2
            loss =  self.weights[0] * ((1-target) * torch.log(1- input_clamped)) + \
                    self.weights[1] *  (target    * torch.log(input_clamped)) 
        else:
            loss = (1-target) * torch.log(1 - input_clamped) + \
                    target    * torch.log(input_clamped)

        return torch.neg(torch.mean(loss))


# Train
lr = 1e-1
epochs = 10
loss_fn = WeightedBinaryCrossEntropyLoss(weights=classWeights)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

def train_loop(dataloader, model, loss_fn, optimizer, report_frequency=4):
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    for batchNumber, batch in enumerate(dataloader):     # batch, sample = next(enumerate(dataloader))
        # Compute prediction and loss
        batch = batch
        targets_row = batch['label']['row'].to(DEVICE)
        preds_row = model(batch).to(DEVICE)
        loss = loss_fn(preds_row, targets_row)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Report 4 times
        report_batch_size = (size / batch_size) // report_frequency
        if (batchNumber+1) % report_batch_size == 0:
            loss, current = loss.item(), (batchNumber+1) * batch_size
            print(f'Loss: {loss:>7f} [{current:>5d}/{size:>5d}]')

def val_loop(dataloader, model, loss_fn):
    sampleCount = len(dataloader.dataset)
    batchCount = len(dataloader)
    val_loss, correct = 0,0
    with torch.no_grad():
        for batch in dataloader:     # batch = next(iter(dataloader))
            # Compute prediction and loss
            targets_row = batch['label']['row']
            preds_row = model(batch)

            val_loss += loss_fn(preds_row, targets_row).item()
            correct += ((preds_row >= 0.5) == targets_row).sum().item()

    rows_per_image = targets_row.shape[1]
    val_loss = val_loss / batchCount
    correct = correct / (sampleCount * rows_per_image)

    print(f'''Validation
        Accuracy: {(100*correct):>0.1f}%
        Avg val loss: {val_loss:>8f}''')

for t in range(epochs):
    print(f"Epoch {t+1} -------------------------------")
    train_loop(dataloader=dataloader_train, model=model, loss_fn=loss_fn, optimizer=optimizer, report_frequency=1)
    val_loop(dataloader=dataloader_val, model=model, loss_fn=loss_fn)


# Predict
def eval_loop(dataloader, model, loss_fn, outPath=None):
    batchCount = len(dataloader)
    eval_loss, correct, maxCorrect = 0,0,0
    with torch.no_grad():
        for batchNumber, batch in enumerate(dataloader):
            # Predict
            targets_row = batch['label']['row']
            preds_row = model(batch)

            # Eval
            eval_loss += loss_fn(preds_row, targets_row).item()
            correct += ((preds_row >= 0.5) == targets_row).sum().item()
            maxCorrect += preds_row.numel()

            # Visualise
            if outPath:
                imagePixels = torch.flatten(batch['image'], end_dim=2)
                row_target_pixels = torch.broadcast_to(input=torch.flatten(targets_row, end_dim=1), size=(targets_row.numel(), 40)) * TARGET_MAX_LUMINOSITY
                row_prediction_pixels = (torch.broadcast_to(input=torch.flatten(preds_row, end_dim=1), size=(preds_row.numel(), 40)) >= 0.5)* PREDS_MAX_LUMINOSITY

                rowPixels = torch.cat([row_target_pixels, row_prediction_pixels], dim=1)

                img = torch.cat([imagePixels, rowPixels], dim=1).numpy().astype(np.uint8)
                cv.imwrite(str(outPath / f'img_{batchNumber}.jpg'), img)

    eval_loss = eval_loss / batchCount
    correct = correct / maxCorrect

    print(f'''Evaluation (normally on val)
        Accuracy: {(100*correct):>0.1f}%
        Avg val loss: {eval_loss:>8f}''')
    
    if outPath:
        return outPath
    
eval_loop(dataloader=dataloader_val, model=model, loss_fn=loss_fn, outPath=pathData / 'val_annotated')

# Visualise
def visualize(batchedResults, outPath):
    for batchedResult in batchedResults:
        batchSize = batchedResult['images'].shape[0]
        for sampleNumber in range(batchSize):
            image = batchedResult['images'][sampleNumber].squeeze()
            rowPrediction = (batchedResult['predictions'][sampleNumber] > 0.5)
            rowTarget = batchedResult['targets'][sampleNumber]

            prediction = torch.broadcast_to(rowPrediction, size=(rowPrediction.shape[0], 40))
            target = torch.broadcast_to(rowTarget, size=(rowTarget.shape[0], 40))

            img = torch.cat([image, prediction, target], dim=1).numpy().astype(np.uint8)
            cv.imwrite(str(outPath / f'img_{i}.jpg'), img)

print('End')