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
COMPLEXITY = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        sample['label']['row'] = Tensor(labelData['row'])
    
        # Load sample | Features
        with open(pathFeatures, 'r') as f:
            featuresData = json.load(f)
        sample['features'] = {}
        sample['features']['row_absDiff'] = Tensor(featuresData['row_absDiff'])
        sample['features']['row_avg'] = Tensor(featuresData['row_avg'])

        # Optional transform
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        if self.target_transform:
            for labelType in sample['label']:
                sample['label'][labelType] = self.target_transform(sample['label'][labelType])

        # Return sample
        return sample    

dataset_train = TableDataset(dir_data=pathData / 'train')
dataloader_train = DataLoader(dataset=dataset_train, batch_size=1, shuffle=True)

dataset_val = TableDataset(dir_data=pathData / 'val')
dataloader_val = DataLoader(dataset=dataset_val, batch_size=1, shuffle=True)

dataset_test = TableDataset(dir_data=pathData / 'test')
dataloader_test = DataLoader(dataset=dataset_val, batch_size=1, shuffle=True)

sample = next(iter(dataloader_train))
img = sample['image'][0].squeeze().numpy()
# cv.imshow('img', img); cv.waitKey(0)
label = sample['label']['row'][0]
features = sample['features']
feature_row_absDiff = features['row_absDiff'][0]
feature_row_avg = features['row_avg'][0]

# Model
class TabliterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_row_avg = nn.Linear(in_features=1, out_features=1)
        self.layer_logit = nn.Sigmoid()
    
    def forward(self, sample):
        row_avg_inputs = [row.view(1,) for row in sample['features']['row_avg'].squeeze()]
        row_avgs = [self.layer_row_avg(row_avg_input) for row_avg_input in row_avg_inputs]

        intermediate_inputs = [Tensor(row_avgs[i]) for i in range(len(row_avgs))]
        logits = torch.cat([self.layer_logit(intermediate_input) for intermediate_input in intermediate_inputs])
        
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
batch_size = 1
epochs = 20
loss_fn = WeightedBinaryCrossEntropyLoss(weights=classWeights)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    for batch, sample in enumerate(dataloader):     # batch, sample = next(enumerate(dataloader))
        # Compute prediction and loss
        sample = sample
        targets_row = sample['label']['row'].to(DEVICE)
        preds_row = model(sample).to(DEVICE)
        preds_row = preds_row.view(batch_size, preds_row.size(0) // batch_size)
        loss = loss_fn(preds_row, targets_row)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Report 4 times
        quarter_size = (size / batch_size) // 4
        if batch % quarter_size == 0:
            loss, current = loss.item(), (batch + 1) * batch_size
            print(f'Loss: {loss:>7f} [{current:>5d}/{size:>5d}]')

def val_loop(dataloader, model, loss_fn):
    sampleCount = len(dataloader.dataset)
    batchCount = len(dataloader)
    val_loss, correct = 0,0
    with torch.no_grad():
        for sample in dataloader:     # sample = next(iter(dataloader))
            # Compute prediction and loss
            targets_row = sample['label']['row']
            preds_row = model(sample)
            preds_row = preds_row.view(batch_size, preds_row.size(0) // batch_size)

            val_loss += loss_fn(preds_row, targets_row).item()
            correct += ((preds_row >= 0.5) == targets_row).sum().item()

    rows_per_image = targets_row.shape[-1]
    val_loss = val_loss / batchCount
    correct = correct / (sampleCount * rows_per_image)

    print(f'''Validation
        Accuracy: {(100*correct):>0.1f}%
        Avg val loss: {val_loss:>8f}''')

for t in range(epochs):
    print(f"Epoch {t+1} -------------------------------")
    train_loop(dataloader=dataloader_train, model=model, loss_fn=loss_fn, optimizer=optimizer)
    val_loop(dataloader=dataloader_val, model=model, loss_fn=loss_fn)


# Predict
def eval_loop(dataloader, model, loss_fn):
    sampleCount = len(dataloader.dataset)
    batchCount = len(dataloader)
    eval_loss, correct = 0,0
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in dataloader:
            sample = batch
            targets_row = sample['label']['row']
            preds_row = model(sample)
            preds_row = preds_row.view(batch_size, preds_row.size(0) // batch_size)

            eval_loss += loss_fn(preds_row, targets_row).item()
            correct += ((preds_row >= 0.5) == targets_row).sum().item()
            predictions.append(preds_row)
            targets.append(targets_row)

    rows_per_image = targets_row.shape[-1]
    eval_loss = eval_loss / batchCount
    correct = correct / (sampleCount * rows_per_image)

    print(f'''Evaluation (normally on val)
        Accuracy: {(100*correct):>0.1f}%
        Avg val loss: {eval_loss:>8f}''')
    return predictions, targets
    
predictions, targets = eval_loop(dataloader=dataloader_val, model=model, loss_fn=loss_fn)

# Visualise
def visualize(prediction, target, i, outPath):
    rowPrediction = prediction.squeeze() > 0.5
    rowTarget = target.squeeze()

    prediction = rowPrediction.unsqueeze(1).repeat(1, 300) * 255
    target = rowTarget.unsqueeze(1).repeat(1, 50) * 128

    img = torch.cat([prediction, target], dim=1).numpy().astype(np.uint8)
    # cv.imshow("img", img); cv.waitKey(0)
    cv.imwrite(str(outPath / f'img_{i}.jpg'), img)

for i in range(len(predictions)):
    visualize(prediction=predictions[i], target=targets[i], i=i, outPath=pathData / 'val_annotated')

print('End')