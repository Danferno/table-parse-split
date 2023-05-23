# Imports
import shutil
import os, sys
from pathlib import Path
import json
import torch
from torch import nn
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from prettytable import PrettyTable
from torchviz import make_dot
import cv2 as cv
import numpy as np
import string
from collections import namedtuple, OrderedDict

# Constants
PATH_ROOT = Path(r"F:\ml-parsing-project\table-parse-split")
COMPLEXITY = 4
BATCH_SIZE = 2
LOSS_TYPES = ['row', 'col']
TARGET_MAX_LUMINOSITY = 60
PREDS_MAX_LUMINOSITY = TARGET_MAX_LUMINOSITY * 2
EMPTY_LUMINOSITY = 0

# Derived constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LOSS_TYPES_COUNT = len(LOSS_TYPES)

# Paths
def replaceDirs(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        shutil.rmtree(path)
        os.makedirs(path)
pathData = PATH_ROOT / 'data' / f'fake_{COMPLEXITY}'
pathLogs = PATH_ROOT / 'torchlogs'
replaceDirs(pathData / 'val_annotated')

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
        sample['label']['col'] = Tensor(labelData['col']).unsqueeze(-1)
    
        # Load sample | Features
        with open(pathFeatures, 'r') as f:
            featuresData = json.load(f)
        sample['features'] = {}
        sample['features']['row_avg'] = Tensor(featuresData['row_avg']).unsqueeze(-1)
        sample['features']['col_avg'] = Tensor(featuresData['col_avg']).unsqueeze(-1)
        sample['features']['row_absDiff'] = Tensor(featuresData['row_absDiff']).unsqueeze(-1)
        sample['features']['col_absDiff'] = Tensor(featuresData['col_absDiff']).unsqueeze(-1)

        sample['features']['row_spell_mean'] = Tensor(featuresData['row_spell_mean']).unsqueeze(-1)
        sample['features']['row_spell_sd'] = Tensor(featuresData['row_spell_sd']).unsqueeze(-1)
        sample['features']['col_spell_mean'] = Tensor(featuresData['col_spell_mean']).unsqueeze(-1)
        sample['features']['col_spell_sd'] = Tensor(featuresData['col_spell_sd']).unsqueeze(-1)

        sample['features']['row_firstletter_capitalOrNumeric'] = Tensor(featuresData['row_firstletter_capitalOrNumeric']).unsqueeze(-1)

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

# Model
Output = namedtuple('output', LOSS_TYPES)

class TabliterModel(nn.Module):
    def __init__(self, hidden_sizes=[15,10], layer_depth=3):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.layer_depth = layer_depth

        self.layer_linear_row = self.addLayer(layerType=nn.Linear, in_features=5, out_features=1)
        self.layer_linear_col = self.addLayer(layerType=nn.Linear, in_features=4, out_features=1)

        self.layer_logit = nn.Sigmoid()
    
    def addLayer(self, layerType:nn.Module, in_features:int, out_features:int, activation=nn.PReLU):
        sequence = nn.Sequential(OrderedDict([
            (f'lin_from{in_features}_to{self.hidden_sizes[0]}', layerType(in_features=in_features, out_features=self.hidden_sizes[0])),
            (f'relu_1', activation()),
            (f'lin_from{self.hidden_sizes[0]}_to{self.hidden_sizes[1]}', layerType(in_features=self.hidden_sizes[0], out_features=self.hidden_sizes[1])),
            (f'relu_2', activation()),
            (f'lin_from{self.hidden_sizes[1]}_to{out_features}', layerType(in_features=self.hidden_sizes[1], out_features=out_features))
        ]))
        return sequence
    
    def forward(self, sample):
        # Row
        # Row | Inputs
        row_avg_inputs = sample['features']['row_avg']
        row_absDiff_inputs = sample['features']['row_absDiff']
        row_spell_mean_inputs = sample['features']['row_spell_mean']
        row_spell_sd_inputs = sample['features']['row_spell_sd']
        row_firstletter_capitalOrNumeric_inputs = sample['features']['row_firstletter_capitalOrNumeric']
        row_inputs = torch.cat([row_avg_inputs, row_absDiff_inputs, row_spell_mean_inputs, row_spell_sd_inputs, row_firstletter_capitalOrNumeric_inputs], dim=-1)

        # Row | Layers
        row_intermediate_inputs = self.layer_linear_row(row_inputs)
        row_probs = self.layer_logit(row_intermediate_inputs)

        # Col
        col_avg_inputs = sample['features']['col_avg']
        col_absDiff_inputs = sample['features']['col_absDiff']
        col_spell_mean_inputs = sample['features']['col_spell_mean']
        col_spell_sd_inputs = sample['features']['col_spell_sd']
        col_inputs = torch.cat([col_avg_inputs, col_absDiff_inputs, col_spell_mean_inputs, col_spell_sd_inputs], dim=-1)

        # Col | Layers
        col_intermediate_inputs = self.layer_linear_col(col_inputs)
        col_probs = self.layer_logit(col_intermediate_inputs)

        # Output
        return Output(row=row_probs, col=col_probs)

model = TabliterModel()
#model = torch.compile(model=model, mode='reduce-overhead')
sample = next(iter(dataloader_train))
logits = model(sample)

# Loss function
# Loss function | Calculate target ratio to avoid dominant focus
def calculateWeights(targets):    
    ones = sum([sample[0] for sample in targets])
    total = sum([sample[1] for sample in targets])

    shareOnes = ones / total
    shareZeros = 1-shareOnes
    classWeights = Tensor([1.0/shareZeros, 1.0/shareOnes])
    classWeights = classWeights / classWeights.sum()
    return classWeights

targets_row = [(sample['label']['row'].sum().item(), sample['label']['row'].shape[0]) for sample in iter(dataset_train)]
targets_col = [(sample['label']['col'].sum().item(), sample['label']['col'].shape[0]) for sample in iter(dataset_train)]
classWeights = {'row': calculateWeights(targets=targets_row),
                'col': calculateWeights(targets=targets_col)}

# Loss function | Define weighted loss function
class WeightedBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, weights=[]):
        super(WeightedBinaryCrossEntropyLoss, self).__init__()
        self.weights = weights
    def forward(self, input, target):
        input_clamped = torch.clamp(input, min=1e-7, max=1-1e-7)
        if self.weights is not None:
            assert len(self.weights) == 2
            loss =  self.weights[0] * ((1-target) * torch.log(1- input_clamped)) + \
                    self.weights[1] *  (target    * torch.log(input_clamped)) 
        else:
            loss = (1-target) * torch.log(1 - input_clamped) + \
                    target    * torch.log(input_clamped)

        return torch.neg(torch.mean(loss))

def calculateLoss(batch, preds, lossFunctions:dict, calculateCorrect=False):   
    loss = torch.empty(size=(LOSS_TYPES_COUNT,1), device=DEVICE, dtype=torch.float32)
    correct, maxCorrect = torch.empty(size=(LOSS_TYPES_COUNT,1), device=DEVICE, dtype=torch.int64), torch.empty(size=(LOSS_TYPES_COUNT,1), device=DEVICE, dtype=torch.int64)

    for idx, lossType in enumerate(LOSS_TYPES):
        target = batch['label'][lossType].to(DEVICE)
        pred = preds[idx].to(DEVICE)
        loss_fn = lossFunctions[lossType]

        loss[idx] = loss_fn(pred, target)

        if calculateCorrect:
            correct[idx] = ((pred >= 0.5) == target).sum().item()
            maxCorrect[idx] = pred.numel()
    
    if calculateCorrect:
        return loss, correct, maxCorrect
    else:
        return loss.sum()


# Train
lr = 9e-2
epochs = 30
lossFunctions = {'row': WeightedBinaryCrossEntropyLoss(weights=classWeights['row']),
                 'col': WeightedBinaryCrossEntropyLoss(weights=classWeights['col'])} 
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

def train_loop(dataloader, model, lossFunctions, optimizer, report_frequency=4):
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    for batchNumber, batch in enumerate(dataloader):     # batch, sample = next(enumerate(dataloader))
        # Compute prediction and loss
        preds = model(batch)
        loss = calculateLoss(batch, preds, lossFunctions)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Report 4 times
        report_batch_size = (size / batch_size) // report_frequency
        if (batchNumber+1) % report_batch_size == 0:
            loss, current = loss.item(), (batchNumber+1) * batch_size
            print(f'Loss: {loss:>7f} [{current:>5d}/{size:>5d}]')

def val_loop(dataloader, model, lossFunctions):
    batchCount = len(dataloader)
    val_loss, correct, maxCorrect = torch.zeros(size=(LOSS_TYPES_COUNT,1), device=DEVICE), torch.zeros(size=(LOSS_TYPES_COUNT,1), device=DEVICE, dtype=torch.int64), torch.zeros(size=(LOSS_TYPES_COUNT,1), device=DEVICE, dtype=torch.int64)
    with torch.no_grad():
        for batch in dataloader:     # batch = next(iter(dataloader))
            # Compute prediction and loss
            preds = model(batch)
            val_loss_batch, correct_batch, maxCorrect_batch = calculateLoss(batch, preds, lossFunctions, calculateCorrect=True)
            val_loss += val_loss_batch
            correct  += correct_batch
            maxCorrect  += maxCorrect_batch

    val_loss = val_loss / batchCount
    shareCorrect = correct / maxCorrect

    print(f'''Validation
        Accuracy: {(100*shareCorrect[0].item()):>0.1f}% (row) | {(100*shareCorrect[1].item()):>0.1f}% (col)
        Avg val loss: {val_loss.sum().item():>6f} (total) | {val_loss[0].item():>6f} (row) | {val_loss[1].item():>6f} (col)''')

for t in range(epochs):
    print(f"Epoch {t+1} -------------------------------")
    train_loop(dataloader=dataloader_train, model=model, lossFunctions=lossFunctions, optimizer=optimizer, report_frequency=1)
    val_loop(dataloader=dataloader_val, model=model, lossFunctions=lossFunctions)


# Predict
def eval_loop(dataloader, model, lossFunctions, outPath=None):
    batchCount = len(dataloader)
    eval_loss, correct, maxCorrect = torch.zeros(size=(LOSS_TYPES_COUNT,1), device=DEVICE), torch.zeros(size=(LOSS_TYPES_COUNT,1), device=DEVICE, dtype=torch.int64), torch.zeros(size=(LOSS_TYPES_COUNT,1), device=DEVICE, dtype=torch.int64)
    with torch.no_grad():
        for batchNumber, batch in enumerate(dataloader):
            # Compute prediction and loss
            preds = model(batch)
            eval_loss_batch, correct_batch, maxCorrect_batch = calculateLoss(batch, preds, lossFunctions, calculateCorrect=True)
            eval_loss += eval_loss_batch
            correct  += correct_batch
            maxCorrect  += maxCorrect_batch

            # Visualise
            if outPath:
                batch_size = dataloader.batch_size
                for sampleNumber in range(batch_size):      # sampleNumber = 0
                    # Get sample info
                    image = batch['image'][sampleNumber]
                    row_targets = batch['label']['row'][sampleNumber]
                    row_predictions = preds.row[sampleNumber]
                    col_targets = batch['label']['col'][sampleNumber]
                    col_predictions = preds.col[sampleNumber]
                    outName = 'img_' + str(batchNumber) + string.ascii_lowercase[sampleNumber]  

                    imagePixels = torch.flatten(image, end_dim=1)                   
                    row_target_pixels = torch.broadcast_to(input=row_targets, size=(row_targets.numel(), 40)) * TARGET_MAX_LUMINOSITY
                    row_prediction_pixels = (torch.broadcast_to(input=row_predictions, size=(row_predictions.numel(), 40)) >= 0.5)* PREDS_MAX_LUMINOSITY
                    rowPixels = torch.cat([row_target_pixels, row_prediction_pixels], dim=1)
                    img = torch.cat([imagePixels, rowPixels], dim=1)

                    col_target_pixels = torch.broadcast_to(input=col_targets, size=(col_targets.numel(), 40)).T * TARGET_MAX_LUMINOSITY
                    col_prediction_pixels = (torch.broadcast_to(input=col_predictions, size=(col_predictions.numel(), 40)) >= 0.5).T * PREDS_MAX_LUMINOSITY
                    colPixels = torch.cat([col_target_pixels, col_prediction_pixels], dim=0)
                    
                    emptyBlock = torch.full(size=(colPixels.shape[0], rowPixels.shape[1]), fill_value=EMPTY_LUMINOSITY)
                    colPixels = torch.cat([colPixels, emptyBlock], axis=1)

                    img = torch.cat([img, colPixels], dim=0).numpy().astype(np.uint8)
                    cv.imwrite(str(outPath / f'{outName}.jpg'), img)

    eval_loss = eval_loss / batchCount
    shareCorrect = correct / maxCorrect

    print(f'''Validation
        Accuracy: {(100*shareCorrect[0].item()):>0.1f}% (row) | {(100*shareCorrect[1].item()):>0.1f}% (col)
        Avg val loss: {eval_loss.sum().item():>6f} (total) | {eval_loss[0].item():>6f} (row) | {eval_loss[1].item():>6f} (col)''')
    
    if outPath:
        return outPath

# Visualize results
eval_loop(dataloader=dataloader_val, model=model, lossFunctions=lossFunctions, outPath=pathData / 'val_annotated')

# Visualize model
# Model | Graph
y = model(sample)
make_dot(y, params=dict(model.named_parameters()), show_attrs=True, show_saved=True).render(pathLogs / 'graph', format='png')

# Count parameters
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
count_parameters(model=model)

# Show weights
# for name, param in model.named_parameters():
#     print(name, param.data, param.requires_grad)
...