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
from torchvision.transforms import Normalize
from prettytable import PrettyTable
from torchviz import make_dot
import cv2 as cv
import numpy as np
import string
from collections import namedtuple, OrderedDict

# Constants
PATH_ROOT = Path(r"F:\ml-parsing-project\table-parse-split")
SUFFIX = 'wide'
DATA_TYPE = 'real'
BATCH_SIZE = 1

LOSS_TYPES = ['row', 'col']
FEATURE_TYPES = LOSS_TYPES + ['image']

TARGET_MAX_LUMINOSITY = 60
PREDS_MAX_LUMINOSITY = TARGET_MAX_LUMINOSITY * 2
EMPTY_LUMINOSITY = 0

COMMON_VARIABLES = ['{}_avg', '{}_absDiff', '{}_spell_mean', '{}_spell_sd', '{}_wordsCrossed_count', '{}_wordsCrossed_relToMax']

# Model parameters
EPOCHS = 2
MAX_LR = 0.2
HIDDEN_SIZES = [8, 3]
CONV_LETTER_KERNEL = [4, 4]
CONV_LETTER_CHANNELS = 2
CONV_SEQUENCE_KERNEL = [60, 30]
CONV_SEQUENCE_CHANNELS = 2

CONV_FINAL_CHANNELS = CONV_LETTER_CHANNELS
CONV_PREDS_CHANNELS = 2

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
pathData = PATH_ROOT / 'data' / f'{DATA_TYPE}_{SUFFIX}'
pathLogs = PATH_ROOT / 'torchlogs'
replaceDirs(pathData / 'val_annotated')

# Data
Sample = namedtuple('sample', ['features', 'targets', 'meta'])
Meta = namedtuple('meta', ['path_image'])
Targets = namedtuple('target', LOSS_TYPES)
Features = namedtuple('features', FEATURE_TYPES)
def move_values_to_gpu(d):
    for _, value in d.items():
        if isinstance(value, dict):
            move_values_to_gpu(value)
        else:
            value = value.to(DEVICE)

class TableDataset(Dataset):
    def __init__(self, dir_data,
                    image_format='.jpg',
                    transform_image=None, transform_target=None):
        # Store in self 
        dir_data = Path(dir_data)
        self.dir_images = dir_data / 'images'
        self.dir_features = dir_data / 'features'
        self.dir_labels = dir_data / 'labels'
        self.transform_image = transform_image
        self.transform_target = transform_target
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
        # Load sample | Meta
        meta = Meta(path_image=pathImage)

        # Load sample | Label
        with open(pathLabel, 'r') as f:
            labelData = json.load(f)
        targets = Targets(row=Tensor(labelData['row']).unsqueeze(-1).to(DEVICE),
                                         col=Tensor(labelData['col']).unsqueeze(-1).to(DEVICE))
    
        # Load sample | Features
        with open(pathFeatures, 'r') as f:
            featuresData = json.load(f)

        # Load sample | Features | Rows
        features_row = torch.cat(tensors=[Tensor(featuresData[common_variable.format('row')]).unsqueeze(-1) for common_variable in COMMON_VARIABLES], dim=1)
        features_row = torch.cat([features_row, Tensor(featuresData['row_firstletter_capitalOrNumeric']).unsqueeze(-1)], dim=1)

        # Load sample | Features | Cols
        features_col = torch.cat(tensors=[Tensor(featuresData[common_variable.format('col')]).unsqueeze(-1) for common_variable in COMMON_VARIABLES], dim=1)
       
        # Load sample | Features | Image (0-1 float)
        image = read_image(pathImage, mode=ImageReadMode.GRAY).to(dtype=torch.float32) / 255

        # Optional transform
        if self.transform_image:
            image = self.transform_image(image)
        if self.transform_target:
            for labelType in sample['labels']:
                sample['labels'][labelType] = self.transform_target(sample['labels'][labelType])

        # Collect in namedtuples
        features = Features(row=features_row.to(DEVICE), col=features_col.to(DEVICE), image=image.to(DEVICE))
        sample = Sample(features=features, targets=targets, meta=meta)

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

        self.layer_linear_row = self.addLayer(layerType=nn.Linear, in_features=7+CONV_FINAL_CHANNELS, out_features=1)
        self.layer_linear_col = self.addLayer(layerType=nn.Linear, in_features=6+CONV_FINAL_CHANNELS, out_features=1)

        self.layer_conv = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=CONV_LETTER_CHANNELS, kernel_size=CONV_LETTER_KERNEL, padding='same', padding_mode='replicate')),
            ('relu1', nn.PReLU())
            # ('conv2', nn.Conv2d(in_channels=5, out_channels=2, kernel_size=CONV_SEQUENCE_KERNEL, padding='same', padding_mode='replicate')),
        ]))
        self.layer_avg_row = nn.AdaptiveAvgPool2d(output_size=(None, 1))
        self.layer_avg_col = nn.AdaptiveAvgPool2d(output_size=(1, None))

        self.layer_conv_preds_row = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(in_channels=1, out_channels=CONV_PREDS_CHANNELS, kernel_size=4, padding='same', padding_mode='replicate')),
            ('relu1', nn.PReLU()),
            ('conv2', nn.Conv1d(in_channels=CONV_PREDS_CHANNELS, out_channels=CONV_PREDS_CHANNELS, kernel_size=16, padding='same', padding_mode='replicate')),
        ]))
        self.layer_conv_preds_col = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(in_channels=1, out_channels=CONV_PREDS_CHANNELS, kernel_size=4, padding='same', padding_mode='replicate')),
            ('relu1', nn.PReLU()),
            ('conv2', nn.Conv1d(in_channels=CONV_PREDS_CHANNELS, out_channels=CONV_PREDS_CHANNELS, kernel_size=16, padding='same', padding_mode='replicate')),
        ]))
        
        self.layer_fc_row = nn.Sequential(OrderedDict([
            ('lin1', nn.Linear(in_features=1+CONV_PREDS_CHANNELS, out_features=3)),
            ('relu1', nn.PReLU()),
            ('lin2', nn.Linear(in_features=3, out_features=1))
        ]))
        self.layer_fc_col = nn.Sequential(OrderedDict([
            ('lin1', nn.Linear(in_features=1+CONV_PREDS_CHANNELS, out_features=3)),
            ('relu1', nn.PReLU()),
            ('lin2', nn.Linear(in_features=3, out_features=1))
        ]))

        self.layer_logit = nn.Sigmoid()

    
    def addLayer(self, layerType:nn.Module, in_features:int, out_features:int, activation=nn.PReLU):
        sequence = nn.Sequential(OrderedDict([
            (f'lin1_from{in_features}_to{self.hidden_sizes[0]}', layerType(in_features=in_features, out_features=self.hidden_sizes[0])),
            (f'relu_1', activation()),
            (f'lin2_from{self.hidden_sizes[0]}_to{self.hidden_sizes[1]}', layerType(in_features=self.hidden_sizes[0], out_features=self.hidden_sizes[1])),
            (f'relu_2', activation()),
            (f'lin3_from{self.hidden_sizes[1]}_to{out_features}', layerType(in_features=self.hidden_sizes[1], out_features=out_features))
        ]))
        return sequence
    
    def forward(self, features):
        # Convert tuple to namedtuple
        if type(features) == type((0,)):
            features = Features(**{field: features[i] for i, field in enumerate(Features._fields)})
            
        # Image
        # Image | Convolutional layers based on image
        img_intermediate_values = self.layer_conv(features.image)
        row_conv_values = self.layer_avg_row(img_intermediate_values).view(1, -1, CONV_FINAL_CHANNELS)
        col_conv_values = self.layer_avg_col(img_intermediate_values).view(1, -1, CONV_FINAL_CHANNELS)

        # Row
        row_inputs = torch.cat([features.row, row_conv_values], dim=-1)

        # Row | Layers
        row_direct_preds = self.layer_linear_row(row_inputs)
        row_convolved_preds = self.layer_conv_preds_row(row_direct_preds.view(1, 1, -1)).view(1, -1, CONV_PREDS_CHANNELS)

        row_preds = self.layer_fc_row(torch.cat([row_direct_preds, row_convolved_preds], dim=-1))
        row_probs = self.layer_logit(row_preds)

        # Col
        col_inputs = torch.cat([features.col, col_conv_values], dim=-1)

        # Col | Layers
        col_direct_preds = self.layer_linear_col(col_inputs)
        col_convolved_preds = self.layer_conv_preds_col(col_direct_preds.view(1, 1, -1)).view(1, -1, CONV_PREDS_CHANNELS)

        col_preds = self.layer_fc_col(torch.cat([col_direct_preds, col_convolved_preds], dim=-1))
        col_probs = self.layer_logit(col_preds)

        # Output
        return Output(row=row_probs, col=col_probs)

model = TabliterModel(hidden_sizes=HIDDEN_SIZES).to(DEVICE)
sample = next(iter(dataloader_train))
logits = model(sample.features)

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

targets_row = [(sample.targets.row.sum().item(), sample.targets.row.shape[0]) for sample in iter(dataset_train)]
targets_col = [(sample.targets.col.sum().item(), sample.targets.col.shape[0]) for sample in iter(dataset_train)]
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

def calculateLoss(targets, preds, lossFunctions:dict, calculateCorrect=False):   
    loss = torch.empty(size=(LOSS_TYPES_COUNT,1), device=DEVICE, dtype=torch.float32)
    correct, maxCorrect = torch.empty(size=(LOSS_TYPES_COUNT,1), device=DEVICE, dtype=torch.int64), torch.empty(size=(LOSS_TYPES_COUNT,1), device=DEVICE, dtype=torch.int64)

    for idx, lossType in enumerate(LOSS_TYPES):
        target = targets[idx].to(DEVICE)
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
lossFunctions = {'row': WeightedBinaryCrossEntropyLoss(weights=classWeights['row']),
                 'col': WeightedBinaryCrossEntropyLoss(weights=classWeights['col'])} 
optimizer = torch.optim.SGD(model.parameters(), lr=MAX_LR)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', verbose=True)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, steps_per_epoch=len(dataloader_train), epochs=EPOCHS)

def train_loop(dataloader, model, lossFunctions, optimizer, report_frequency=4):
    print('Train')
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    epoch_loss = 0
    for batchNumber, batch in enumerate(dataloader):     # batch, sample = next(enumerate(dataloader))
        # Compute prediction and loss
        preds = model(batch.features)
        loss = calculateLoss(batch.targets, preds, lossFunctions)
        epoch_loss += loss
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        # Report intermediate losses
        report_batch_size = (size / batch_size) // report_frequency
        if (batchNumber+1) % report_batch_size == 0:
            epoch_loss, current = epoch_loss.item(), (batchNumber+1) * batch_size
            print(f'\tAvg epoch loss: {epoch_loss/current:>7f} [{current:>5d}/{size:>5d}]')

def val_loop(dataloader, model, lossFunctions):
    batchCount = len(dataloader)
    val_loss, correct, maxCorrect = torch.zeros(size=(LOSS_TYPES_COUNT,1), device=DEVICE), torch.zeros(size=(LOSS_TYPES_COUNT,1), device=DEVICE, dtype=torch.int64), torch.zeros(size=(LOSS_TYPES_COUNT,1), device=DEVICE, dtype=torch.int64)
    with torch.no_grad():
        for batch in dataloader:     # batch = next(iter(dataloader))
            # Compute prediction and loss
            preds = model(batch.features)
            val_loss_batch, correct_batch, maxCorrect_batch = calculateLoss(batch.targets, preds, lossFunctions, calculateCorrect=True)
            val_loss += val_loss_batch
            correct  += correct_batch
            maxCorrect  += maxCorrect_batch

    val_loss = val_loss / batchCount
    shareCorrect = correct / maxCorrect

    print(f'''Validation
        Accuracy: {(100*shareCorrect[0].item()):>0.1f}% (row) | {(100*shareCorrect[1].item()):>0.1f}% (col)
        Avg val loss: {val_loss.sum().item():>6f} (total) | {val_loss[0].item():>6f} (row) | {val_loss[1].item():>6f} (col)''')
    
    return val_loss

# Describe model
# Model description | Graph
y = model(sample.features)
make_dot(y, params=dict(model.named_parameters()), show_attrs=True, show_saved=True).render(pathLogs / 'graph', format='png')

# Model description | Count parameters
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
print(model)
count_parameters(model=model)

# Model description | Tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("torchlogs/")
writer.add_graph(model, input_to_model=[sample.features])
writer.close()


for t in range(EPOCHS):
    print(f"\nEpoch {t+1} of {EPOCHS}. Learning rate: {scheduler.get_last_lr()[0]:03f}")
    train_loop(dataloader=dataloader_train, model=model, lossFunctions=lossFunctions, optimizer=optimizer, report_frequency=4)
    val_loss = val_loop(dataloader=dataloader_val, model=model, lossFunctions=lossFunctions)


# Predict
def eval_loop(dataloader, model, lossFunctions, outPath=None):
    batchCount = len(dataloader)
    eval_loss, correct, maxCorrect = torch.zeros(size=(LOSS_TYPES_COUNT,1), device=DEVICE), torch.zeros(size=(LOSS_TYPES_COUNT,1), device=DEVICE, dtype=torch.int64), torch.zeros(size=(LOSS_TYPES_COUNT,1), device=DEVICE, dtype=torch.int64)
    with torch.no_grad():
        for batchNumber, batch in enumerate(dataloader):
            # Compute prediction and loss
            preds = model(batch.features)
            eval_loss_batch, correct_batch, maxCorrect_batch = calculateLoss(batch.targets, preds, lossFunctions, calculateCorrect=True)
            eval_loss += eval_loss_batch
            correct  += correct_batch
            maxCorrect  += maxCorrect_batch

            # Visualise
            if outPath:
                batch_size = dataloader.batch_size
                for sampleNumber in range(batch_size):      # sampleNumber = 0
                    # Get sample info
                    image = read_image(batch.meta.path_image[sampleNumber]).to(DEVICE)
                    row_targets = batch.targets.row[sampleNumber]
                    row_predictions = preds.row[sampleNumber]
                    col_targets = batch.targets.col[sampleNumber]
                    col_predictions = preds.col[sampleNumber]
                    outName = os.path.basename(batch.meta.path_image[sampleNumber])

                    imagePixels = torch.flatten(image, end_dim=1)
                    row_target_pixels = torch.broadcast_to(input=row_targets, size=(row_targets.numel(), 40)) * TARGET_MAX_LUMINOSITY
                    row_prediction_pixels = (torch.broadcast_to(input=row_predictions, size=(row_predictions.numel(), 40)) >= 0.5)* PREDS_MAX_LUMINOSITY
                    rowPixels = torch.cat([row_target_pixels, row_prediction_pixels], dim=1)
                    img = torch.cat([imagePixels, rowPixels], dim=1)

                    col_target_pixels = torch.broadcast_to(input=col_targets, size=(col_targets.numel(), 40)).T * TARGET_MAX_LUMINOSITY
                    col_prediction_pixels = (torch.broadcast_to(input=col_predictions, size=(col_predictions.numel(), 40)) >= 0.5).T * PREDS_MAX_LUMINOSITY
                    colPixels = torch.cat([col_target_pixels, col_prediction_pixels], dim=0)
                    
                    emptyBlock = torch.full(size=(colPixels.shape[0], rowPixels.shape[1]), fill_value=EMPTY_LUMINOSITY, device=DEVICE)
                    colPixels = torch.cat([colPixels, emptyBlock], axis=1)

                    img = torch.cat([img, colPixels], dim=0).cpu().numpy().astype(np.uint8)
                    cv.imwrite(str(outPath / f'{outName}'), img)

    eval_loss = eval_loss / batchCount
    shareCorrect = correct / maxCorrect

    print(f'''Validation
        Accuracy: {(100*shareCorrect[0].item()):>0.1f}% (row) | {(100*shareCorrect[1].item()):>0.1f}% (col)
        Avg val loss: {eval_loss.sum().item():>6f} (total) | {eval_loss[0].item():>6f} (row) | {eval_loss[1].item():>6f} (col)''')
    
    if outPath:
        return outPath

# Visualize results
eval_loop(dataloader=dataloader_val, model=model, lossFunctions=lossFunctions, outPath=pathData / 'val_annotated')

