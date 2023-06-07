def run():
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
    from collections import namedtuple, OrderedDict
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    from time import perf_counter
    from functools import cache
    from PIL import Image, ImageDraw
    import pandas as pd
    import easyocr
    
    # Constants
    PATH_ROOT = Path(r"F:\ml-parsing-project\table-parse-split")
    SUFFIX = 'narrow'
    DATA_TYPE = 'real'
    PROFILE = False
    TASKS = {'train': False, 'eval': False, 'postprocess':True}
    BEST_RUN = '2023_06_07__13_12'

    LOSS_TYPES = ['row', 'col']
    FEATURE_TYPES = LOSS_TYPES + ['image']

    COMMON_VARIABLES = ['{}_avg', '{}_absDiff', '{}_spell_mean', '{}_spell_sd', '{}_wordsCrossed_count', '{}_wordsCrossed_relToMax']
    ROW_VARIABLES = ['row_between_textlines', 'row_between_textlines_like_rowstart']
    COL_VARIABLES = ['col_nearest_right_is_startlike_share']

    COMMON_GLOBAL_VARIABLES = ['global_{}Avg_p0', 'global_{}Avg_p5', 'global_{}Avg_p10']

    LUMINOSITY_GT_FEATURES_MAX = 240
    LUMINOSITY_FILLER = 255

    PADDING = 40
    TRUTH_THRESHOLD = 0.6
    COLOR_CELL = (102, 153, 255, int(0.1*255))      # light blue
    COLOR_OUTLINE = (255, 255, 255, int(0.9*255))

    # Model parameters
    EPOCHS = 100
    BATCH_SIZE = 1
    MAX_LR = 0.08
    HIDDEN_SIZES = [40, 10]
    CONV_LETTER_KERNEL = [4, 4]
    CONV_LETTER_CHANNELS = 2

    CONV_FINAL_CHANNELS = CONV_LETTER_CHANNELS
    CONV_FINAL_AVG_COUNT = 4

    CONV_PRED_WINDOWS = (4, 10)
    CONV_PRED_CHANNELS = 3

    LAG_LEAD_STRUCTURE = [-4, -2, -1, 1, 2, 4]

    FEATURE_COUNT_ROW = (len(COMMON_VARIABLES + ROW_VARIABLES + COMMON_GLOBAL_VARIABLES) + CONV_FINAL_CHANNELS*CONV_FINAL_AVG_COUNT)*(len(LAG_LEAD_STRUCTURE) + 1)
    FEATURE_COUNT_COL = (len(COMMON_VARIABLES + COL_VARIABLES + COMMON_GLOBAL_VARIABLES) + CONV_FINAL_CHANNELS*CONV_FINAL_AVG_COUNT)*(len(LAG_LEAD_STRUCTURE) + 1)

    # Derived constants
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LOSS_TYPES_COUNT = len(LOSS_TYPES)
    RUN_NAME = datetime.now().strftime("%Y_%m_%d__%H_%M")

    # Paths
    def replaceDirs(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            shutil.rmtree(path)
            os.makedirs(path)
    pathData = PATH_ROOT / 'data' / f'{DATA_TYPE}_{SUFFIX}'
    pathLogs = PATH_ROOT / 'torchlogs'
    pathModels = PATH_ROOT / 'models'

    # Timing
    timers = {}

    # Data
    start_data = perf_counter()
    Sample = namedtuple('sample', ['features', 'targets', 'meta'])
    Meta = namedtuple('meta', ['path_image'])
    Targets = namedtuple('target', LOSS_TYPES)
    Features = namedtuple('features', FEATURE_TYPES)

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
            self.image_fileEntries = list(os.listdir(self.dir_images))
            self.feature_fileEntries = list(os.listdir(self.dir_features))
            self.label_fileEntries = list(os.listdir(self.dir_labels))

            items_images   = set([os.path.splitext(file)[0] for file in self.image_fileEntries])
            items_features = set([os.path.splitext(file)[0] for file in self.feature_fileEntries])
            items_labels   = set([os.path.splitext(file)[0] for file in self.label_fileEntries])

            # Verify consistency between image/feature/label
            self.items = sorted(list(items_images.intersection(items_features).intersection(items_labels)))
            assert len(self.items) == len(items_images) == len(items_features) == len(items_labels), 'Set of images, features and labels do not match.'

        def __len__(self):
            return len(self.items)  
        
        @cache
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
            features_row = torch.cat([features_row, *[Tensor(featuresData[row_variable]).unsqueeze(-1) for row_variable in ROW_VARIABLES]], dim=1)
            features_row = torch.cat([features_row, *[Tensor([featuresData[global_common_variable.format('row')]]).broadcast_to((features_row.shape[0], 1)) for global_common_variable in COMMON_GLOBAL_VARIABLES]], dim=1)

            # Load sample | Features | Cols
            features_col = torch.cat(tensors=[Tensor(featuresData[common_variable.format('col')]).unsqueeze(-1) for common_variable in COMMON_VARIABLES], dim=1)
            features_col = torch.cat([features_col, *[Tensor(featuresData[col_variable]).unsqueeze(-1) for col_variable in COL_VARIABLES]], dim=1)
            features_col = torch.cat([features_col, *[Tensor([featuresData[global_common_variable.format('col')]]).broadcast_to((features_col.shape[0], 1)) for global_common_variable in COMMON_GLOBAL_VARIABLES]], dim=1)
        
            # Load sample | Features | Image (0-1 float)
            image = read_image(pathImage, mode=ImageReadMode.GRAY).to(dtype=torch.float32) / 255

            # Optional transform
            if self.transform_image:
                image = self.transform_image(image)
            if self.transform_target:
                for labelType in sample['labels']:
                    sample['labels'][labelType] = self.transform_target(sample['labels'][labelType])

            # Collect in namedtuples
            features = Features(row=features_row, col=features_col, image=image)
            sample = Sample(features=features, targets=targets, meta=meta)

            # Return sample
            return sample

    dataset_train = TableDataset(dir_data=pathData / 'train')
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True)

    dataset_val = TableDataset(dir_data=pathData / 'val')
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=BATCH_SIZE, shuffle=False)

    dataset_test = TableDataset(dir_data=pathData / 'test')
    dataloader_test = DataLoader(dataset=dataset_val, batch_size=BATCH_SIZE, shuffle=False)

    timers['data'] = perf_counter() - start_data
    

    # Model
    start_model_definition = perf_counter()
    Output = namedtuple('output', LOSS_TYPES)
    class TabliterModel(nn.Module):
        def __init__(self, hidden_sizes=[15,10], layer_depth=3):
            super().__init__()
            self.hidden_sizes = hidden_sizes
            self.layer_depth = layer_depth

            # Convolution on image
            self.layer_conv_img = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(in_channels=1, out_channels=CONV_LETTER_CHANNELS, kernel_size=CONV_LETTER_KERNEL, padding='same', padding_mode='replicate', bias=False)),
                ('norm_conv1', nn.BatchNorm2d(CONV_LETTER_CHANNELS)),
                ('relu1', nn.ReLU()),
                # ('conv2', nn.Conv2d(in_channels=5, out_channels=2, kernel_size=CONV_SEQUENCE_KERNEL, padding='same', padding_mode='replicate')),
            ]))
            self.layer_avg_row = nn.AdaptiveAvgPool2d(output_size=(None, CONV_FINAL_AVG_COUNT))
            self.layer_avg_col = nn.AdaptiveAvgPool2d(output_size=(CONV_FINAL_AVG_COUNT, None))

            # Feature+Conv Neural Net
            self.layer_linear_row = self.addLinearLayer(layerType=nn.Linear, in_features=FEATURE_COUNT_ROW, out_features=1, activation=nn.ReLU)
            self.layer_linear_col = self.addLinearLayer(layerType=nn.Linear, in_features=FEATURE_COUNT_COL, out_features=1, activation=nn.ReLU)

            # Convolution on predictions
            self.layer_conv_preds_row = self.addPredConvLayer()
            self.layer_conv_preds_col = self.addPredConvLayer()
            self.layer_conv_preds_fc_row = self.addPredConvFullyConnectedLayer(in_features=CONV_PRED_CHANNELS+1)
            self.layer_conv_preds_fc_col = self.addPredConvFullyConnectedLayer(in_features=CONV_PRED_CHANNELS+1)

            # Logit model
            self.layer_logit = nn.Sigmoid()
        
        def addLinearLayer(self, layerType:nn.Module, in_features:int, out_features:int, activation=nn.PReLU):
            sequence = nn.Sequential(OrderedDict([
                (f'lin1_from{in_features}_to{self.hidden_sizes[0]}', layerType(in_features=in_features, out_features=self.hidden_sizes[0])),
                (f'relu_1', activation()),
                # ('norm_lin1', nn.BatchNorm1d(self.hidden_sizes[0])),
                (f'lin2_from{self.hidden_sizes[0]}_to{self.hidden_sizes[1]}', layerType(in_features=self.hidden_sizes[0], out_features=self.hidden_sizes[1])),
                (f'relu_2', activation()),
                # ('norm_lin2', nn.BatchNorm1d(self.hidden_sizes[1])),
                (f'lin3_from{self.hidden_sizes[1]}_to{out_features}', layerType(in_features=self.hidden_sizes[1], out_features=out_features))
            ]))
            return sequence
            
        def addPredConvLayer(self):
            sequence = nn.Sequential(OrderedDict([
                ('predconv1', nn.Conv1d(in_channels=1, out_channels=CONV_PRED_CHANNELS, kernel_size=CONV_PRED_WINDOWS[0], padding='same', padding_mode='replicate', bias=False)),
                ('predconv1_norm', nn.BatchNorm1d(CONV_PRED_CHANNELS)),
                ('predconv1_relu', nn.ReLU()),
                ('predconv2', nn.Conv1d(in_channels=CONV_PRED_CHANNELS, out_channels=CONV_PRED_CHANNELS, kernel_size=CONV_PRED_WINDOWS[1], padding='same', padding_mode='replicate', bias=False)),
                ('predconv2_norm', nn.BatchNorm1d(CONV_PRED_CHANNELS)),
                ('predconv2_relu', nn.ReLU())
            ]))
            return sequence
        def addPredConvFullyConnectedLayer(self, in_features:int, hidden_sizes=[6]):
            sequence = nn.Sequential(OrderedDict([
                (f'lin1_from{in_features}_to{hidden_sizes[0]}', nn.Linear(in_features=in_features, out_features=hidden_sizes[0])),
                (f'relu_1', nn.ReLU()),
                (f'lin2_from{hidden_sizes[0]}_to1', nn.Linear(in_features=hidden_sizes[0], out_features=1))
            ]))
            return sequence
        
        def forward(self, features):
            # Convert tuple to namedtuple
            if type(features) == type((0,)):
                features = Features(**{field: features[i] for i, field in enumerate(Features._fields)})
            
            # Load features to GPU
            features = Features(**{field: features[i].to(DEVICE) for i, field in enumerate(Features._fields)})
                
            # Image
            # Image | Convolutional layers based on image
            img_intermediate_values = self.layer_conv_img(features.image)
            row_conv_values = self.layer_avg_row(img_intermediate_values).view(1, -1, CONV_FINAL_CHANNELS*CONV_FINAL_AVG_COUNT)
            col_conv_values = self.layer_avg_col(img_intermediate_values).view(1, -1, CONV_FINAL_CHANNELS*CONV_FINAL_AVG_COUNT)

            # Row
            # Row | Gather input
            row_inputs = torch.cat([features.row, row_conv_values], dim=-1)
            row_inputs_lag_leads = torch.cat([torch.roll(row_inputs, shifts=shift, dims=1) for shift in LAG_LEAD_STRUCTURE], dim=-1)     # dangerous if no padding applied !!!          
            row_inputs_complete = torch.cat([row_inputs, row_inputs_lag_leads], dim=-1)
            
            # Row | Linear prediction
            row_direct_preds = self.layer_linear_row(row_inputs_complete)
            # row_preds = row_direct_preds      # uncomment for linear prediction

            # Row | Convolved prediction
            row_conv_preds = self.layer_conv_preds_row(row_direct_preds.view(1, 1, -1)).view(1, -1, CONV_PRED_CHANNELS)
            row_preds = self.layer_conv_preds_fc_row(torch.cat([row_direct_preds, row_conv_preds], dim=-1))

            # Col
            # Col | Gather input
            col_inputs = torch.cat([features.col, col_conv_values], dim=-1)
            col_inputs_lag_leads = torch.cat([torch.roll(col_inputs, shifts=shift, dims=1) for shift in LAG_LEAD_STRUCTURE], dim=-1)     # dangerous if no padding applied !!!
            col_inputs_complete = torch.cat([col_inputs, col_inputs_lag_leads], dim=-1)

            # Col | Linear prediction
            col_direct_preds = self.layer_linear_col(col_inputs_complete)
            # col_preds = col_direct_preds         # uncomment for linear prediction

            # Col | Convolved prediction
            col_conv_preds = self.layer_conv_preds_col(col_direct_preds.view(1, 1, -1)).view(1, -1, CONV_PRED_CHANNELS)
            col_preds = self.layer_conv_preds_fc_col(torch.cat([col_direct_preds, col_conv_preds], dim=-1))
           
            # Turn into probabilities
            row_probs = self.layer_logit(row_preds)
            col_probs = self.layer_logit(col_preds)

            # Output
            return Output(row=row_probs, col=col_probs)

    model = TabliterModel(hidden_sizes=HIDDEN_SIZES).to(DEVICE)

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

    timers['model_definition'] = perf_counter() - start_model_definition

    # Train
    lossFunctions = {'row': WeightedBinaryCrossEntropyLoss(weights=classWeights['row']),
                    'col': WeightedBinaryCrossEntropyLoss(weights=classWeights['col'])} 
    optimizer = torch.optim.SGD(model.parameters(), lr=MAX_LR)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, steps_per_epoch=len(dataloader_train), epochs=EPOCHS)

    def train_loop(dataloader, model, lossFunctions, optimizer, report_frequency=4):
        print('Train')
        start = perf_counter()
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
                print(f'\tAvg epoch loss: {epoch_loss/current:.3f} [{current:>5d}/{size:>5d}]')
        
        print(f'\tEpoch duration: {perf_counter()-start:.0f}s')
        return epoch_loss / len(dataloader)

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
            Avg val loss: {val_loss.sum().item():.3f} (total) | {val_loss[0].item():.3f} (row) | {val_loss[1].item():.3f} (col)''')
        
        return val_loss.sum()

    if TASKS['train']:
        # Prepare folders
        replaceDirs(pathData / 'val_annotated')  
        pathModel = pathModels / RUN_NAME
        os.makedirs(pathModel, exist_ok=True)

        # Describe model
        # Model description | Graph
        sample = next(iter(dataloader_train))
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
        writer = SummaryWriter(f"torchlogs/{RUN_NAME}")
        writer.add_graph(model, input_to_model=[sample.features])
        
        start_train = perf_counter()
        with torch.autograd.profiler.profile(enabled=PROFILE) as prof:
            best_val_loss = 9e20
            for epoch in range(EPOCHS):
                learning_rate = scheduler.get_last_lr()[0]
                print(f"\nEpoch {epoch+1} of {EPOCHS}. Learning rate: {learning_rate:03f}")
                train_loss = train_loop(dataloader=dataloader_train, model=model, lossFunctions=lossFunctions, optimizer=optimizer, report_frequency=4)
                val_loss = val_loop(dataloader=dataloader_val, model=model, lossFunctions=lossFunctions)

                writer.add_scalar('Train loss', scalar_value=train_loss, global_step=epoch)
                writer.add_scalar('Val loss', scalar_value=val_loss, global_step=epoch)
                writer.add_scalar('Learning rate', scalar_value=learning_rate, global_step=epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), pathModel / 'best.pt')

            torch.save(model.state_dict(), pathModel / f'last.pt')

        if PROFILE:
            print(dataset_train.__getitem__.cache_info())
            print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        timers['train'] = perf_counter() - start_train

    if TASKS['eval']:
        pathModelDict = PATH_ROOT / 'models' / BEST_RUN / 'best.pt'
        model.load_state_dict(torch.load(pathModelDict))
        model.eval()

        # Predict
        start_eval = perf_counter()
        def convert_01_array_to_visual(array, invert=False, width=40) -> np.array:
            luminosity = (1 - array) * LUMINOSITY_GT_FEATURES_MAX if invert else array * LUMINOSITY_GT_FEATURES_MAX
            luminosity = luminosity.round(0).astype(np.uint8)
            luminosity = np.expand_dims(luminosity, axis=1)
            luminosity = np.broadcast_to(luminosity, shape=(luminosity.shape[0], width))
            return luminosity

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
                            # Sample data
                            # Sample data | Image
                            pathImage = Path(batch.meta.path_image[sampleNumber])
                            img_annot = cv.imread(str(pathImage), flags=cv.IMREAD_GRAYSCALE)
                            img_initial_size = img_annot.shape
                            
                            # Sample data | Ground truth
                            gt = {}
                            gt['row'] = batch.targets.row[sampleNumber].squeeze().cpu().numpy()
                            gt['col'] = batch.targets.col[sampleNumber].squeeze().cpu().numpy()

                            predictions = {}
                            predictions['row'] = preds.row[sampleNumber].squeeze().cpu().numpy()
                            predictions['col'] = preds.col[sampleNumber].squeeze().cpu().numpy()
                            outName = f'{pathImage.stem}.png'

                            # Sample data | Features
                            pathFeatures = pathImage.parent.parent / 'features' / f'{pathImage.stem}.json'
                            with open(pathFeatures, 'r') as f:
                                features = json.load(f)
                            features = {key: np.array(value) for key, value in features.items()}

                            # Draw
                            row_annot = []
                            col_annot = []

                            # Draw | Ground truth
                            gt_row = convert_01_array_to_visual(gt['row'], width=40)
                            row_annot.append(gt_row)
                            gt_col = convert_01_array_to_visual(gt['col'], width=40)
                            col_annot.append(gt_col)
                        
                            # Draw | Features | Text is startlike
                            indicator_textline_like_rowstart = convert_01_array_to_visual(features['row_between_textlines_like_rowstart'], width=20)
                            row_annot.append(indicator_textline_like_rowstart)
                            indicator_nearest_right_is_startlike = convert_01_array_to_visual(features['col_nearest_right_is_startlike_share'], width=20)
                            col_annot.append(indicator_nearest_right_is_startlike)

                            # Draw | Features | Words crossed (lighter = fewer words crossed)
                            wc_row = convert_01_array_to_visual(features['row_wordsCrossed_relToMax'], invert=True, width=20)
                            row_annot.append(wc_row)
                            wc_col = convert_01_array_to_visual(features['col_wordsCrossed_relToMax'], invert=True, width=20)
                            col_annot.append(wc_col)

                            # Draw | Features | Add feature bars
                            row_annot = np.concatenate(row_annot, axis=1)
                            img_annot = np.concatenate([img_annot, row_annot], axis=1)

                            col_annot = np.concatenate(col_annot, axis=1).T
                            col_annot = np.concatenate([col_annot, np.full(shape=(col_annot.shape[0], row_annot.shape[1]), fill_value=LUMINOSITY_FILLER, dtype=np.uint8)], axis=1)
                            img_annot = np.concatenate([img_annot, col_annot], axis=0)

                            # Draw | Predictions
                            img_predictions_row = np.full(img_annot.shape, fill_value=255, dtype=np.uint8)
                            indicator_predictions_row = convert_01_array_to_visual(1-predictions['row'], width=img_initial_size[1])
                            img_predictions_row[:indicator_predictions_row.shape[0], :indicator_predictions_row.shape[1]] = indicator_predictions_row
                            img_predictions_row = cv.cvtColor(img_predictions_row, code=cv.COLOR_GRAY2RGB)
                            img_predictions_row[:, :, 0] = 255
                            img_predictions_row = Image.fromarray(img_predictions_row).convert('RGBA')
                            img_predictions_row.putalpha(int(0.1*255))

                            img_predictions_col = np.full(img_annot.shape, fill_value=255, dtype=np.uint8)
                            indicator_predictions_col = convert_01_array_to_visual(1-predictions['col'], width=img_initial_size[0]).T
                            img_predictions_col[:indicator_predictions_col.shape[0], :indicator_predictions_col.shape[1]] = indicator_predictions_col
                            img_predictions_col = cv.cvtColor(img_predictions_col, code=cv.COLOR_GRAY2RGB)
                            img_predictions_col[:, :, 0] = 255
                            img_predictions_col = Image.fromarray(img_predictions_col).convert('RGBA')
                            img_predictions_col.putalpha(int(0.1*255))

                            img_annot_color = Image.fromarray(cv.cvtColor(img_annot, code=cv.COLOR_GRAY2RGB)).convert('RGBA')
                            img_predictions = Image.alpha_composite(img_predictions_col, img_predictions_row)
                            img_complete = Image.alpha_composite(img_annot_color, img_predictions).convert('RGB')

                            img_complete.save(outPath / f'{outName}', format='png')

            eval_loss = eval_loss / batchCount
            shareCorrect = correct / maxCorrect

            print(f'''Validation
                Accuracy: {(100*shareCorrect[0].item()):>0.1f}% (row) | {(100*shareCorrect[1].item()):>0.1f}% (col)
                Avg val loss: {eval_loss.sum().item():.3f} (total) | {eval_loss[0].item():.3f} (row) | {eval_loss[1].item():.3f} (col)''')
            
            if outPath:
                return outPath

        # Visualize results
        eval_loop(dataloader=dataloader_val, model=model, lossFunctions=lossFunctions, outPath=pathData / 'val_annotated')
        timers['eval'] = perf_counter() - start_eval

        # Close tensorboard writer
        writer.add_hparams(hparam_dict={'epochs': EPOCHS,
                                        'batch_size': BATCH_SIZE,
                                        'max_lr': MAX_LR},
                        metric_dict={'val_loss': val_loss.sum().item()/len(dataloader_val)})
        writer.close()

    if TASKS['postprocess']:
        def preds_to_separators(predArray, paddingSeparator, threshold=0.8):
            if isinstance(predArray, torch.Tensor):
                predArray = predArray.cpu().numpy().squeeze()
                
            is_separator = (predArray > threshold)
            diff_in_separator_modus = np.diff(is_separator.astype(np.int8))
            separators_start = np.where(diff_in_separator_modus == 1)[0]
            separators_end = np.where(diff_in_separator_modus == -1)[0]
            separators = np.stack([separators_start, separators_end], axis=1)
            separators = np.concatenate([paddingSeparator, separators], axis=0)
            return separators

        # Load model
        pathModel = PATH_ROOT / 'models' / BEST_RUN
        pathModelDict =  pathModel / 'best.pt'
        model.load_state_dict(torch.load(pathModelDict))
        model.eval()

        # Load OCR reader
        reader = easyocr.Reader(lang_list=['nl', 'fr', 'de', 'en'], gpu=True, quantize=True)

        # Prepare output folder
        outPath = pathModel / 'predictions_data'
        replaceDirs(outPath)

        # Define loop
        dataloader = dataloader_val
        outPath = outPath
        model = model

        # Padding separator
        paddingSeparator = np.array([[0, 40]])

        with torch.no_grad():
            for batchNumber, batch in enumerate(dataloader):
                # Compute prediction
                preds = model(batch.features)

                # Convert to data
                batch_size = dataloader.batch_size
                for sampleNumber in range(batch_size):
                    # Words
                    # Words | Parse sample name
                    image_path = Path(batch.meta.path_image[sampleNumber])
                    name_full = image_path.stem
                    name_stem = name_full.split('-p')[0]

                    name_pdf = f"{name_stem}.pdf"
                    pageNumber = int(name_full.split('-p')[-1].split('_t')[0])
                    tableNumber = int(name_full.split('_t')[-1])

                    name_words = f"{name_stem}-p{pageNumber}.pq"
                    wordsPath = PATH_ROOT / 'data' / 'words' / f"{name_words}"

                    # Words | Use pdf-based words if present
                    wordsDf = pd.read_parquet(wordsPath)
                    finalOcrNeeded = (len(wordsDf.query('ocrLabel == "pdf"')) == 0)

                    # Cells     # td: use a class here with rows cols etc
                    # Cells | Convert predictions to boundaries
                    separators_row = preds_to_separators(predArray=preds.row[sampleNumber], paddingSeparator=paddingSeparator)
                    separators_col = preds_to_separators(predArray=preds.col[sampleNumber], paddingSeparator=paddingSeparator)

                    # Cells | Convert boundaries to cells
                    cells = [dict(x0=separators_col[c][1]+1, y0=separators_row[r][1]+1, x1=separators_col[c+1][1], y1=separators_row[r+1][1], row=r, col=c)
                             for r in range(len(separators_row)-1) for c in range(len(separators_col)-1)]
                    
                    # Data 
                    # Data | OCR by initial cell
                    img_array = cv.imread(str(image_path), flags=cv.IMREAD_GRAYSCALE)
                    
                    for cell in cells:
                        cell['text'] = ' '.join(reader.readtext(image=img_array[cell['y0']:cell['y1'], cell['x0']:cell['x1']], batch_size=60, detail=0))
                        print(cell['text'])


                    
                    # Visualise
                    # Visualise | Load image
                    image = Image.open(image_path).convert('RGBA')

                    # Visualise | Cell annotations
                    overlay = Image.new('RGBA', image.size, (0,0,0,0))
                    img_annot = ImageDraw.Draw(overlay)
                    for cell in cells:
                        img_annot.rectangle(xy=(cell['x0'], cell['y0'], cell['x1'], cell['y1']), fill=COLOR_CELL, outline=COLOR_OUTLINE, width=2)
                    
                    # Visualise | Save image
                    image = Image.alpha_composite(image, overlay)
                    image.show()


        

    # Reporting timings
    for key, value in timers.items():
        print(f'{key}: {value:>4f}')

if __name__ == '__main__':
    run()