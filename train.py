# Imports
import os, shutil, sys
from pathlib import Path
from time import perf_counter
from datetime import datetime
from tqdm import tqdm, trange

import torch
import utils
from models import TableLineModel, LOSS_ELEMENTS_LINELEVEL_COUNT, LOSS_ELEMENTS_SEPARATORLEVEL_COUNT
import models
from loss import defineLossFunctions_lineLevel, defineLossFunctions_separatorLevel, calculateLoss_lineLevel, calculateLoss_separatorLevel
from dataloaders import get_dataloader_lineLevel, get_dataloader_separatorLevel
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')
import numpy as np
from math import ceil
from collections import defaultdict

# Helper functions
def add_weights(model, epoch, writer, disable=False):
    if not disable:
        for name, parameter in tqdm(list(model.named_parameters()), desc='Visualizing weights', leave=False, position=1):
            weight = parameter.data.flatten().cpu().numpy()       
            fig = plt.figure()
            weightCount = len(weight)
            plt.bar(np.arange(weightCount), weight, width=1)
            if 1 < weightCount < 10:
                plt.xticks(np.arange(weightCount), np.arange(weightCount) + 1)
            plt.xlabel(name)
            writer.add_figure(tag=name, figure=fig, global_step=epoch, close=True)

def train_lineLevel(path_data_train, path_data_val, path_model, path_model_add_timestamp=False, shuffle_train_data=True, epochs=3, max_lr=0.1, 
          profile=False, device='cuda', writer=None, path_writer=None, tensorboard_detail_frequency=10, disable_weight_visualisation=True,
          replace_dirs='warn', legacy_folder_names=False, batch_size=1):
    # Parse parameters
    path_model = Path(path_model)
    path_model = path_model if not path_model_add_timestamp else path_model / datetime.now().strftime("%Y_%m_%d__%H_%M")

    path_writer = path_writer or f"torchlogs/{path_model.stem}"
    utils.makeDirs(path_writer, replaceDirs=replace_dirs)
    writer = writer or SummaryWriter(path_writer)

    # Create folders
    utils.makeDirs(path_model, replaceDirs=replace_dirs)

    # Initialize elements
    model = TableLineModel().to(device)

    dataloader_train = get_dataloader_lineLevel(dir_data=path_data_train, ground_truth=True, legacy_folder_names=legacy_folder_names, shuffle=shuffle_train_data, device=device, batch_size=batch_size)
    dataloader_val = get_dataloader_lineLevel(dir_data=path_data_val, ground_truth=True, legacy_folder_names=legacy_folder_names, shuffle=False, device=device, batch_size=batch_size)

    lossFunctions = defineLossFunctions_lineLevel(dataloader=dataloader_train, path_model=path_model)
    optimizer = torch.optim.SGD(model.parameters(), lr=max_lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/10, end_factor=1, total_iters=epochs//5+1)

    # Define single epoch training loop
    def train_loop(dataloader, model, lossFunctions, optimizer, report_frequency=4, device=device):
        start = perf_counter()
        size = len(dataloader.dataset)
        batch_size = dataloader.batch_size or dataloader.batch_sampler.batch_size
        epoch_loss = 0
        for batchNumber, batch in tqdm(enumerate(dataloader), desc='Looping over batches', total=len(dataloader), unit=' batches', smoothing=0.1, position=1, leave=False):     # batch, sample = next(enumerate(dataloader))
            # Compute prediction and loss
            preds = model(batch.features)
            loss = calculateLoss_lineLevel(batch.targets, preds, lossFunctions, shapes=batch.meta.size_image, device=device)
            epoch_loss += loss
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Report intermediate losses
            report_batch_size = (size / batch_size) // report_frequency
            if (batchNumber+1) % report_batch_size == 0:
                epoch_loss, current = epoch_loss.item(), (batchNumber+1) * batch_size
                tqdm.write(f'\tAvg epoch loss: {epoch_loss/current:.3f} [{current:>5d}/{size:>5d}]')
        
        print(f'\n\tEpoch duration: {perf_counter()-start:.0f}s')
        return epoch_loss / len(dataloader)
    
    # Define single epoch validation loop
    def val_loop(dataloader, model, lossFunctions, device=device):
        batchCount = len(dataloader)
        val_loss, correct, maxCorrect = torch.zeros(size=(LOSS_ELEMENTS_LINELEVEL_COUNT,1), device=device), torch.zeros(size=(LOSS_ELEMENTS_LINELEVEL_COUNT,1), device=device, dtype=torch.int64), torch.zeros(size=(LOSS_ELEMENTS_LINELEVEL_COUNT,1), device=device, dtype=torch.int64)
        with torch.no_grad():
            for batch in dataloader:     # batch = next(iter(dataloader))
                # Compute prediction and loss
                preds = model(batch.features)
                val_loss_batch, correct_batch, maxCorrect_batch = calculateLoss_lineLevel(batch.targets, preds, lossFunctions, shapes=batch.meta.size_image, calculateCorrect=True)
                val_loss += val_loss_batch
                correct  += correct_batch
                maxCorrect  += maxCorrect_batch

        val_loss = val_loss / batchCount
        shareCorrect = correct / maxCorrect

        print(f'''Validation
        Accuracy line-level: {(100*shareCorrect[0].item()):>0.1f}% (row) | {(100*shareCorrect[2].item()):>0.1f}% (col)
        Separator count (relative to truth): {(100*shareCorrect[1].item()):>0.1f}% (row) | {(100*shareCorrect[3].item()):>0.1f}% (col)
        Avg val loss: {val_loss.sum().item():.3f} (total) | {val_loss[0].item():.3f} (row-line) | {val_loss[2].item():.3f} (col-line) | {val_loss[1].item():.3f} (row-separator) | {val_loss[3].item():.3f} (col-separator)''')
        
        return val_loss
    
    # Train
    start_train = perf_counter()
    with torch.autograd.profiler.profile(enabled=profile) as prof:
        best_val_loss = 9e20
        for epoch in trange(epochs, desc='Looping over epochs', position=2, unit=' epochs', smoothing=0.1, file=sys.stdout):
            learning_rate = scheduler.get_last_lr()[0]
            print(f"\nEpoch {epoch+1} of {epochs}. Learning rate: {learning_rate:03f}")
            model.train()
            train_loss = train_loop(dataloader=dataloader_train, model=model, lossFunctions=lossFunctions, optimizer=optimizer, report_frequency=4, device=device)
            model.eval()
            val_loss = val_loop(dataloader=dataloader_val, model=model, lossFunctions=lossFunctions, device=device)
            scheduler.step()

            writer.add_scalar('Train/Loss', scalar_value=train_loss, global_step=epoch)
            writer.add_scalar('Val/Loss/Total', scalar_value=val_loss.sum(), global_step=epoch)
            writer.add_scalar('Val/Loss/Row Line', val_loss[0], global_step=epoch)
            writer.add_scalar('Val/Loss/Col Line', val_loss[2], global_step=epoch)
            writer.add_scalar('Val/Loss/Row Separator Count', val_loss[1], global_step=epoch)
            writer.add_scalar('Val/Loss/Col Separator Count', val_loss[3], global_step=epoch)
            writer.add_scalar('Learning rate', scalar_value=learning_rate, global_step=epoch)

            if val_loss.sum() < best_val_loss:
                best_val_loss = val_loss.sum()
                torch.save(model.state_dict(), path_model / 'model_best.pt')

            if (epoch % tensorboard_detail_frequency == 0):
                add_weights(model=model, epoch=epoch, writer=writer, disable=disable_weight_visualisation)
        torch.save(model.state_dict(), path_model / f'model_last.pt')
        print('\n\n', flush=True)

        if profile:
            print(prof.key_averages().table(sort_by="self_cpu_time_total"))

        # Report
        train_duration = perf_counter() - start_train
        print(f'Trained for {train_duration/60:0.2f} minutes')

        # Tensorboard
        # Tensorboard | Add summary statistics to tensorboard
        writer.add_hparams(hparam_dict={'epochs': epochs,
                                'batch_size': dataloader_train.batch_size,
                                'max_lr': max_lr},
                metric_dict={'val_loss': best_val_loss.sum().item()})
        
        # Tensorboard | Add precision-recall curve
        model.load_state_dict(torch.load(path_model / 'model_best.pt')); model.eval()
        preds = []
        targets = []
        with torch.no_grad():
            for batch in tqdm(dataloader_train, desc="Adding precision-recall curve", leave=False):
                #### NEEDS UPDATING TO BATCHED PREDS (see separator model for inspiration)
                pred = torch.cat([torch.flatten(preds) for preds in model(batch.features)])
                preds.append(pred)

                target = torch.cat([torch.flatten(batch.targets.row_line), torch.flatten(batch.targets.col_line)])
                targets.append(target)
        preds = torch.cat([pred for pred in preds])
        targets = torch.cat([target for target in targets])

        writer.add_pr_curve(tag='Train/All', labels=targets, predictions=preds, global_step=epochs, num_thresholds=1024)

        # Tensorboard | Add weights histogram
        add_weights(model=model, epoch=epochs, writer=writer, disable=disable_weight_visualisation)
        writer.close()

def train_separatorLevel(path_data_train, path_data_val, path_model, path_model_add_timestamp=False, shuffle_train_data=True, epochs=3, max_lr=0.08, 
          profile=False, device='cuda', writer=None, path_writer=None, tensorboard_detail_frequency=10, disable_weight_visualisation=False,
          replace_dirs='warn', batch_size=1):
    # Parse parameters
    path_model = Path(path_model)
    path_model = path_model if not path_model_add_timestamp else path_model / datetime.now().strftime("%Y_%m_%d__%H_%M")

    path_writer = path_writer or f"torchlogs/{path_model.stem}"
    utils.makeDirs(path_writer, replaceDirs=replace_dirs)
    writer = writer or SummaryWriter(path_writer)

    # Create folders
    utils.makeDirs(path_model, replaceDirs=replace_dirs)

    # Initialize elements
    model = models.TableSeparatorModel(device=device).to(device)
    model.train()

    dataloader_train = get_dataloader_separatorLevel(dir_data=path_data_train, ground_truth=True, shuffle=shuffle_train_data, device=device, batch_size=batch_size)
    dataloader_val = get_dataloader_separatorLevel(dir_data=path_data_val, ground_truth=True, shuffle=False, device=device, batch_size=batch_size)

    lossFunctions = defineLossFunctions_separatorLevel(dataloader=dataloader_train, path_model=path_model)
    optimizer = torch.optim.SGD(model.parameters(), lr=max_lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/10, end_factor=1, total_iters=epochs//5+1)

    # Define single epoch training loop
    def train_loop(dataloader, model, lossFunctions, optimizer, report_frequency=4, device=device):
        start = perf_counter()
        size = len(dataloader.dataset)
        batch_size = dataloader.batch_size or dataloader.batch_sampler.batch_size
        epoch_loss = 0
        for batchNumber, batch in tqdm(enumerate(dataloader), desc='Looping over batches', total=len(dataloader), unit=' batches', smoothing=0.1, position=1, leave=False):     # batch, sample = next(enumerate(dataloader))
            # Compute prediction and loss
            preds = model(batch.features)
            loss = calculateLoss_separatorLevel(batch.targets, preds, lossFunctions, shapes=batch.meta.count_separators, device=device)     # fix loss calculation
            epoch_loss += loss
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Report intermediate losses
            report_batch_size = (size / batch_size) // report_frequency
            if (batchNumber+1) % report_batch_size == 0:
                epoch_loss, current = epoch_loss.item(), (batchNumber+1) * batch_size
                tqdm.write(f'\tAvg epoch loss: {epoch_loss/current:.3f} [{current:>5d}/{size:>5d}]')
        
        tqdm.write(f'\n\tEpoch duration: {perf_counter()-start:.0f}s')
        return epoch_loss / len(dataloader)
    
    # Define single epoch validation loop
    def val_loop(dataloader, model, lossFunctions, device=device):
        batchCount = len(dataloader)
        val_loss, correct, maxCorrect = torch.zeros(size=(LOSS_ELEMENTS_SEPARATORLEVEL_COUNT,1), device=device), torch.zeros(size=(LOSS_ELEMENTS_SEPARATORLEVEL_COUNT,1), device=device, dtype=torch.int64), torch.zeros(size=(LOSS_ELEMENTS_SEPARATORLEVEL_COUNT,1), device=device, dtype=torch.int64)
        with torch.no_grad():
            for batch in dataloader:     # batch = next(iter(dataloader))
                # Compute prediction and loss
                preds = model(batch.features)
                val_loss_batch, correct_batch, maxCorrect_batch = calculateLoss_separatorLevel(batch.targets, preds, lossFunctions, shapes=batch.meta.count_separators, calculateCorrect=True)
                val_loss += val_loss_batch
                correct  += correct_batch
                maxCorrect  += maxCorrect_batch

        val_loss = val_loss / batchCount
        shareCorrect = correct / maxCorrect

        tqdm.write(f'''Validation
        Accuracy separator-level: {(100*shareCorrect[0].item()):>0.1f}% (row) | {(100*shareCorrect[1].item()):>0.1f}% (col)
        Avg val loss: {val_loss.sum().item():.3f} (total) | {val_loss[0].item():.3f} (row) | {val_loss[1].item():.3f} (col)''')
        
        return val_loss
    
    # Train
    start_train = perf_counter()
    weightDict = {}
    print(model)
    from torchviz import make_dot
    make_dot(next(iter(dataloader_train)).features, params=dict(model.named_parameters()), show_attrs=True).render(Path(path_writer) / 'graph', format='png')
    with torch.autograd.profiler.profile(enabled=profile) as prof:
        best_val_loss = 9e20
        for epoch in trange(epochs, desc='Looping over epochs', position=2, unit=' epochs', smoothing=0.1):
            learning_rate = scheduler.get_last_lr()[0]
            tqdm.write(f"\nEpoch {epoch+1} of {epochs}. Learning rate: {learning_rate:03f}")
            model.train()
            train_loss = train_loop(dataloader=dataloader_train, model=model, lossFunctions=lossFunctions, optimizer=optimizer, report_frequency=4, device=device)
            
            for param in model.named_parameters():
                name = param[0]
                newMean = torch.sqrt(torch.mean(torch.square(param[1])))
                if name in weightDict:
                    tqdm.write(f'{param[0]}: {(newMean - weightDict[name])/weightDict[name]:0.2f}')
                else:
                    weightDict[name] = newMean
                    tqdm.write(f'Grad {param[1].grad_fn} | Weight {name} | {newMean:0.2f}')
            
            model.eval()
            val_loss = val_loop(dataloader=dataloader_val, model=model, lossFunctions=lossFunctions, device=device)
            
            scheduler.step()
            writer.add_scalar('Train/Loss', scalar_value=train_loss, global_step=epoch)
            writer.add_scalar('Val/Loss/Total', scalar_value=val_loss.sum(), global_step=epoch)
            writer.add_scalar('Val/Loss/Row Separator', val_loss[0], global_step=epoch)
            writer.add_scalar('Val/Loss/Col Separator', val_loss[1], global_step=epoch)
            writer.add_scalar('Learning rate', scalar_value=learning_rate, global_step=epoch)

            if val_loss.sum() < best_val_loss:
                best_val_loss = val_loss.sum()
                torch.save(model.state_dict(), path_model / 'model_best.pt')

            if (epoch % tensorboard_detail_frequency == 0):
                add_weights(model=model, epoch=epoch, writer=writer, disable=disable_weight_visualisation)

        print('', flush=True)
        torch.save(model.state_dict(), path_model / f'model_last.pt')

        if profile:
            print(prof.key_averages().table(sort_by="self_cpu_time_total"))

        # Report
        train_duration = perf_counter() - start_train
        print(f'Trained for {train_duration/60:0.2f} minutes')

        # Tensorboard
        # Tensorboard | Add summary statistics to tensorboard
        writer.add_hparams(hparam_dict={'epochs': epochs,
                                'batch_size': dataloader_train.batch_size,
                                'max_lr': max_lr},
                metric_dict={'val_loss': best_val_loss.sum().item()})
        
        # Tensorboard | Add precision-recall curve
        model.load_state_dict(torch.load(path_model / 'model_best.pt')); model.eval()
        preds = []
        targets = []
        with torch.no_grad():
            for batch in tqdm(dataloader_train, desc="Adding precision-recall curve", leave=False):
                preds_batch = model(batch.features)
                shapes = batch.meta.count_separators
                preds_row = torch.cat([preds_batch.row[sampleNumber, :shape['row']] for sampleNumber, shape in enumerate(shapes)])
                preds_col = torch.cat([preds_batch.col[sampleNumber, :shape['col']] for sampleNumber, shape in enumerate(shapes)])
                pred = torch.cat([preds_row, preds_col]).squeeze()
                preds.append(pred)

                targets_row = torch.cat([batch.targets.row[sampleNumber, :shape['row']] for sampleNumber, shape in enumerate(shapes)])
                targets_col = torch.cat([batch.targets.col[sampleNumber, :shape['col']] for sampleNumber, shape in enumerate(shapes)])
                target = torch.cat([targets_row, targets_col]).squeeze()
                targets.append(target)

        preds = torch.cat([pred for pred in preds])
        targets = torch.cat([target for target in targets])

        writer.add_pr_curve(tag='Train/All', labels=targets, predictions=preds, global_step=epochs, num_thresholds=1024)

        # Tensorboard | Add weights histogram
        add_weights(model=model, epoch=epochs, writer=writer)
        writer.close()

if __name__ == '__main__':
    PATH_ROOT = Path(r'F:\ml-parsing-project\table-parse-split\data\tableparse_round2\splits')
    path_model = Path(r"F:\ml-parsing-project\table-parse-split\models") / 'batchsize_test'
    batch_size = 3
    # train_lineLevel(path_data_train=PATH_ROOT / 'val', path_data_val=PATH_ROOT / 'val', path_model=path_model, replace_dirs='warn', batch_size=batch_size)
    train_separatorLevel(path_data_train=PATH_ROOT / 'val', path_data_val=PATH_ROOT / 'val', path_model=path_model, replace_dirs=True, batch_size=batch_size)