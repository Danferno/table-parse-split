# Imports
import torch
from torch import nn
from model import LOSS_ELEMENTS, LOSS_ELEMENTS_COUNT, ORIENTATIONS
 

# Loss classes
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
    
class LogisticLoss(nn.Module):
    ''' The logistic loss is 0 when the distance between the two targets is 0 and has a maximum of 1 when the distance is infinitely large.
    Higher distances lead to higher losses, but the effect diminishes to keep the size of the loss bounded between [0, 1) '''
    def __init__(self, limit_upper:float=1):
        super(LogisticLoss, self).__init__()
        self.limit_upper = limit_upper
        self.scale = self.limit_upper / (1/2)

    def forward(self, input, target):
        distance = torch.abs(target - input)
        loss = self.scale / (1 + torch.exp(-0.5*distance)) - self.limit_upper
        return loss

# Calculate weights to compensate for separator imbalance
def __calculateWeightsFromTargets(targets):    
    ones = sum([sample[0] for sample in targets])
    total = sum([sample[1] for sample in targets])

    shareOnes = ones / total
    shareZeros = 1-shareOnes
    classWeights = torch.tensor([1.0/shareZeros, 1.0/shareOnes])
    classWeights = classWeights / classWeights.sum()
    return classWeights

def __calculateWeights(dataloader):
    targets_row = [(batch.targets.row_line.sum().item(), batch.targets.row_line.shape[0]) for batch in iter(dataloader.dataset)]
    targets_col = [(batch.targets.col_line.sum().item(), batch.targets.col_line.shape[0]) for batch in iter(dataloader.dataset)]
    classWeights = {'row': __calculateWeightsFromTargets(targets=targets_row),
                'col': __calculateWeightsFromTargets(targets=targets_col)}
    
    return classWeights

def getLossFunctions(dataloader):
    ''' Attach dataloader of your train data to calculate relative weights for separator indicators.
    These weights make it less likely the model will just always predict the most common class.'''
    classWeights = __calculateWeights(dataloader=dataloader)
    lossFunctions = {'row_line': WeightedBinaryCrossEntropyLoss(weights=classWeights['row']), 'row_separator_count': LogisticLoss(limit_upper=1/2),
                    'col_line': WeightedBinaryCrossEntropyLoss(weights=classWeights['col']), 'col_separator_count': LogisticLoss(limit_upper=1/2)} 
    return lossFunctions


# Loss function | Define weighted loss function
def calculateLoss(targets, preds, lossFunctions:dict, calculateCorrect=False, device='cuda'):   
    loss = torch.empty(size=(LOSS_ELEMENTS_COUNT,1), device=device, dtype=torch.float32)
    correct, maxCorrect = torch.empty(size=(LOSS_ELEMENTS_COUNT,1), device=device, dtype=torch.int64), torch.empty(size=(LOSS_ELEMENTS_COUNT,1), device=device, dtype=torch.int64)

    for idx_orientation, _ in enumerate(ORIENTATIONS):     # idx = 0; orientation = ORIENTATIONS[idx]
        # Line
        idx_line = 2*idx_orientation
        target_line = targets[idx_line].to(device)
        pred_line = preds[idx_orientation].to(device)

        loss_element = LOSS_ELEMENTS[idx_line]
        loss_fn = lossFunctions[loss_element]
        loss[idx_line] = loss_fn(pred_line, target_line)

        if calculateCorrect:
            correct[idx_line] = ((pred_line >= 0.5) == target_line).sum().item()
            maxCorrect[idx_line] = pred_line.numel()

        # Separator count
        idx_count = 2*idx_orientation + 1
        target_count = targets[idx_count].to(device)
        pred_separators = torch.as_tensor(pred_line >= 0.5, dtype=torch.int8).squeeze(-1)
        pred_count = torch.min(torch.tensor([torch.where(torch.diff(pred_separators) == 1)[0].numel(), torch.where(torch.diff(pred_separators) == -1)[0].numel()]))
        
        loss_element = LOSS_ELEMENTS[idx_count]
        loss_fn = lossFunctions[loss_element]
        loss[idx_count] = loss_fn(pred_count, target_count)

        if calculateCorrect:
            correct[idx_count] = pred_count.item()
            maxCorrect[idx_count] = target_count.item()
    
    if calculateCorrect:
        return loss, correct, maxCorrect
    else:
        return loss.sum()
