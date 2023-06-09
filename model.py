# Imports
from collections import namedtuple, OrderedDict
from torch import nn
import torch.nn.functional as F
import torch
from einops.layers.torch import Rearrange
# from torch_scatter import segment_csr

# Constants
# Constants | Common
ORIENTATIONS = ['row', 'col']

# Constants | Line Level
FEATURE_TYPES = ORIENTATIONS + ['image', 'row_global', 'col_global']

COMMON_VARIABLES = ['{}_avg', '{}_absDiff', '{}_spell_mean', '{}_spell_sd', '{}_wordsCrossed_count', '{}_wordsCrossed_relToMax', '{}_in_textrectangle']
COMMON_GLOBAL_VARIABLES = ['global_{}Avg_p0', 'global_{}Avg_p5', 'global_{}Avg_p10']

ROW_VARIABLES = ['row_between_textlines', 'row_between_textlines_like_rowstart', 'row_between_textlines_capital_ratio']
COL_VARIABLES = ['col_nearest_right_is_startlike_share', 'col_ratio_p60p40_textdistance_left', 'col_ratio_p60p40_textdistance_right']

LOSS_CHARACTERISTICS = ['line', 'separator_count']
LOSS_ELEMENTS_LINELEVEL = [f'{orientation}_{characteristic}' for orientation in ORIENTATIONS for characteristic in LOSS_CHARACTERISTICS]
LOSS_ELEMENTS_LINELEVEL_COUNT = len(LOSS_ELEMENTS_LINELEVEL)

# Constants | Separator Level
FEATURE_TYPES_SEPARATORLEVEL = ORIENTATIONS + ['image'] + ['proposedSeparators_row', 'proposedSeparators_col']
COMMON_VARIABLES_SEPARATORLEVEL = ['text_between_separators_{}']
LOSS_ELEMENTS_SEPARATORLEVEL = [f'{orientation}_separator' for orientation in ORIENTATIONS]
LOSS_ELEMENTS_SEPARATORLEVEL_COUNT = len(LOSS_ELEMENTS_SEPARATORLEVEL)


# Named tuples
# Named tuples | Common
Output = namedtuple('output', ORIENTATIONS)

# Named tuples | Line Level
Sample = namedtuple('sample', ['features', 'targets', 'meta'])
Meta = namedtuple('meta', ['path_image', 'table_coords', 'dpi_pdf', 'dpi_model', 'dpi_words', 'name_stem', 'padding_model', 'image_angle', 'size_image'])
Features = namedtuple('features', FEATURE_TYPES)
Targets = namedtuple('target', LOSS_ELEMENTS_LINELEVEL)

# Named tuples | Separator Level
SeparatorTargets = namedtuple('target_separatorLevel', ORIENTATIONS)
SeparatorFeatures = namedtuple('feature_separatorLevel', FEATURE_TYPES_SEPARATORLEVEL)


# Models
class TableLineModel(nn.Module):
    def __init__(self, 
                    image_convolution_parameters={'channels_1': 32, 'size_1': (4, 4), 'pool_count_1': 4},
                    preds_convolution_parameters={'channels_1': 6, 'channels_2': 6, 'size_1': (4), 'size_2': (10)},
                    linescanner_parameters={'size': 10, 'channels': 2 , 'keepTopX': 5},
                    lag_lead_structure = [-4,-3, -2, -1, 1, 2, 3, 4],
                    hidden_sizes_features=[48, 16], hidden_sizes_separators=[24, 8],
                    info_variableCount={'common_orientationSpecific': 7+1, 'common_global': 3, 'row_specific': 3, 'col_specific': 3},
                    truth_threshold=0.8, device='cuda'):
        super().__init__()
        # Parameters
        self.device = device
        self.hidden_sizes_features = hidden_sizes_features
        self.hidden_sizes_separators = hidden_sizes_separators
        self.truth_threshold = truth_threshold
        self.image_convolution_parameters = image_convolution_parameters
        self.image_convolution_parameters['channels_final'] = self.image_convolution_parameters['channels_1']
        self.image_convolution_parameters['pool_count_final'] = self.image_convolution_parameters['pool_count_1']
        self.preds_convolution_parameters = preds_convolution_parameters
        self.linescanner_parameters = linescanner_parameters
        self.lag_lead_structure = lag_lead_structure
        self.info_variableCount = info_variableCount
        self.info_variableCount['image_convolution_channels'] = self.image_convolution_parameters['channels_final']
        self.info_variableCount['image_convolution_pool_count'] = self.image_convolution_parameters['pool_count_final']
        self.info_variableCount['lag_leads_count'] = len(self.lag_lead_structure) + 1
        self.info_variableCount['linescanner_channels'] = self.linescanner_parameters['channels']
        self.info_variableCount['linescanner_keepTopX'] = self.linescanner_parameters['keepTopX']

        self.feature_count_row = ((self.info_variableCount['common_orientationSpecific'] + self.info_variableCount['row_specific']
                                        + self.info_variableCount['image_convolution_channels'] * self.info_variableCount['image_convolution_pool_count']) 
                                    * (self.info_variableCount['lag_leads_count']) 
                                   + self.info_variableCount['common_global'] 
                                   + (self.info_variableCount['linescanner_channels'] * self.info_variableCount['linescanner_keepTopX']))
        self.feature_count_col = ((self.info_variableCount['common_orientationSpecific'] + self.info_variableCount['col_specific']
                                        + self.info_variableCount['image_convolution_channels']*self.info_variableCount['image_convolution_pool_count']) 
                                    * (self.info_variableCount['lag_leads_count']) 
                                    + self.info_variableCount['common_global'] 
                                    + (self.info_variableCount['linescanner_channels'] * self.info_variableCount['linescanner_keepTopX']))
        self.feature_count_row_separators = ((self.info_variableCount['common_orientationSpecific'] + self.info_variableCount['row_specific'])*2 
            + self.info_variableCount['common_global'])
        self.feature_count_col_separators = ((self.info_variableCount['common_orientationSpecific'] + self.info_variableCount['col_specific'])*2 
            + self.info_variableCount['common_global'])
        

        # Line scanner
        self.layer_ls_row = self.addLineScanner(orientation='row')
        self.layer_ls_col = self.addLineScanner(orientation='col')

        # Convolution on image
        self.layer_conv_img = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=self.image_convolution_parameters['channels_1'], kernel_size=self.image_convolution_parameters['size_1'], padding='same', padding_mode='replicate', bias=False)),
            ('norm_conv1', nn.BatchNorm2d(self.image_convolution_parameters['channels_1'])),
            ('relu1', nn.ReLU()),
            # ('conv2', nn.Conv2d(in_channels=5, out_channels=2, kernel_size=CONV_SEQUENCE_KERNEL, padding='same', padding_mode='replicate')),
        ]))
        self.layer_conv_avg_row = nn.AdaptiveAvgPool2d(output_size=(None, self.image_convolution_parameters['pool_count_1']))
        self.layer_conv_avg_col = nn.AdaptiveAvgPool2d(output_size=(self.image_convolution_parameters['pool_count_1'], None))

        # Feature+Conv Neural Net
        self.layer_linear_row = self.addLinearLayer(layerType=nn.Linear, in_features=self.feature_count_row, out_features=1, activation=nn.ReLU, hidden_sizes=self.hidden_sizes_features)
        self.layer_linear_col = self.addLinearLayer(layerType=nn.Linear, in_features=self.feature_count_col, out_features=1, activation=nn.ReLU, hidden_sizes=self.hidden_sizes_features)

        # Convolution on predictions
        self.layer_conv_preds_row = self.addPredConvLayer()
        self.layer_conv_preds_col = self.addPredConvLayer()
        self.layer_conv_preds_fc_row = self.addLinearLayer_depth2(in_features=self.preds_convolution_parameters['channels_2']+1)
        self.layer_conv_preds_fc_col = self.addLinearLayer_depth2(in_features=self.preds_convolution_parameters['channels_2']+1)



        # Logit model
        self.layer_logit = nn.Sigmoid()
    
    def addLineScanner(self, orientation):
        kernel = (1, self.linescanner_parameters['size']) if orientation == 'row' else (self.linescanner_parameters['size'], 1)
        max_transformer = (None, 1) if orientation == 'row' else (1, None)
        sequence = nn.Sequential(OrderedDict([
            (f'conv1', nn.Conv2d(in_channels=1, out_channels=self.linescanner_parameters['channels'], kernel_size=kernel, stride=kernel, padding='valid', bias=False)),
            (f'norm', nn.BatchNorm2d(self.linescanner_parameters['channels'])),
            (f'relu', nn.ReLU()),
            (f'pool', nn.AdaptiveMaxPool2d(max_transformer))
        ]))
        return sequence
        
    def addLinearLayer(self, layerType:nn.Module, in_features:int, out_features:int, hidden_sizes, activation=nn.PReLU):
        sequence = nn.Sequential(OrderedDict([
            (f'lin1_from{in_features}_to{hidden_sizes[0]}', layerType(in_features=in_features, out_features=hidden_sizes[0])),
            (f'relu_1', activation()),
            # ('norm_lin1', nn.BatchNorm1d(self.hidden_sizes[0])),
            (f'lin2_from{hidden_sizes[0]}_to{hidden_sizes[1]}', layerType(in_features=hidden_sizes[0], out_features=hidden_sizes[1])),
            (f'relu_2', activation()),
            # ('norm_lin2', nn.BatchNorm1d(self.hidden_sizes[1])),
            (f'lin3_from{hidden_sizes[1]}_to{out_features}', layerType(in_features=hidden_sizes[1], out_features=out_features))
        ]))
        return sequence
        
    def addPredConvLayer(self):
        sequence = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(in_channels=1, out_channels=self.preds_convolution_parameters['channels_1'], kernel_size=self.preds_convolution_parameters['size_1'], padding='same', padding_mode='replicate', bias=False)),
            ('conv1_norm', nn.BatchNorm1d(self.preds_convolution_parameters['channels_1'])),
            ('conv1_relu', nn.ReLU()),
            ('conv2', nn.Conv1d(in_channels=self.preds_convolution_parameters['channels_1'], out_channels=self.preds_convolution_parameters['channels_2'], kernel_size=self.preds_convolution_parameters['size_2'], padding='same', padding_mode='replicate', bias=False)),
            ('conv2_norm', nn.BatchNorm1d(self.preds_convolution_parameters['channels_2'])),
            ('conv2_relu', nn.ReLU())
        ]))
        return sequence
    def addLinearLayer_depth2(self, in_features:int, hidden_sizes=[6]):
        sequence = nn.Sequential(OrderedDict([
            (f'lin1_from{in_features}_to{hidden_sizes[0]}', nn.Linear(in_features=in_features, out_features=hidden_sizes[0])),
            (f'relu_1', nn.ReLU()),
            (f'lin2_from{hidden_sizes[0]}_to1', nn.Linear(in_features=hidden_sizes[0], out_features=1))
        ]))
        return sequence
    
    def preds_to_separators(self, predTensor, threshold):
        # Apply threshold
        is_separator = torch.cat([torch.full(size=(1,1), fill_value=0, device=self.device, dtype=torch.int8), torch.as_tensor(predTensor >= threshold, dtype=torch.int8).squeeze(-1)[:, 1:-1], torch.full(size=(1,1), fill_value=0, device=self.device, dtype=torch.int8)], dim=-1)
        diff_in_separator_modus = torch.diff(is_separator, dim=-1)
        separators_start = torch.where(diff_in_separator_modus == 1)
        separators_end = torch.where(diff_in_separator_modus == -1)
        separators = torch.stack([separators_start[1], separators_end[1]], axis=1).unsqueeze(0)     # not sure if this handles batches well

        return separators

    
    def forward(self, features):
        # Convert tuple to namedtuple
        if type(features) == type((0,)):
            features = Features(**{field: features[i] for i, field in enumerate(Features._fields)})
        
        # Load features to GPU
        features = Features(**{field: features[i].to(self.device) for i, field in enumerate(Features._fields)})

        # Get batch_size
        batch_size = features.image.shape[0]
            
        # Image
        # Image | Convolutional layers based on image
        img_intermediate_values = self.layer_conv_img(features.image)
        row_conv_values = self.layer_conv_avg_row(img_intermediate_values).view(batch_size, -1, self.image_convolution_parameters['channels_final']*self.image_convolution_parameters['pool_count_final'])
        col_conv_values = self.layer_conv_avg_col(img_intermediate_values).view(batch_size, -1, self.image_convolution_parameters['channels_final']*self.image_convolution_parameters['pool_count_final'])

        # Row
        # Row | Global features
        row_inputs_global = features.row_global

        # Row | Global Linescanner
        row_linescanner_values = self.layer_ls_row(features.image).squeeze(-1)
        row_linescanner_top5 = torch.topk(row_linescanner_values, k=self.linescanner_parameters['keepTopX'], dim=2, sorted=True).values.view(batch_size, -1, self.linescanner_parameters['keepTopX']*self.linescanner_parameters['channels']).broadcast_to((-1, features.image.shape[2], -1))
        
        # Row | Gather features
        row_inputs = torch.cat([features.row, row_conv_values], dim=-1)
        row_inputs_lag_leads = torch.cat([torch.roll(row_inputs, shifts=shift, dims=1) for shift in self.lag_lead_structure], dim=-1)     # dangerous if no padding applied !!!          
        row_inputs_complete = torch.cat([row_inputs, row_inputs_lag_leads, row_inputs_global, row_linescanner_top5], dim=-1)
        
        # Row | Linear prediction
        row_direct_preds = self.layer_linear_row(row_inputs_complete)

        # Row | Convolved prediction
        row_conv_preds = self.layer_conv_preds_row(row_direct_preds.view(batch_size, 1, -1)).view(batch_size, -1, self.preds_convolution_parameters['channels_1'])
        row_preds = self.layer_conv_preds_fc_row(torch.cat([row_direct_preds, row_conv_preds], dim=-1))

        # Col
        # Col | Global features
        col_inputs_global = features.col_global

        # Col | Global Linescanner
        col_linescanner_values = self.layer_ls_col(features.image).squeeze(-2)
        col_linescanner_top5 = torch.topk(col_linescanner_values, k=self.linescanner_parameters['keepTopX'], dim=2, sorted=True).values.view(batch_size, -1, self.linescanner_parameters['keepTopX']*self.linescanner_parameters['channels']).broadcast_to((-1, features.image.shape[3], -1))

        # Col | Gather features
        col_inputs = torch.cat([features.col, col_conv_values], dim=-1)
        col_inputs_lag_leads = torch.cat([torch.roll(col_inputs, shifts=shift, dims=1) for shift in self.lag_lead_structure], dim=-1)     # dangerous if no padding applied !!!
        col_inputs_complete = torch.cat([col_inputs, col_inputs_lag_leads, col_inputs_global, col_linescanner_top5], dim=-1)

        # Col | Linear prediction
        col_direct_preds = self.layer_linear_col(col_inputs_complete)

        # Col | Convolved prediction
        col_conv_preds = self.layer_conv_preds_col(col_direct_preds.view(batch_size, 1, -1)).view(batch_size, -1, self.preds_convolution_parameters['channels_1'])
        col_preds = self.layer_conv_preds_fc_col(torch.cat([col_direct_preds, col_conv_preds], dim=-1))
        

        # Turn into probabilities
        row_probs = self.layer_logit(row_preds)
        col_probs = self.layer_logit(col_preds)

        # Output
        return Output(row=row_probs, col=col_probs)

class TableSeparatorModel(nn.Module):
    def __init__(self,
                    linescanner_parameters={'size': 10, 'channels': 32, 'keepTopX_local': 8, 'keepTopX_global': 8},
                    areascanner_parameters={'size': [8, 8], 'channels': 32, 'keepTopX_local': 8},
                    fc_parameters={'hidden_sizes': [16, 8]},
                    info_variableCount={'common_orientationSpecific': 2*7+1, 'common_global': 3},
                    device='cuda'):
        super().__init__()
        # Parameters
        self.device = device
        self.linescanner_parameters = linescanner_parameters
        self.areascanner_parameters = areascanner_parameters
        self.fc_parameters = fc_parameters
        self.fc_parameters['in_features'] = info_variableCount['common_orientationSpecific'] + info_variableCount['common_global'] + self.linescanner_parameters['keepTopX_local'] + self.linescanner_parameters['keepTopX_global'] + self.areascanner_parameters['keepTopX_local']*2

        # Layers
        # Layers | Area scanner
        self.layer_areascanner = self.addAreaScanner(params=self.areascanner_parameters)

        # Layers | Line scanner
        self.layer_ls_row = self.addLineScanner(orientation='row', params=self.linescanner_parameters)
        self.layer_ls_col = self.addLineScanner(orientation='col', params=self.linescanner_parameters)

        # Layers | Rearrangers
        self.rearrange_row = Rearrange(pattern='b c h w -> b h (c w)')
        self.rearrange_col = Rearrange(pattern='b c h w -> b w (c h)')
        self.rearrange_sep = Rearrange(pattern='b s l o -> b s (l o)')      # batch sep lines options > batch sep lines-options
        self.rearrange_ls = Rearrange(pattern='b c l -> b l c')


        # Layers | Fully connected
        self.layer_fc_row = self.addLinearLayer_depth3(params=fc_parameters)
        self.layer_fc_col = self.addLinearLayer_depth3(params=fc_parameters)

        # Logit Model
        self.layer_logit = nn.Sigmoid()

    def addAreaScanner(self, params):
        kernel = params['size']
        sequence= nn.Sequential(OrderedDict([
            (f'conv1', nn.Conv2d(in_channels=1, out_channels=params['channels'], kernel_size=kernel, padding='same', padding_mode='replicate', bias=False)),
            (f'relu', nn.ReLU()),
            (f'norm', nn.BatchNorm2d(params['channels'])),
        ]))
        return sequence

    def addLineScanner(self, orientation, params):
        kernel = (1, params['size']) if orientation == 'row' else (params['size'], 1)
        max_transformer = (None, 1)  if orientation == 'row' else (1, None)
        sequence = nn.Sequential(OrderedDict([
            (f'conv1', nn.Conv2d(in_channels=1, out_channels=params['channels'], kernel_size=kernel, stride=kernel, padding='valid', bias=False)),
            (f'relu', nn.ReLU()),
            (f'norm', nn.BatchNorm2d(params['channels'])),
            (f'pool', nn.AdaptiveMaxPool2d(max_transformer))
        ]))
        return sequence
    
    def addLinearLayer_depth3(self, params:dict, activation=nn.ReLU):
        hidden_sizes = params['hidden_sizes']
        out_features = params.get('out_features') or 1
        in_features = params['in_features']
        sequence = nn.Sequential(OrderedDict([
            (f'lin1_from{in_features}_to{hidden_sizes[0]}', nn.Linear(in_features=in_features, out_features=hidden_sizes[0])),
            (f'relu_1', activation()),
            (f'lin2_from{hidden_sizes[0]}_to{hidden_sizes[1]}', nn.Linear(in_features=hidden_sizes[0], out_features=hidden_sizes[1])),
            (f'relu_2', activation()),
            (f'lin3_from{hidden_sizes[1]}_to{out_features}', nn.Linear(in_features=hidden_sizes[1], out_features=out_features))
        ]))
        return sequence
    
    def forward(self, features):
        # Convert tuple to namedtuple
        if type(features) == type((0,)):
            features = SeparatorFeatures(**{field: features[i] for i, field in enumerate(SeparatorFeatures._fields)})
        
        # Load features to GPU
        features = SeparatorFeatures(**{field: features[i].to(self.device) for i, field in enumerate(SeparatorFeatures._fields)})
        batch_size = 1

        # Scanners
        area_scanner_values = self.layer_areascanner(features.image)
        row_linescanner_values = self.rearrange_ls(self.layer_ls_row(features.image).squeeze(-1))
        col_linescanner_values = self.rearrange_ls(self.layer_ls_col(features.image).squeeze(-2))

        # Row
        # Row | Info
        row_inputs_precalculated = features.row
        count_separators_row = features.proposedSeparators_row.shape[1]

        # Row | TopX of area scanner
        area_scanner_values_row = self.rearrange_row(area_scanner_values)
        area_scanner_topX_byRow = torch.topk(area_scanner_values_row, k=self.areascanner_parameters['keepTopX_local'], dim=-1).values
        row_areascanner_before = torch.topk(
                                    self.rearrange_sep(
                                        torch.stack(
                                            [area_scanner_topX_byRow[:, start-self.areascanner_parameters['size'][0]:start, :] for start, _ in features.proposedSeparators_row.squeeze(0)]
                                        , dim=1)
                                    ),
                                 k=self.areascanner_parameters['keepTopX_local'], dim=-1).values
        row_areascanner_after = torch.topk(
                                    self.rearrange_sep(
                                        torch.stack(
                                            [area_scanner_topX_byRow[:, end+1:end+self.areascanner_parameters['size'][0]+1, :] for _, end in features.proposedSeparators_row.squeeze(0)]
                                        , dim=1)
                                    ),
                                k=self.areascanner_parameters['keepTopX_local'], dim=-1).values

        # Row | TopX of line scanner
        longestSeparator = torch.max(torch.diff(features.proposedSeparators_row.squeeze(0))+1)
        row_linescanner_topX_local = torch.topk(
                                        self.rearrange_sep(
                                            torch.stack([F.pad(row_linescanner_values[:, start:end+1, :], pad=(0, 0, 0, longestSeparator-(end+1-start), 0, 0), value=float('-inf')) for start, end in features.proposedSeparators_row.squeeze(0)], dim=1)
                                        ),
                                    k=self.linescanner_parameters['keepTopX_local'], dim=-1).values
        row_image_features = torch.cat([row_areascanner_before, row_areascanner_after, row_linescanner_topX_local], dim=-1)

        # Row | Global linescanner        
        row_linescanner_topX_global = torch.topk(row_linescanner_values.reshape(batch_size, -1), k=self.linescanner_parameters['keepTopX_global'], sorted=True).values.broadcast_to((batch_size, count_separators_row, -1))

        # Row | Fully connected
        row_inputs = torch.cat([row_inputs_precalculated, row_image_features, row_linescanner_topX_global], dim=2)
        row_preds = self.layer_fc_row(row_inputs)


        # Col
        # Col | Info
        col_inputs = features.col
        count_separators_col = features.proposedSeparators_col.shape[1]

        # Col | TopX of area scanner
        area_scanner_values_col = self.rearrange_col(area_scanner_values)
        area_scanner_topX_byCol = torch.topk(area_scanner_values_col, k=self.areascanner_parameters['keepTopX_local'], dim=-1).values
        col_areascanner_before = torch.topk(
                                    self.rearrange_sep(
                                        torch.stack(
                                            [area_scanner_topX_byCol[:, start-self.areascanner_parameters['size'][0]:start, :] for start, _ in features.proposedSeparators_col.squeeze(0)]
                                        , dim=1)
                                    ),
                                 k=self.areascanner_parameters['keepTopX_local'], dim=-1).values
        col_areascanner_after = torch.topk(
                                    self.rearrange_sep(
                                        torch.stack(
                                            [area_scanner_topX_byCol[:, end+1:end+self.areascanner_parameters['size'][0]+1, :] for _, end in features.proposedSeparators_col.squeeze(0)]
                                        , dim=1)
                                    ),
                                k=self.areascanner_parameters['keepTopX_local'], dim=-1).values

        # Col | TopX of line scanner
        longestSeparator = torch.max(torch.diff(features.proposedSeparators_col.squeeze(0))+1)
        col_linescanner_topX_local = torch.topk(
                                        self.rearrange_sep(
                                            torch.stack([F.pad(col_linescanner_values[:, start:end+1, :], pad=(0, 0, 0, longestSeparator-(end+1-start), 0, 0), value=float('-inf')) for start, end in features.proposedSeparators_col.squeeze(0)], dim=1)
                                        ),
                                    k=self.linescanner_parameters['keepTopX_local'], dim=-1).values
        col_image_features = torch.cat([col_areascanner_before, col_areascanner_after, col_linescanner_topX_local], dim=-1)

        # Col | Global linescanner
        col_linescanner_topX_global = torch.topk(col_linescanner_values.reshape(batch_size, -1), k=self.linescanner_parameters['keepTopX_global'], sorted=True).values.broadcast_to((batch_size, count_separators_col, -1))

        # Col | Fully connected
        col_inputs = torch.cat([col_inputs, col_image_features, col_linescanner_topX_global], dim=2)
        col_preds = self.layer_fc_col(col_inputs)

        # Turn into probabilities
        row_probs = self.layer_logit(row_preds)
        col_probs = self.layer_logit(col_preds)

        # Output
        return Output(row=row_probs, col=col_probs)