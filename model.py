# Imports
from collections import namedtuple, OrderedDict
from torch import nn
import torch
from torch_scatter import segment_csr


# Constants
ORIENTATIONS = ['row', 'col']
FEATURE_TYPES = ORIENTATIONS + ['image', 'row_global', 'col_global']

COMMON_VARIABLES = ['{}_avg', '{}_absDiff', '{}_spell_mean', '{}_spell_sd', '{}_wordsCrossed_count', '{}_wordsCrossed_relToMax']
COMMON_GLOBAL_VARIABLES = ['global_{}Avg_p0', 'global_{}Avg_p5', 'global_{}Avg_p10']

ROW_VARIABLES = ['row_between_textlines', 'row_between_textlines_like_rowstart']
COL_VARIABLES = ['col_nearest_right_is_startlike_share']

LOSS_CHARACTERISTICS = ['line', 'separator_count']
LOSS_ELEMENTS = [f'{orientation}_{characteristic}' for orientation in ORIENTATIONS for characteristic in LOSS_CHARACTERISTICS]
LOSS_ELEMENTS_COUNT = len(LOSS_ELEMENTS)


# Named tuples
Sample = namedtuple('sample', ['features', 'targets', 'meta'])
Meta = namedtuple('meta', ['path_image', 'table_coords', 'dpi_pdf', 'dpi_model', 'dpi_words', 'name_stem', 'padding_model', 'image_angle'])
Features = namedtuple('features', FEATURE_TYPES)
Targets = namedtuple('target', LOSS_ELEMENTS)
Output = namedtuple('output', ORIENTATIONS)


# Models
class TabliterModel(nn.Module):
    def __init__(self, 
                    image_convolution_parameters={'channels_1': 2, 'size_1': (4, 4), 'pool_count_1': 4},
                    preds_convolution_parameters={'channels_1': 3, 'channels_2': 3, 'size_1': (4), 'size_2': (10)},
                    linescanner_parameters={'size': 10, 'channels': 2 , 'keepTopX': 5},
                    lag_lead_structure = [-4, -2, -1, 1, 2, 4],
                    hidden_sizes_features=[48, 16], hidden_sizes_separators=[24, 8],
                    info_variableCount={'common_orientationSpecific': 6, 'common_global': 3, 'row_specific': 2, 'col_specific': 1},
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
        self.layer_ls_row = self.addLineScanner(orientation='rows')
        self.layer_ls_col = self.addLineScanner(orientation='cols')

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

        # Separator evaluator
        self.layer_separators_row = self.addLinearLayer(layerType=nn.Linear, in_features=self.feature_count_row_separators, out_features=1, activation=nn.ReLU, hidden_sizes=self.hidden_sizes_separators)
        self.layer_separators_col = self.addLinearLayer(layerType=nn.Linear, in_features=self.feature_count_col_separators, out_features=1, activation=nn.ReLU, hidden_sizes=self.hidden_sizes_separators)

        # Prediction scores
        self.layer_pred_row = self.addLinearLayer_depth2(in_features=2, hidden_sizes=[2])
        self.layer_pred_col = self.addLinearLayer_depth2(in_features=2, hidden_sizes=[2])

        # Logit model
        self.layer_logit = nn.Sigmoid()
    
    def addLineScanner(self, orientation):
        kernel = (1, self.linescanner_parameters['size']) if orientation == 'rows' else (self.linescanner_parameters['size'], 1)
        max_transformer = (None, 1) if orientation == 'rows' else (None, 1)
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
            
        # Image
        # Image | Convolutional layers based on image
        img_intermediate_values = self.layer_conv_img(features.image)
        row_conv_values = self.layer_conv_avg_row(img_intermediate_values).view(1, -1, self.image_convolution_parameters['channels_final']*self.image_convolution_parameters['pool_count_final'])
        col_conv_values = self.layer_conv_avg_col(img_intermediate_values).view(1, -1, self.image_convolution_parameters['channels_final']*self.image_convolution_parameters['pool_count_final'])

        # Row
        # Row | Global features
        row_inputs_global = features.row_global

        # Row | Linescanner
        row_linescanner_values = self.layer_ls_row(features.image)
        row_linescanner_top5 = torch.topk(row_linescanner_values, k=self.linescanner_parameters['keepTopX'], dim=2, sorted=True).values.view(1, -1, self.linescanner_parameters['keepTopX']*self.linescanner_parameters['channels']).broadcast_to((-1, features.image.shape[2], -1))
        
        # Row | Gather features
        row_inputs = torch.cat([features.row, row_conv_values], dim=-1)
        row_inputs_lag_leads = torch.cat([torch.roll(row_inputs, shifts=shift, dims=1) for shift in self.lag_lead_structure], dim=-1)     # dangerous if no padding applied !!!          
        row_inputs_complete = torch.cat([row_inputs, row_inputs_lag_leads, row_inputs_global, row_linescanner_top5], dim=-1)
        
        # Row | Linear prediction
        row_direct_preds = self.layer_linear_row(row_inputs_complete)

        # Row | Convolved prediction
        row_conv_preds = self.layer_conv_preds_row(row_direct_preds.view(1, 1, -1)).view(1, -1, self.preds_convolution_parameters['channels_1'])
        row_preds = self.layer_conv_preds_fc_row(torch.cat([row_direct_preds, row_conv_preds], dim=-1))

        # Col
        # Col | Global features
        col_inputs_global = features.col_global

        # Col | Linescanner
        col_linescanner_values = self.layer_ls_col(features.image)
        col_linescanner_top5 = torch.topk(col_linescanner_values, k=self.linescanner_parameters['keepTopX'], dim=2, sorted=True).values.view(1, -1, self.linescanner_parameters['keepTopX']*self.linescanner_parameters['channels']).broadcast_to((-1, features.image.shape[3], -1))

        # Col | Gather features
        col_inputs = torch.cat([features.col, col_conv_values], dim=-1)
        col_inputs_lag_leads = torch.cat([torch.roll(col_inputs, shifts=shift, dims=1) for shift in self.lag_lead_structure], dim=-1)     # dangerous if no padding applied !!!
        col_inputs_complete = torch.cat([col_inputs, col_inputs_lag_leads, col_inputs_global, col_linescanner_top5], dim=-1)

        # Col | Linear prediction
        col_direct_preds = self.layer_linear_col(col_inputs_complete)

        # Col | Convolved prediction
        col_conv_preds = self.layer_conv_preds_col(col_direct_preds.view(1, 1, -1)).view(1, -1, self.preds_convolution_parameters['channels_1'])
        col_preds = self.layer_conv_preds_fc_col(torch.cat([col_direct_preds, col_conv_preds], dim=-1))
        

        # Turn into probabilities
        row_probs = self.layer_logit(row_preds)
        col_probs = self.layer_logit(col_preds)

        # Generate separator-specific features
        row_separators = self.preds_to_separators(predTensor=row_probs, threshold=self.truth_threshold)     # this will fail for batches (unequal length of separators)
        if row_separators.numel():
            row_points = torch.cat([row_separators[:, 0, 0].unsqueeze(1), row_separators[:, :, 1]], dim=1)
            row_min_per_separator = segment_csr(src=features.row, indptr=row_points, reduce='min')
            row_max_per_separator = segment_csr(src=features.row, indptr=row_points, reduce='max')
            row_separator_features = torch.cat([row_min_per_separator, row_max_per_separator,
                                                row_inputs_global[:, :row_separators.shape[1], :]], dim=-1)
            row_separator_scores = self.layer_separators_row(row_separator_features)

            row_separators_scores_broadcast = row_preds
            start_indices = row_separators[:, :, 0]
            end_indices = row_separators[:, :, 1]
            for i in range(row_separators.shape[1]):
                row_separators_scores_broadcast[:, start_indices[:, i]:end_indices[:, i], :] = row_separator_scores[:, i, :]

        else:
            row_separators_scores_broadcast = row_preds

        col_separators = self.preds_to_separators(predTensor=col_probs, threshold=self.truth_threshold)
        if col_separators.numel():               
            col_points = torch.cat([col_separators[:, 0, 0].unsqueeze(1), col_separators[:, :, 1]], dim=1)           
            col_min_per_separator = segment_csr(src=features.col, indptr=col_points, reduce='min')               
            col_max_per_separator = segment_csr(src=features.col, indptr=col_points, reduce='max')
            col_separator_features = torch.cat([col_min_per_separator, col_max_per_separator,
                                                col_inputs_global[:, :col_separators.shape[1], :]], dim=-1)
            col_separator_scores = self.layer_separators_col(col_separator_features)

            col_separators_scores_broadcast = col_preds
            start_indices = col_separators[:, :, 0]
            end_indices = col_separators[:, :, 1]
            for i in range(col_separators.shape[1]):
                col_separators_scores_broadcast[:, start_indices[:, i]:end_indices[:, i], :] = col_separator_scores[:, i, :]
        else:
            col_separators_scores_broadcast = col_preds

        # Evaluate separators
        row_prob_features = torch.cat([row_preds, row_separators_scores_broadcast], dim=-1)
        col_prob_features = torch.cat([col_preds, col_separators_scores_broadcast], dim=-1)
        
        row_prob_scores = self.layer_pred_row(row_prob_features)
        col_prob_scores = self.layer_pred_col(col_prob_features)

        # Turn into probabilities
        row_probs = self.layer_logit(row_prob_scores)
        col_probs = self.layer_logit(col_prob_scores)

        # Output
        return Output(row=row_probs, col=col_probs)
