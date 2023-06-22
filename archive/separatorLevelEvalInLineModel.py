# init
        # Separator evaluator
        # self.layer_separators_row = self.addLinearLayer(layerType=nn.Linear, in_features=self.feature_count_row_separators, out_features=1, activation=nn.ReLU, hidden_sizes=self.hidden_sizes_separators)
        # self.layer_separators_col = self.addLinearLayer(layerType=nn.Linear, in_features=self.feature_count_col_separators, out_features=1, activation=nn.ReLU, hidden_sizes=self.hidden_sizes_separators)

        # # Prediction scores
        # self.layer_pred_row = self.addLinearLayer_depth2(in_features=2, hidden_sizes=[2])
        # self.layer_pred_col = self.addLinearLayer_depth2(in_features=2, hidden_sizes=[2])

# forward
        # Generate separator-specific features
        # row_separators = self.preds_to_separators(predTensor=row_probs, threshold=self.truth_threshold)     # this will fail for batches (unequal length of separators)
        # if row_separators.numel():
        #     row_points = torch.cat([row_separators[:, 0, 0].unsqueeze(1), row_separators[:, :, 1]], dim=1)
        #     row_min_per_separator = segment_csr(src=features.row, indptr=row_points, reduce='min')
        #     row_max_per_separator = segment_csr(src=features.row, indptr=row_points, reduce='max')
        #     row_separator_features = torch.cat([row_min_per_separator, row_max_per_separator,
        #                                         row_inputs_global[:, :row_separators.shape[1], :]], dim=-1)
        #     row_separator_scores = self.layer_separators_row(row_separator_features)

        #     row_separators_scores_broadcast = row_preds
        #     start_indices = row_separators[:, :, 0]
        #     end_indices = row_separators[:, :, 1]
        #     for i in range(row_separators.shape[1]):
        #         row_separators_scores_broadcast[:, start_indices[:, i]:end_indices[:, i], :] = row_separator_scores[:, i, :]

        # else:
        #     row_separators_scores_broadcast = row_preds

        # col_separators = self.preds_to_separators(predTensor=col_probs, threshold=self.truth_threshold)
        # if col_separators.numel():               
        #     col_points = torch.cat([col_separators[:, 0, 0].unsqueeze(1), col_separators[:, :, 1]], dim=1)           
        #     col_min_per_separator = segment_csr(src=features.col, indptr=col_points, reduce='min')               
        #     col_max_per_separator = segment_csr(src=features.col, indptr=col_points, reduce='max')
        #     col_separator_features = torch.cat([col_min_per_separator, col_max_per_separator,
        #                                         col_inputs_global[:, :col_separators.shape[1], :]], dim=-1)
        #     col_separator_scores = self.layer_separators_col(col_separator_features)

        #     col_separators_scores_broadcast = col_preds
        #     start_indices = col_separators[:, :, 0]
        #     end_indices = col_separators[:, :, 1]
        #     for i in range(col_separators.shape[1]):
        #         col_separators_scores_broadcast[:, start_indices[:, i]:end_indices[:, i], :] = col_separator_scores[:, i, :]
        # else:
        #     col_separators_scores_broadcast = col_preds&

        # # Evaluate separators
        # row_prob_features = torch.cat([row_preds, row_separators_scores_broadcast], dim=-1)
        # col_prob_features = torch.cat([col_preds, col_separators_scores_broadcast], dim=-1)
        
        # row_prob_scores = self.layer_pred_row(row_prob_features)
        # col_prob_scores = self.layer_pred_col(col_prob_features)

        # # Turn into probabilities
        # row_probs = self.layer_logit(row_prob_scores)
        # col_probs = self.layer_logit(col_prob_scores)