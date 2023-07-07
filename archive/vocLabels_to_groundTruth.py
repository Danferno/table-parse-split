def vocLabels_to_groundTruth(path_labels, img, features, padding, adjust_labels_to_textboxes=False, add_edge_separators=False):
    # Parse textbox-information
    if adjust_labels_to_textboxes:
        row_between_textlines = features['row_between_textlines']
        col_wordsCrossed_relToMax = features['col_wordsCrossed_relToMax']
        # Row
        textline_boundaries_horizontal = np.diff(row_between_textlines.astype(np.int8), append=0)
        if textline_boundaries_horizontal.max() < 1:
            textline_boundaries_horizontal[-1] = 0
        textline_boundaries_horizontal = np.column_stack([np.where(textline_boundaries_horizontal == 1)[0], np.where(textline_boundaries_horizontal == -1)[0]])

        # Column
        textline_boundaries_vertical = np.diff((col_wordsCrossed_relToMax < 0.001).astype(np.int8), append=0)
        if textline_boundaries_vertical.max() < 1:
            textline_boundaries_vertical[-1] = 0
        else:
            textline_boundaries_vertical[0] = 1
        textline_boundaries_vertical = np.column_stack([np.where(textline_boundaries_vertical == 1)[0], np.where(textline_boundaries_vertical == -1)[0]])      

    # Parse xml
    root = etree.parse(path_labels)
    objectCount = len(root.findall('.//object'))
    if objectCount:
        rows = root.findall('object[name="row separator"]')
        cols = root.findall('object[name="column separator"]')
        spanners = root.findall('object[name="spanning cell interior"]')

        # Get separator locations
        row_separators = [(int(row.find('.//ymin').text), int(row.find('.//ymax').text)) for row in rows]
        row_separators = sorted(row_separators, key= lambda x: x[0])

        if (adjust_labels_to_textboxes) and (len(textline_boundaries_horizontal) > 0):
            row_separators = utils.adjust_initialBoundaries_to_betterBoundariesB(arrayInitial=row_separators, arrayBetter=textline_boundaries_horizontal)

        col_separators = [(int(col.find('.//xmin').text), int(col.find('.//xmax').text)) for col in cols]
        col_separators = sorted(col_separators, key= lambda x: x[0])
        if (adjust_labels_to_textboxes) and (len(textline_boundaries_vertical) > 0):
            col_separators = utils.adjust_initialBoundaries_to_betterBoundariesB(arrayInitial=col_separators, arrayBetter=textline_boundaries_vertical)

        # Optionally add edge borders (top, left, right , bottom)
        #   Excluding these can confuse the model, as they really look like borders
        if add_edge_separators:
            # Row
            if len(row_separators) > 0:
                first_nonwhite = np.where(features['row_avg'] < 1)[0][0]
                try:
                    first_withtext = np.where(features['row_wordsCrossed_count'] > 0)[0][0]
                except IndexError:
                    first_withtext = 0
                if (first_nonwhite < first_withtext) and (first_withtext - first_nonwhite > 2) and (first_withtext < row_separators[0][0]):
                    top_separator = (first_nonwhite, first_withtext-1)
                    row_separators.insert(0, top_separator)

                last_nonwhite = np.where(features['row_avg'] < 1)[0][-1]
                try:
                    last_withtext = np.where(features['row_wordsCrossed_count'] > 0)[0][-1]
                except IndexError:
                    last_withtext = features['row_wordsCrossed_count'].size
                if (last_nonwhite > last_withtext) and (last_nonwhite - last_withtext > 2) and (last_withtext > row_separators[-1][0]):
                    bot_separator = (last_withtext+1, last_nonwhite)
                    row_separators.append(bot_separator)

            # Column
            if len(col_separators) > 0:
                first_nonwhite = np.where(features['col_avg'] < 1)[0][0]
                try:
                    first_withtext = np.where(features['col_wordsCrossed_count'] > 0)[0][0]
                except:
                    first_withtext = 0
                if (first_nonwhite < first_withtext) and (first_withtext - first_nonwhite > 2) and (first_withtext < col_separators[0][0]):
                    left_separator = (first_nonwhite, first_withtext-1)
                    col_separators.insert(0, left_separator)

                last_nonwhite = np.where(features['col_avg'] < 1)[0][-1]
                try:
                    last_withtext = np.where(features['col_wordsCrossed_count'] > 0)[0][-1]
                except:
                    last_withtext = features['col_wordsCrossed_count'].size
                if (last_nonwhite > last_withtext) and (last_nonwhite - last_withtext > 2) and (last_withtext > col_separators[-1][0]):
                    right_separator = (last_withtext+1, last_nonwhite)
                    col_separators.append(right_separator)

        # Create ground truth arrays
        gt_row = np.zeros(shape=img.shape[0], dtype=np.uint8)
        for separator in row_separators:        # separator = row_separators[0]
            gt_row[separator[0]:separator[1]+1] = 1
        
        gt_col = np.zeros(shape=img.shape[1], dtype=np.uint8)
        for separator in col_separators:        # separator = row_separators[0]
            gt_col[separator[0]:separator[1]+1] = 1
    else:
        gt_row = np.zeros(shape=img.shape[0], dtype=np.uint8)
        gt_col = np.zeros(shape=img.shape[1], dtype=np.uint8)
    
    # Adjust to textboxes
    gt = {}
    gt['row'] = gt_row
    gt['col'] = gt_col
    return gt