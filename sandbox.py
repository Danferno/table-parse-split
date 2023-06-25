def run():
    # Imports
    from pathlib import Path
    import torch    # type : ignore
        
    from model import TableLineModel
    from train import train_lineLevel, train_separatorLevel
    import evaluate
    from describe import describe_model
    from process import predict_and_process, generate_featuresAndTargets_separatorLevel
    
    # Constants
    # RUN_NAME = datetime.now().strftime("%Y_%m_%d__%H_%M")
    RUN_NAME = '8_lineScannerAdapted'
    PADDING = 40

    PATH_ROOT = Path(r"F:\ml-parsing-project\table-parse-split")
    TASKS = {'train_linemodel': False, 'eval_linemodel': False, 'train_separatormodel': True, 'eval_separatormodel':True, 'postprocess': False}
    BEST_RUN_LINE = Path(r"F:\ml-parsing-project\table-parse-split\models\7_separatorModelAdded_lineLevel\model_best.pt")
    BEST_RUN_SEPARATOR = Path(r"F:\ml-parsing-project\table-parse-split\models\7_separatorModelAdded_separatorLevel\model_best.pt")

    # Model parameters
    EPOCHS_LINELEVEL = 50
    MAX_LR_LINELEVEL = 0.08

    EPOCHS_SEPARATORLEVEL = 50
    MAX_LR_SEPARATORLEVEL = 0.2

    # Derived constants
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Paths
    path_data = PATH_ROOT / 'data' / 'real_narrow'
    path_model_lineLevel = PATH_ROOT / 'models' / f'{RUN_NAME}_lineLevel'
    path_model_separatorLevel = PATH_ROOT / 'models' / f'{RUN_NAME}_separatorLevel'
    path_words = PATH_ROOT / 'data' / 'words'
    path_pdfs = PATH_ROOT / 'data' / 'pdfs'

    # Define model
    if TASKS['train_linemodel']:
        # Describe model
        model_lineLevel = TableLineModel().to(DEVICE)
        describe_model(model_lineLevel)     

        # Train
        train_lineLevel(epochs=EPOCHS_LINELEVEL, max_lr=MAX_LR_LINELEVEL, 
                path_data_train=path_data / 'train', path_data_val=path_data / 'val',
                path_model=path_model_lineLevel, device=DEVICE, replace_dirs=True)

    if TASKS['eval_linemodel']:
        path_best_model_line = BEST_RUN_LINE if not TASKS['train_linemodel'] else path_model_lineLevel / 'model_best.pt'
        evaluate.evaluate_lineLevel(path_model_file=path_best_model_line, path_data=path_data / 'val', device=DEVICE, replace_dirs=True)

    if TASKS['train_separatormodel']:
        path_best_model_line = BEST_RUN_LINE if not TASKS['train_linemodel'] else path_model_lineLevel / 'model_best.pt'
        print(path_best_model_line)
        generate_featuresAndTargets_separatorLevel(path_best_model_line=path_best_model_line, path_data=path_data / 'all', path_words=path_words, replace_dirs=True, draw_images=True)
        train_separatorLevel(profile=False, epochs=EPOCHS_SEPARATORLEVEL, max_lr=MAX_LR_SEPARATORLEVEL, 
                path_data_train=path_data / 'train', path_data_val=path_data / 'val',
                path_model=path_model_separatorLevel, device=DEVICE, replace_dirs=True)
        
    if TASKS['eval_separatormodel']:
        path_best_model_separator = BEST_RUN_SEPARATOR if not TASKS['train_separatormodel'] else path_model_separatorLevel / 'model_best.pt'
        print(path_best_model_separator)
        evaluate.evaluate_separatorLevel(path_model_file=path_best_model_separator, path_data=path_data / 'val', device=DEVICE, replace_dirs=True)



    if TASKS['postprocess']:
        path_best_model_line = BEST_RUN_LINE if not TASKS['train'] else path_model_lineLevel / 'model_best.pt'
        predict_and_process(path_model_file=path_best_model_line, path_data=path_data / 'val', device=DEVICE, replace_dirs=True,
                    path_pdfs=path_pdfs, path_words=path_words, padding=PADDING)

if __name__ == '__main__':
    run()