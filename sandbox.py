def run():
    # Imports
    from pathlib import Path
    import torch    # type : ignore
        
    from model import TabliterModel
    from train import train
    from evaluate import evaluate
    from describe import describe_model
    from process import process
    
    # Constants
    # RUN_NAME = datetime.now().strftime("%Y_%m_%d__%H_%M")
    RUN_NAME = '5_weightsReported'
    PADDING = 40

    PATH_ROOT = Path(r"F:\ml-parsing-project\table-parse-split")
    TASKS = {'train': True, 'eval': True, 'postprocess': True}
    BEST_RUN = Path(r"F:\ml-parsing-project\table-parse-split\models\test\model_best.pt")

    # Model parameters
    EPOCHS = 80
    MAX_LR = 0.08

    # Derived constants
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Paths
    path_data = PATH_ROOT / 'data' / 'real_narrow'
    path_model = PATH_ROOT / 'models' / RUN_NAME
    path_words = PATH_ROOT / 'data' / 'words'
    path_pdfs = PATH_ROOT / 'data' / 'pdfs'

    # Define model
    model = TabliterModel().to(DEVICE)

    if TASKS['train']:
        # Describe model
        describe_model(model)     

        # Train
        train(epochs=EPOCHS, max_lr=MAX_LR, 
                path_data_train=path_data / 'train', path_data_val=path_data / 'val',
                path_model=path_model, device=DEVICE, replace_dirs=True)

    if TASKS['eval']:
        path_best_model = BEST_RUN if not TASKS['train'] else path_model / 'model_best.pt'
        evaluate(path_model_file=path_best_model, path_data=path_data / 'val', device=DEVICE, replace_dirs=True)


    if TASKS['postprocess']:
        path_best_model = BEST_RUN if not TASKS['train'] else path_model / 'model_best.pt'
        process(path_model_file=path_best_model, path_data=path_data / 'val', device=DEVICE, replace_dirs=True,
                    path_pdfs=path_pdfs, path_words=path_words, padding=PADDING)

if __name__ == '__main__':
    run()