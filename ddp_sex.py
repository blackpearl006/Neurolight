'''
to run: torchrun --standalone --nproc_per_node=7 ddp_sex.py
'''
from models import *
from utils import *
from utils.trainer_updated import Trainer
from data import *

import argparse
import torch
import warnings
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from torch.distributed import destroy_process_group

warnings.filterwarnings("ignore")

def create_dataset(LOGGER, config):
    dataset = Fieldmapdata(root_dir=config['dataset_dir'],label_dir=config['label_dir'],task=config['task'])
    if config['stratify']:
        train_dataset, test_dataset = stratified_split_classification(dataset, test_size=config['test_ratio'])
    else:
        train_dataset, test_dataset = random_split(dataset, [1-config['test_ratio'], config['test_ratio']])
    return train_dataset, test_dataset

def iniatlise_model_loss_optimiser(config):
    scheduler = False
    if config['model'] == 'SFCN': model = SFCN()
    if config['model'] == 'Resnet': 
        model = generate_model_resnet(model_depth=config['others']['resnetdepth'])
    if config['loss'] == 'WeightedBCEWithLogitsLoss': 
        criterion = nn.BCEWithLogitsLoss(pos_weight=config['pos_weight'])
    elif config['loss'] == 'BCEWithLogitsLoss': 
        criterion = nn.BCEWithLogitsLoss()
    elif config['loss'] == 'BCELoss': 
        criterion = nn.BCELoss()
    if config['optimizer'] == 'Adam': 
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    elif config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'])
    elif config['optimizer'] == 'Scheduler':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.3) #from peng et al
    return model, criterion, optimizer, scheduler

def KFoldtrain(train_dataset, test_dl, x_dl= None, LOGGER=None, config=None):
    K_FOLDS = config['kfolds']
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=config['fold_seed'])
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
        RESULT_DIR = f'{config['result_dir']}/{config['model']}_seed_{config['np_seed']}/fold_{fold}'
        os.makedirs(RESULT_DIR,exist_ok=True)   
        train_ds = Subset(train_dataset, train_idx)
        val_ds = Subset(train_dataset, val_idx)        
        train_dl = prepare_dataloader(train_ds, batch_size=config['batch_size'])
        val_dl = prepare_dataloader(val_ds, batch_size=config['batch_size'])

        model, criterion, optimizer, scheduler = iniatlise_model_loss_optimiser(config=config)
        

        trainer = Trainer(model= model, train_data=train_dl,
                          val_data=val_dl,
                          test_data=test_dl,
                          x_data=x_dl,
                          early_stopping_patience=config['early_stopping_patience'],
                          criterion=criterion,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          logger=LOGGER,
                          snapshot_path=f'{RESULT_DIR}/snapshot.pt',
                          save_every=config['save_every'],
                          best_snapshot_path=f'{RESULT_DIR}/best_snaposhot.pt',
                          result_path=RESULT_DIR,
                          note = f'Fold {fold+1}/{K_FOLDS}'
                          )
        trainer.train(config['total_epochs'])
        trainer.test()
        torch.cuda.empty_cache()

def main(config):
    np.random.seed(config['np_seed'])
    torch.manual_seed(config['torch_seed'])
    ddp_setup()
    LOGGER = setup_logger(logs_dir=f'{config['result_dir']}/{config['model']}_seed_{config['np_seed']}')
    LOGGER.info(config['note'])
    if int(os.environ["LOCAL_RANK"]) == 0:
        for i,j in config.items():
            LOGGER.info(f'Variable: {i} Value: {j}')
    train_dataset, test_dataset = create_dataset(
                                                LOGGER=LOGGER,
                                                config=config,
                                            )

    test_dl = prepare_dataloader(test_dataset, batch_size=config['batch_size'])

    KFoldtrain(
        train_dataset=train_dataset,
        test_dl=test_dl,
        LOGGER=LOGGER,
        config=config
    )

    destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distributed training job')
    parser.add_argument('--total_epochs', type=int, help='Total epochs to train the model')    
    parser.add_argument('--np_seed', type=int, help='Set seed for the model run (numpy)')    
    parser.add_argument('--torch_seed', type=int, help='Set seed for the model run (torch)')    
    parser.add_argument('--batch_size', type=int, help='Input batch size on each device (default: 16)')
    parser.add_argument('--config_filename', default='config1', type=str, help='Config path (default: SFCN)')
    parser.add_argument('--lr', type=float, help='learning rate')
    

    args = parser.parse_args()

    config = load_config(f'/home/neelamlab/ninad/DWI/configs/{args.config_filename}.yaml')

    config['total_epochs'] = args.total_epochs if args.total_epochs is not None else config.get('total_epochs')
    config['np_seed'] = args.np_seed if args.np_seed is not None else config.get('np_seed', 32)
    config['torch_seed'] = args.torch_seed if args.torch_seed is not None else config.get('torch_seed', 32)
    config['batch_size'] = args.batch_size if args.batch_size is not None else config.get('batch_size', 16)
    config['lr'] = args.lr if args.lr is not None else config.get('lr', 0.0001)
    config['pos_weight'] = config.get('pos_weight', 1.255)

    main(config)