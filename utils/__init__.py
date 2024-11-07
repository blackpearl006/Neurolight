import os
from .visualisation import *
# from .trainer import Trainer
from .trainer_updated import Trainer
from .logger_utils import setup_logger
from torch.distributed import init_process_group
import yaml

def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config