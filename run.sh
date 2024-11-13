#!bin/bash
torchrun --standalone --nproc_per_node=7 ddp_sex.py --config_filename config1 --np_seed 1
torchrun --standalone --nproc_per_node=7 ddp_sex.py --config_filename config2 --np_seed 2
torchrun --standalone --nproc_per_node=7 ddp_sex.py --config_filename config3 --np_seed 3
torchrun --standalone --nproc_per_node=7 ddp_sex.py --config_filename config4 --np_seed 4
torchrun --standalone --nproc_per_node=7 ddp_sex.py --config_filename config5 --np_seed 5
