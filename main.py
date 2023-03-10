import os
import numpy as np
import argparse

from data import CropDataset
from torch.utils.data import DataLoader
import yaml

from utils import Config
from src import ResNetLSTM, VGGLSTM, DeformableConvLSTM

import torch
import torch.nn as nn

from utils import Trainer, set_seed

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default = "config.yaml", help = "Config File")
    
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
        config = Config(**config)

    set_seed(config.seed)

    data_path = os.path.join(config.data.data_dir, config.data.data_file)
    gt_path = os.path.join(config.data.data_dir, config.data.gt_file)

    train_dataset = CropDataset(data_path, gt_path, mode = "train", eval_mode = False, fold = config.data.fold,
                                time_downsample_factor = config.data.time_downsample_factor, num_channel = config.data.num_channel, 
                                apply_cloud_masking = config.data.apply_cloud_masking, cloud_threshold = config.data.cloud_threshold,
                                return_cloud_cover = config.data.return_cloud_cover, small_train_set_mode = config.data.small_train_set_mode)
    test_dataset = CropDataset(data_path, gt_path, mode = "test", eval_mode = False, fold = config.data.fold,
                                time_downsample_factor = config.data.time_downsample_factor, num_channel = config.data.num_channel, 
                                apply_cloud_masking = config.data.apply_cloud_masking, cloud_threshold = config.data.cloud_threshold,
                                return_cloud_cover = config.data.return_cloud_cover, small_train_set_mode = config.data.small_train_set_mode)

    train_dataloader = DataLoader(train_dataset, batch_size = config.data.batch_size, shuffle = True, num_workers = config.num_workers, 
                                collate_fn = train_dataset.collate_fn)
    test_dataloader = DataLoader(train_dataset, batch_size = config.data.batch_size, shuffle = False, num_workers = config.num_workers, 
                                collate_fn = test_dataset.collate_fn)

    if config.model.model_name == "ResNetLSTM":
        model = ResNetLSTM(config)
    elif config.model.model_name == "VGGLSTM":
        model = VGGLSTM(config)
    elif config.model.model_name == "DeformableConvLSTM":
        model = DeformableConvLSTM(config)
    
    print(f"Model: {config.model.model_name} \n{model}")
    exit(0)

    total_params = sum(param.numel() for param in model.parameters())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = torch.optim.Adam(model.parameters(), lr = float(config.training.optim_params.lr), weight_decay = float(config.training.optim_params.weight_decay))
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(config, model, optimizer, criterion, device)

    trainer.train(train_dataloader, test_dataloader)
    train_metrics = trainer.compute_metrics(train_dataloader)
    test_metrics = trainer.compute_metrics(test_dataloader)
    print(f"\nTrain Metrics: {train_metrics}")
    print(f"Test Metrics: {test_metrics}\n")