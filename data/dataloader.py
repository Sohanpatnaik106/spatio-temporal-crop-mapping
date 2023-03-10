import os
import csv
import glob
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

class CropDataset(Dataset):
    
    def __init__(self, data_path, gt_path, t = 0.9, mode = 'all', eval_mode = False, fold = None,
                 time_downsample_factor = 2, num_channel = 4, apply_cloud_masking = False, cloud_threshold = 0.1,
                 return_cloud_cover = False, small_train_set_mode = False):
        
        self.data_path = data_path
        self.gt_path = gt_path

        self.data = h5py.File(self.data_path, "r", libver='latest', swmr = True)

        print(f"Loading {mode} data\n")

        self.samples = self.data["data"].shape[0]
        self.max_obs = self.data["data"].shape[1]
        self.spatial = self.data["data"].shape[2:-1]
        self.t = t
        self.augment_rate = 0.66
        self.eval_mode = eval_mode
        self.fold = fold
        self.num_channel = num_channel
        self.apply_cloud_masking = apply_cloud_masking
        self.cloud_threshold = cloud_threshold
        self.return_cloud_cover = return_cloud_cover
        self.small_train_set_mode = small_train_set_mode

        self.mode = mode

        if self.small_train_set_mode:
            print('Small training-set mode - fold: ', fold, '  Mode: ', mode)
            self.valid_list = self._split_train_test_23(mode, self.fold)

        elif self.fold != None:
            print('5fold: ', fold, '  Mode: ', mode)
            self.valid_list = self._split_5fold(mode, self.fold) 

        else:
            self.valid_list = self._split(mode)

        self.valid_samples = self.valid_list.shape[0]

        self.time_downsample_factor = time_downsample_factor
        self.max_obs = int(142 / self.time_downsample_factor)   # In the paper, they considered 71 time frames      
            
        file = open(self.gt_path, "r")
        tier_1, tier_2, tier_3, tier_4 = [], [], [], []

        reader = csv.reader(file)
        for line in reader:
            tier_1.append(line[-5])
            tier_2.append(line[-4])
            tier_3.append(line[-3])
            tier_4.append(line[-2])
    
        tier_2[0], tier_3[0], tier_4[0] = '0_unknown', '0_unknown', '0_unknown'
    
        self.label_list = []
        for i in range(len(tier_2)):
            
            if tier_1[i] == 'Vegetation' and tier_4[i] != '':
                self.label_list.append(i)
                
            if tier_2[i] == '':
                tier_2[i] = '0_unknown'

            if tier_3[i] == '':
                tier_3[i] = '0_unknown'
            
            if tier_4[i] == '':
                tier_4[i] = '0_unknown'

        tier_2_elements, tier_3_elements, tier_4_elements = list(set(tier_2)), list(set(tier_3)), list(set(tier_4))
        tier_2_elements.sort()
        tier_3_elements.sort()
        tier_4_elements.sort()

            
        tier_2_, tier_3_, tier_4_ = [], [], []
        for i in range(len(tier_2)):
            tier_2_.append(tier_2_elements.index(tier_2[i]))
            tier_3_.append(tier_3_elements.index(tier_3[i]))
            tier_4_.append(tier_4_elements.index(tier_4[i])) 

        self.label_list_local_1, self.label_list_local_2, self.label_list_glob = [], [], []
        self.label_list_local_1_name, self.label_list_local_2_name, self.label_list_glob_name = [], [], []

        for gt in self.label_list:
            self.label_list_local_1.append(tier_2_[int(gt)])
            self.label_list_local_2.append(tier_3_[int(gt)])
            self.label_list_glob.append(tier_4_[int(gt)])
            
            self.label_list_local_1_name.append(tier_2[int(gt)])
            self.label_list_local_2_name.append(tier_3[int(gt)])
            self.label_list_glob_name.append(tier_4[int(gt)])

        self.n_classes = max(self.label_list_glob) + 1
        self.n_classes_local_1 = max(self.label_list_local_1) + 1
        self.n_classes_local_2 = max(self.label_list_local_2) + 1

        print('Dataset size: ', self.samples)
        print('Valid dataset size: ', self.valid_samples)
        print('Sequence length: ', self.max_obs)
        print('Spatial size: ', self.spatial)
        print('Number of classes: ', self.n_classes)
        print('Number of classes - local-1: ', self.n_classes_local_1)
        print('Number of classes - local-2: ', self.n_classes_local_2)

    def __len__(self):
        return self.valid_samples

    def __getitem__(self, idx):
                     
        idx = self.valid_list[idx]
        X = self.data["data"][idx]

        if self.apply_cloud_masking or self.return_cloud_cover:
            CC = self.data["cloud_cover"][idx]

        target_ = self.data["gt"][idx,...,0]
        if self.eval_mode:
            gt_instance = self.data["gt_instance"][idx,...,0]

        X = np.transpose(X, (0, 3, 1, 2))

        # Temporal downsampling
        X = X[0::self.time_downsample_factor,:self.num_channel,...]

        if self.apply_cloud_masking or self.return_cloud_cover:
            CC = CC[0::self.time_downsample_factor,...]

        #Change labels 
        target = np.zeros_like(target_)

        for i in range(len(self.label_list)):
            target[target_ == self.label_list[i]] = self.label_list_glob[i]
        
        X = torch.from_numpy(X)
        target = torch.from_numpy(target).float()

        if self.apply_cloud_masking or self.return_cloud_cover:
            CC = torch.from_numpy(CC).float()

        if self.eval_mode:
            gt_instance = torch.from_numpy(gt_instance).float()

        #keep values between 0-1
        X = X * 1e-4
        #Previous line should be modified as X = X / 4095 but not tested yet!

        # Cloud masking
        if self.apply_cloud_masking:
            CC_mask = CC < self.cloud_threshold
            CC_mask = CC_mask.view(CC_mask.shape[0],1,CC_mask.shape[1],CC_mask.shape[2])
            X = X * CC_mask.float()

        #augmentation
        if self.eval_mode==False and np.random.rand() < self.augment_rate and self.mode != "test":

            flip_dir  = np.random.randint(3)

            if flip_dir == 0:
                X = X.flip(2)
                target = target.flip(0)

            elif flip_dir == 1:
                X = X.flip(3)
                target = target.flip(1)

            elif flip_dir == 2:
                X = X.flip(2,3)
                target = target.flip(0, 1)  

        if self.return_cloud_cover:

            if self.eval_mode:
                # return X.float(), target.long(), target_local_1.long(), target_local_2.long(), gt_instance.long(), CC.float()
                return X.float(), target.long(), gt_instance.long(), CC.float()
            else:
                # return X.float(), target.long(), target_local_1.long(), target_local_2.long(), CC.float()
                return X.float(), target.long(), CC.float()
        
        else:
            if self.eval_mode:
                # return X.float(), target.long(), target_local_1.long(), target_local_2.long(), gt_instance.long()
                return X.float(), target.long(), gt_instance.long()
            else:
                # return X.float(), target.long(), target_local_1.long(), target_local_2.long()
                return X.float(), target.long()

    def collate_fn(self, items):

        if self.return_cloud_cover:
            if self.eval_mode:
                batch = {
                    'sample': torch.stack([x[0] for x in items], dim=0),
                    'target': torch.stack([x[1] for x in items], dim=0),
                    'gt_instance': torch.stack([x[2] for x in items], dim=0),
                    'cloud_cover': torch.stack([x[3] for x in items], dim=0)
                }
            else:
                batch = {
                    'sample': torch.stack([x[0] for x in items], dim=0),
                    'target': torch.stack([x[1] for x in items], dim=0),
                    'cloud_cover': torch.stack([x[2] for x in items], dim=0)
                }
                
        else:
            if self.eval_mode:
                batch = {
                    'sample': torch.stack([x[0] for x in items], dim=0),
                    'target': torch.stack([x[1] for x in items], dim=0),
                    'gt_instance': torch.stack([x[2] for x in items], dim=0),
                }
            else:
                batch = {
                    'sample': torch.stack([x[0] for x in items], dim=0),
                    'target': torch.stack([x[1] for x in items], dim=0),
                }
                
        return batch
        
    def _split(self, mode):
        valid = np.zeros(self.samples)
        print(valid.shape)

        if mode=='test':
            valid[int(self.samples*0.75):] = 1.
        elif mode=='train':
            valid[:int(self.samples*0.75)] = 1.
        else:
            valid[:] = 1.

        w,h = self.data["gt"][0,...,0].shape
        
        return np.nonzero(valid)[0]

    def _split_5fold(self, mode, fold):
        
        if fold == 1:
            test_s = int(0)
            test_f = int(self.samples*0.2)
        elif fold == 2:
            test_s = int(self.samples*0.2)
            test_f = int(self.samples*0.4)
        elif fold == 3:
            test_s = int(self.samples*0.4)
            test_f = int(self.samples*0.6)
        elif fold == 4:
            test_s = int(self.samples*0.6)
            test_f = int(self.samples*0.8)
        elif fold == 5:
            test_s = int(self.samples*0.8)
            test_f = int(self.samples)            
                     
        if mode=='test':
            valid = np.zeros(self.samples)
            valid[test_s:test_f] = 1.
        elif mode=='train':
            valid = np.ones(self.samples)
            valid[test_s:test_f] = 0.

        w,h = self.data["gt"][0,...,0].shape
        for i in range(self.samples):
            if np.sum( self.data["gt"][i,...,0] != 0 )/(w*h) < self.t:
                valid[i] = 0
        
        return np.nonzero(valid)[0]

    def _split_train_test_23(self, mode, fold):

        if fold == 1:
            train_s = int(0)
            train_f = int(self.samples * 0.4)
        elif fold == 2:
            train_s = int(self.samples * 0.2)
            train_f = int(self.samples * 0.6)
        elif fold == 3:
            train_s = int(self.samples * 0.4)
            train_f = int(self.samples * 0.8)
        elif fold == 4:
            train_s = int(self.samples * 0.6)
            train_f = int(self.samples * 1.0)

        if mode == 'test':
            valid = np.ones(self.samples)
            valid[train_s:train_f] = 0.
        elif mode == 'train':
            valid = np.zeros(self.samples)
            valid[train_s:train_f] = 1.

        w, h = self.data["gt"][0, ..., 0].shape
        for i in range(self.samples):
            if np.sum(self.data["gt"][i, ..., 0] != 0) / (w * h) < self.t:
                valid[i] = 0

        return np.nonzero(valid)[0]