import os
import json
import numpy as np
import pickle as pkl

from PIL import Image
from torch.utils.data import Dataset


# AFFORDANCE_COLOR_CODES = {
#     "sit" : [220, 0, 115],
#     "run" : [0, 100, 100],
#     "grasp" : [15, 50, 70],
# }

class ADE20kAffordanceDataset(Dataset):
    def __init__(self, data_dir, data_type='training'):
        self.ade20k_path = os.path.join(data_dir, 'source')
        self.affordance_path = os.path.join(data_dir, 'affordance')
        self.depth_path = os.path.join(data_dir, 'depth')
        self.control_path = os.path.join(data_dir, 'combined')
        self.prompt_path = data_dir
        self.data_dir = data_dir
        self.data_type = data_type
        
        self.prompt_train_file_path = os.path.join(self.prompt_path, 'ade20k_captions.txt')
        with open(self.prompt_train_file_path, 'r') as f:
            lines = f.readlines()
            self.prompt_list = [line.strip().split(',')[1] for line in lines]


    def __len__(self):
        return len(self.prompt_list)
    
    def __getitem__(self, idx):
        img_file_name = self.ade20k_path+'/src'+str(idx)+'.png'
        ctr_file_name = self.control_path+'/test'+str(idx)+'.png'
        prompt = self.prompt_list[idx].replace('caption:','')

        target = np.asarray(Image.open(img_file_name))
        target = target/127.5 - 1
        source = np.asarray(Image.open(ctr_file_name))
        source = source/255.0
        
        return dict(jpg = target, txt = prompt,  hint = source)
