import os
import json
import numpy as np
import pickle as pkl

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image
import torch
from generate_depth_maps import img_to_depth


AFFORDANCE_COLOR_CODES = {
    "sit" : [220, 0, 115],
    "run" : [0, 100, 100],
    "grasp" : [15, 50, 70],
}

AFFORDANCE_COLOR_CODES_MERGING = {
    "sit" : [50,50,50],
    "run" : [100, 100, 100],
    "grasp" : [150, 150, 150],
}

class ADE20kAffordanceDataset(Dataset):
    def __init__(self, data_dir, data_type='training'):
        self.ade20k_path = os.path.join(data_dir, 'ADE20K_2021_17_01')
        self.affordance_path = os.path.join(data_dir, 'ADE-Affordance')
        self.prompt_path = os.path.join(data_dir, 'prompt')
        self.data_dir = data_dir
        self.data_type = data_type
        
        self.index_file = 'index_ade20k.pkl'
        with open(os.path.join(self.ade20k_path, self.index_file), 'rb') as f:
            self.index_ade20k = pkl.load(f)
        
        self.affordance_train_file_path = os.path.join(self.affordance_path, 'file_path', 'train_file_path.json')
        with open(self.affordance_train_file_path, 'r') as f:
            self.affordance_train_paths = json.load(f)
        
        # print("Loading captions...")
        # self.prompt_train_file_path = os.path.join(self.prompt_path, 'ade20k_train_captions.jsonl')
        # with open(self.prompt_train_file_path, 'r') as f:
        #     self.prompt_list = list(f)

        # self.prompt_dict = {}
        # for json_str in self.prompt_list:
        #     j = json.loads(json_str)
        #     self.prompt_dict[j["image_id"]] = j["caption"]
        # print("Done.")
        
        self.transform_source = transforms.Compose([
            transforms.Resize((512, 512)),
            # transforms.PILToTensor()
        ])
        self.transform_target = transforms.Compose([
            transforms.Resize((512, 512)),
            # transforms.PILToTensor()
        ])

    def __len__(self):
        return len(self.affordance_train_paths)
    
    def __getitem__(self, idx):
        i = int(self.affordance_train_paths[idx].split('_')[-1].split('.')[0]) - 1
        img_file_name = os.path.join(self.data_dir, '{}/{}'.format(self.index_ade20k['folder'][i], self.index_ade20k['filename'][i]))
        seg_file_name = img_file_name.replace('.jpg', '_seg.png')
        relationship_file = os.path.join(self.affordance_path, self.affordance_train_paths[idx] + '_relationship.txt')
        # prompt_id = "ADE_train_{0:08d}".format(i+1)

        depth_map = img_to_depth(np.asarray(self.transform_target(Image.open(img_file_name))))
        merged_map = np.copy(depth_map)

        run_ids = []
        sit_ids = []
        grasp_ids = []
        with open(relationship_file, 'r') as f:
            rels = f.read().splitlines()
            for rel in rels:
                rel = rel.split('|')[0]
                obj_id = int(rel.split('#')[0])
                if int(rel.split('#')[1]) != 0:
                    sit_ids.append(obj_id)
                if int(rel.split('#')[2]) != 0:
                    run_ids.append(obj_id)
                if int(rel.split('#')[3]) != 0:
                    grasp_ids.append(obj_id)
        
        with Image.open(seg_file_name) as io:
            seg = np.array(io)
        obj_ids = seg[:,:,2]
        affordance_seg = np.zeros_like(seg)
        merged_map = np.zeros_like(seg)
        for obj in sit_ids:
            affordance_seg[obj_ids == obj] = AFFORDANCE_COLOR_CODES['sit']
            # MAKE 3RD CHANNEL OF MERGED MAP 1 FOR SIT
            merged_map[obj_ids == obj] = 75

        for obj in grasp_ids:
            affordance_seg[obj_ids == obj] = AFFORDANCE_COLOR_CODES['grasp']
            merged_map[obj_ids == obj] = 150
        
        for obj in run_ids:
            affordance_seg[obj_ids == obj] = AFFORDANCE_COLOR_CODES['run']
            merged_map[obj_ids == obj] = 225
        
        affordance_seg = Image.fromarray(affordance_seg)
        target = np.asarray(self.transform_target(Image.open(img_file_name)))
        target = target/127.5 - 1
        source = np.array(self.transform_source(affordance_seg))
        source = source/255.0

        merged_map = Image.fromarray(merged_map)
        merged_map = np.asarray(self.transform_target(merged_map))
        merged_map = merged_map/255.0
        # copy first channel of depth map to first and second channels of merged map
        merged_map[:,:,0] = depth_map[:,:,0]
        # set second channel of merged map to 0 and third channel to 1 for grasp
        merged_map[:,:,1] = merged_map[:,:,1] * (merged_map[:,:,2] != 150.0/255.0)
        # set third channel of merged map to 0 for run
        merged_map[:,:,2] = merged_map[:,:,2] * (merged_map[:,:,2] != 225.0/255)
        
        return dict(jpg = target,  hint = source, depth = depth_map, merged = merged_map)


if __name__ == '__main__':
    data_dir = '/home/anish/Documents/vlr/project/datasets'
    dataset = ADE20kAffordanceDataset(data_dir)
    L = len(dataset)
    for i in range(L):
        data = dataset[i]
        jpg = data['jpg']   
        hint = data['hint']
        depth = data['depth']
        merged = data['merged']
        try:
            jpg = torch.from_numpy(jpg).permute(2, 0, 1)
            hint = torch.from_numpy(hint).permute(2, 0, 1)
            depth = torch.from_numpy(depth).permute(2, 0, 1)    
            merged = torch.from_numpy(merged).permute(2, 0, 1)
            # print(depth)
            source_dest = '/home/anish/Documents/vlr/project/datasets/Affordance_generated/source/'
            save_image([jpg], source_dest+'src{}.png'.format(i))
            aff_dest = '/home/anish/Documents/vlr/project/datasets/Affordance_generated/affordance/'
            save_image([hint], aff_dest+'aff{}.png'.format(i))
            depth_dest = '/home/anish/Documents/vlr/project/datasets/Affordance_generated/depth/'
            save_image([depth], depth_dest+'test{}.png'.format(i))
            merged_dest = '/home/anish/Documents/vlr/project/datasets/Affordance_generated/combined/'
            save_image([merged], merged_dest+'test{}.png'.format(i))
            print("Saved image {}".format(i))
        except:
            print("Error saving image {}".format(i))
            continue
