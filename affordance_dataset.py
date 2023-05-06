import os
import json
import numpy as np
import pickle as pkl

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image
import cv2


AFFORDANCE_COLOR_CODES = {
    "sit" : [220, 0, 115],
    "run" : [0, 100, 100],
    "grasp" : [15, 50, 70],
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

        self.affordance_val_file_path = os.path.join(self.affordance_path, 'file_path', 'val_file_path.json')
        with open(self.affordance_val_file_path, 'r') as f:
            self.affordance_val_paths = json.load(f)
        
        print("Loading captions...")
        self.prompt_train_file_path = os.path.join(self.prompt_path, 'ade20k_train_captions.jsonl')
        with open(self.prompt_train_file_path, 'r') as f:
            self.prompt_train_list = list(f)

        self.prompt_val_file_path = os.path.join(self.prompt_path, 'ade20k_validation_captions.jsonl')
        with open(self.prompt_val_file_path, 'r') as f:
            self.prompt_val_list = list(f)

        self.prompt_train_dict = {}
        self.prompt_val_dict = {}
        for json_str in self.prompt_train_list:
            j = json.loads(json_str)
            self.prompt_train_dict[j["image_id"]] = j["caption"]
        for json_str in self.prompt_val_list:
            j = json.loads(json_str)
            # print(j["image_id"])
            self.prompt_val_dict[j["image_id"]] = j["caption"]
        print("Done.")
        
        self.transform_source = transforms.Compose([
            transforms.Resize((512, 512)),
            # transforms.PILToTensor()
        ])
        self.transform_target = transforms.Compose([
            transforms.Resize((512, 512)),
            # transforms.PILToTensor()
        ])

    def __len__(self):
        if self.data_type=='training':
            return len(self.affordance_train_paths)
        else:
            return len(self.affordance_val_paths)
    
    def __getitem__(self, idx):
        if self.data_type=='training':
            i = int(self.affordance_train_paths[idx].split('_')[-1].split('.')[0]) - 1
            relationship_file = os.path.join(self.affordance_path, self.affordance_train_paths[idx] + '_relationship.txt')
            prompt_id = "ADE_train_{0:08d}".format(i+1)
            prompt = self.prompt_train_dict[prompt_id]
        else:
            i = int(self.affordance_val_paths[idx].split('_')[-1].split('.')[0]) - 1
            temp_path = self.affordance_val_paths[idx].split('/')
            temp_path[0] = 'validation'
            temp_path = os.path.join(*temp_path)
            # print(temp_path)
            relationship_file = os.path.join(self.affordance_path, temp_path + '_relationship.txt')
            prompt_id = "ADE_train_{0:08d}".format(i+1)
            prompt = self.prompt_train_dict[prompt_id]

        img_file_name = os.path.join(self.data_dir, '{}/{}'.format(self.index_ade20k['folder'][i], self.index_ade20k['filename'][i]))
        seg_file_name = img_file_name.replace('.jpg', '_seg.png')
        # relationship_file = os.path.join(self.affordance_path, self.affordance_train_paths[idx] + '_relationship.txt')
        # prompt_id = "ADE_train_{0:08d}".format(i+1)

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
        for obj in sit_ids:
            affordance_seg[obj_ids == obj] = AFFORDANCE_COLOR_CODES['sit']
        for obj in grasp_ids:
            affordance_seg[obj_ids == obj] = AFFORDANCE_COLOR_CODES['grasp']
        
        affordance_seg = Image.fromarray(affordance_seg)
        target = np.asarray(self.transform_target(Image.open(img_file_name)))
        target = target/127.5 - 1
        source = np.array(self.transform_source(affordance_seg))
        source = source/255.0
        
        return dict(jpg = target, txt = prompt,  hint = source)


if __name__ == '__main__':
    data_dir = '/home/alchemist/storage/vlr_data'
    dataset = ADE20kAffordanceDataset(data_dir, data_type='val')
    print(len(dataset))
    for i in [17,22,30,43]:
        data = dataset[i]
        jpg = data['jpg']
        hint = data['hint']
        txt = data['txt']
        print(txt)
        print(jpg.shape, hint.shape)
        hint_img = Image.fromarray((hint*255).astype(np.uint8))
        jpg_img = Image.fromarray(((jpg+1)*127.5).astype(np.uint8))
        hint_img.save("val_hint_{}.png".format(i))
        jpg_img.save("val_jpg_{}.png".format(i))
        # save_image([hint], 'val_hint_{}.png'.format(i))