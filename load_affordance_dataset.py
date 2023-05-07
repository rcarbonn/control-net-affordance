import os
import json
import numpy as np
import pickle as pkl

from PIL import Image
from torch.utils.data import Dataset
# from torchvision import transforms
# from torchvision.utils import save_image


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
        
        # self.index_file = 'index_ade20k.pkl'
        # with open(os.path.join(self.ade20k_path, self.index_file), 'rb') as f:
        #     self.index_ade20k = pkl.load(f)
        
        # self.affordance_train_file_path = os.path.join(self.affordance_path, 'file_path', 'train_file_path.json')
        # with open(self.affordance_train_file_path, 'r') as f:
        #     self.affordance_train_paths = json.load(f)
        
        # print("Loading captions...")
        self.prompt_train_file_path = os.path.join(self.prompt_path, 'ade20k_captions.txt')
        with open(self.prompt_train_file_path, 'r') as f:
            lines = f.readlines()
            self.prompt_list = [line.strip().split(',')[1] for line in lines]

        # self.prompt_dict = {}
        # for json_str in self.prompt_list:
        #     j = json.loads(json_str)
        #     self.prompt_dict[j["image_id"]] = j["caption"]
        # print("Done.")
        
        # self.transform_source = transforms.Compose([
        #     transforms.Resize((512, 512)),
        #     # transforms.PILToTensor()
        # ])
        # self.transform_target = transforms.Compose([
        #     transforms.Resize((512, 512)),
        #     # transforms.PILToTensor()
        # ])

    def __len__(self):
        return len(self.prompt_list)
    
    def __getitem__(self, idx):
        # img_file_name = os.path.join(self.ade20k_path,'/src{}.png'.format(idx))
        # ctr_file_name = os.path.join(self.control_path,'/test{}.png'.format(idx))
        img_file_name = self.ade20k_path+'/src'+str(idx)+'.png'
        ctr_file_name = self.control_path+'/test'+str(idx)+'.png'
        prompt = self.prompt_list[idx].replace('caption:','')
        # i = int(self.affordance_train_paths[idx].split('_')[-1].split('.')[0]) - 1
        # img_file_name = os.path.join(self.data_dir, '{}/{}'.format(self.index_ade20k['folder'][i], self.index_ade20k['filename'][i]))
        # seg_file_name = img_file_name.replace('.jpg', '_seg.png')
        # relationship_file = os.path.join(self.affordance_path, self.affordance_train_paths[idx] + '_relationship.txt')
        # prompt_id = "ADE_train_{0:08d}".format(i+1)

        # run_ids = []
        # sit_ids = []
        # grasp_ids = []
        # with open(relationship_file, 'r') as f:
        #     rels = f.read().splitlines()
        #     for rel in rels:
        #         rel = rel.split('|')[0]
        #         obj_id = int(rel.split('#')[0])
        #         if int(rel.split('#')[1]) != 0:
        #             sit_ids.append(obj_id)
        #         if int(rel.split('#')[2]) != 0:
        #             run_ids.append(obj_id)
        #         if int(rel.split('#')[3]) != 0:
        #             grasp_ids.append(obj_id)
        
        # with Image.open(seg_file_name) as io:
        #     seg = np.array(io)
        # obj_ids = seg[:,:,2]
        # affordance_seg = np.zeros_like(seg)
        # for obj in sit_ids:
        #     affordance_seg[obj_ids == obj] = AFFORDANCE_COLOR_CODES['sit']
        # for obj in grasp_ids:
        #     affordance_seg[obj_ids == obj] = AFFORDANCE_COLOR_CODES['grasp']
        
        # affordance_seg = Image.fromarray(affordance_seg)
        target = np.asarray(Image.open(img_file_name))
        target = target/127.5 - 1
        source = np.asarray(Image.open(ctr_file_name))
        source = source/255.0
        # prompt = self.prompt_dict[prompt_id]
        
        return dict(jpg = target, txt = prompt,  hint = source)


# if __name__ == '__main__':
#     data_dir = '/proj/'
#     dataset = ADE20kAffordanceDataset(data_dir)
#     print(len(dataset))
#     for i in range(10):
#         data = dataset[i]
#         jpg = data['jpg']
#         hint = data['hint']
#         txt = data['txt']
#         print(txt)
#         print(jpg.shape, hint.shape)
        # save_image([jpg, hint], 'test{}.png'.format(i))