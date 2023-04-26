import os
import json
import numpy as np
import pickle as pkl

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import cv2


AFFORDANCE_COLOR_CODES = {
    "sit" : [220, 0, 115],
    "run" : [0, 100, 100],
    "grasp" : [15, 50, 70],
}

background = [200, 222, 250]  
c1 = [0,0,205]   
c2 = [34,139,34] 
c3 = [192,192,128]   
c4 = [165,42,42]    
c5 = [128,64,128]   
c6 = [204,102,0]  
c7 = [184,134,11] 
c8 = [0,153,153]
c9 = [0,134,141]
c10 = [184,0,141] 
c11 = [184,134,0] 
c12 = [184,134,223]
aff_label_colors = np.array([background, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12], dtype=np.uint8)


class IITFAffordanceDataset(Dataset):
    def __init__(self, data_dir, data_type='training'):
        self.iitf_path = os.path.join(data_dir, 'IIT_Affordances_2017')
        self.affordance_path = os.path.join(self.iitf_path, 'affordances_labels')
        self.prompt_path = os.path.join(data_dir, 'prompt', 'iitaff_captions.txt')
        self.images_path = os.path.join(self.iitf_path, 'rgb')
        self.data_dir = data_dir
        self.data_type = data_type
        
        self.index_file = 'train_and_val.txt'
        with open(os.path.join(self.iitf_path, self.index_file), 'r') as f:
            self.index_iitf = f.read().splitlines()
        
        print('Loading captions')
        with open(self.prompt_path, 'r') as f:
            self.prompt_map = f.read().splitlines()
        
        self.prompt_id = {}
        for prt in self.prompt_map:
            data = prt.split(',')
            self.prompt_id[data[0].split(':')[1]] = data[1].split(':')[1] 
        
        self.transform_source = transforms.Compose([
            transforms.Resize((512, 512)),
            # transforms.PILToTensor()
        ])
        self.transform_target = transforms.Compose([
            transforms.Resize((512, 512)),
            # transforms.PILToTensor()
        ])

    def __len__(self):
        return len(self.index_iitf)
    
    def get_aff_mask(self, label_path):
        aff_labels = np.genfromtxt(label_path, dtype=np.uint8)
        aff_mask = np.zeros((*aff_labels.shape, 3), dtype=np.uint8)
        xx,yy = np.meshgrid(np.arange(aff_labels.shape[0]), np.arange(aff_labels.shape[1]))

        # aff_mask[xx,yy,...] = aff_label_colors[int(aff_labels[i,j])]
        # aff_mask = np.choose(aff_labels, aff_label_colors)
        for i,j in zip(xx.ravel(), yy.ravel()):
            aff_mask[i,j,...] = aff_label_colors[int(aff_labels[i,j])]
        return aff_mask

    
    def __getitem__(self, idx):
        # i = int(self.affordance_train_paths[idx].split('_')[-1].split('.')[0]) - 1
        i = idx
        img_file_name = os.path.join(self.images_path, self.index_iitf[i])
        seg_file_name = os.path.join(self.affordance_path, self.index_iitf[i].split('.')[0]+'.txt')

        affordance_seg = Image.fromarray(self.get_aff_mask(seg_file_name))
        target = np.asarray(self.transform_target(Image.open(img_file_name).convert('RGB')))
        target = target/127.5 - 1
        source = np.array(self.transform_source(affordance_seg))
        source = source/255.0
        prompt = self.prompt_id[self.index_iitf[i]]
        
        return dict(jpg = target, txt = prompt,  hint = source, id=prompt)


if __name__ == '__main__':
    data_dir = '/home/ubuntu/vlr_proj'
    dataset = IITFAffordanceDataset(data_dir)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=4, shuffle=True)
    print(len(dataset))
    # for i,d in enumerate(dataloader):
    #     print(i, d.keys(), d["id"])
    # for i in range(len(dataset)):
    for i in range(100):
        data = dataset[i]
        jpg = data['jpg']
        hint = data['hint']
        txt = data['txt']
        print(i, jpg.shape, hint.shape, len(txt))
        hint_img = Image.fromarray((hint*255).astype(np.uint8))
        jpg_img = Image.fromarray(((jpg+1)*127.5).astype(np.uint8))
        hint_img.save("./log_images/train_hint_{}.png".format(i))
        jpg_img.save("./log_images/train_jpg_{}.png".format(i))
        # save_image([jpg, hint], 'test{}.png'.format(i))