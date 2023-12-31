import cv2
from torch.utils.data import Dataset
import random
import os
import numpy as np
import torch
from collections import OrderedDict
from torchvision import transforms
from .transforms import JointRandHrzFlip, JointResize, \
                              JointNormalize, JointToTensor, \
                               JointRandVertFlip
                            

class SBUDataset(Dataset):
    def __init__(self,
                 data_root,
                 phase=None,
                 img_dirs=['train_A'],
                 mask_dir='train_B',
                #  noshad_dir='train_C',
                 augmentation=False,
                 im_size=400,
                 max_dataset_size=None,
                 normalize=True):
        
        self.root_dir = data_root
        self.img_dirs = img_dirs
        self.noshad_dir = noshad_dir
        self.img_names = sorted(os.listdir(os.path.join(self.root_dir, img_dirs[0])))
        self.mask_dir = mask_dir
        self.augmentation = augmentation
        self.phase = phase
        
        self.size = len(self.img_names)
        # None means doesn't change the size of dataset to be loaded
        if max_dataset_size is not None:
            assert isinstance(max_dataset_size, int) and max_dataset_size > 0
            self.size = min(max_dataset_size, self.size)
            self.img_names = self.img_names[:self.size]
        
        assert phase in ['train', 'val', 'test', None]
        if phase == 'train':
            self.joint_transform = transforms.Compose([JointRandHrzFlip(),
                                                    #    JointRandVertFlip(),
                                                       JointResize(im_size)])
            # self.joint_transform = transforms.Compose([JointRandHrzFlip()])
            img_transform = [ JointToTensor() ]
            if normalize:
                img_transform.append( JointNormalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225]) )  
            self.img_transform = transforms.Compose(img_transform)
            self.target_transform = transforms.ToTensor()

        elif phase in ['val', 'test']:
            self.joint_transform = None

            img_transform = [ JointResize(im_size), JointToTensor() ]
            if normalize:
                img_transform.append( JointNormalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225]) )  
            self.img_transform = transforms.Compose(img_transform)

            self.target_transform = transforms.Compose([JointResize(im_size), transforms.ToTensor()])
            
        else: # pahse is None
            self.joint_transform = None
            self.img_transform = None
            self.target_transform = None

    def _load_sample_pairs(self):
        pass

    def __getitem__(self, index):
        sample = OrderedDict()
        img_name = self.img_names[index]

        if self.augmentation:
            ret_key = ['train_A_input']
            img_dir = random.choice(self.img_dirs)
            img_path = os.path.join(self.root_dir, img_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ret_val= [ img ]


        else:
            ret_key = []
            ret_val = []
            for img_dir in self.img_dirs:
                img_path = os.path.join(self.root_dir, img_dir, img_name)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ret_key.append(img_dir+'_input')
                ret_val.append(img)

        mask_name = os.path.splitext(img_name)[0]+'.png'
        mask_path = os.path.join(self.root_dir, self.mask_dir, mask_name)
        # print(mask_path)
        mask = ((cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 125)*255).astype(np.uint8)
        ret_key.append('gt')
        ret_val.append(mask)

        # noshad_name = os.path.splitext(img_name)[0]+'.png'
        # noshad_path = os.path.join(self.root_dir, self.noshad_dir, noshad_name)
        # # print(noshad_path)
        # noshad = cv2.imread(noshad_path)
        # noshad = cv2.cvtColor(noshad, cv2.COLOR_BGR2RGB)
        # ret_key.append('noshad')
        # ret_val.append(noshad)

        # if self.phase == 'train':
        #     noshad_name = os.path.splitext(img_name)[0]+'.png'
        #     noshad_path = os.path.join(self.root_dir, self.noshad_dir, noshad_name)
        #     noshad = cv2.imread(noshad_path)
        #     noshad = cv2.cvtColor(noshad, cv2.COLOR_BGR2RGB)
        #     ret_key.append('noshad')
        #     ret_val.append(noshad)

        if self.joint_transform:
            ret_val = self.joint_transform(ret_val)

        if self.img_transform:
            ret_val[:-1] = self.img_transform(ret_val[:-1])

        if self.target_transform:
            ret_val[-1] = self.target_transform(ret_val[-1])
            
        ret_key.append('im_name')
        ret_val.append(img_name)
        
        # print(ret_key)
        # print(ret_val)
        return OrderedDict(zip(ret_key, ret_val))


    def __len__(self):
        return self.size
    
    
    