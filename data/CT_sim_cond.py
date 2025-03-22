import os
import cv2
import numpy as np
from PIL import Image

from torch.utils.data import Dataset




class CT_cond_extra(Dataset):
    def __init__(self, path, label_path):
        self.base_root = path
        self.label_root = label_path
        self.base = self._get_base(self.base_root)
        self.label = self._get_base(self.label_root)
        
       

    def _get_base(self, path):
        cond_list = []
        cond_files = os.listdir(path)
        cond_files.sort(key=lambda x: int(x.split('.')[0]))
        for files in cond_files:
            cond_list.append(os.path.join(path, files))
        return cond_list

       
    def __len__(self):
        return len(self.base)
        #  return 10

    def __getitem__(self, i):

        cond = Image.open(self.base[i])
        cond = np.array(cond).astype(np.uint8)
      
        label = Image.open(self.label[i])
        label = np.array(label).astype(np.int64)
        
        example = {}
        example["image"] = (cond/127.5 - 1.0).astype(np.float32)
        example["label"] = label


        return example
