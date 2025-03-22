import os
import cv2

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Subset
from skimage.exposure import histogram_matching


def HU_syn(cond, target, mode="global"):
    if len(cond.shape) == 3 or len(target.shape) == 3:
        print("HU_syn: input needs to be grayscale")
        raise ValueError

    if mode == "global":
        target_match_cond= histogram_matching.match_histograms(target, cond)
        
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                if target[i][j] == 0:
                    target_match_cond[i][j] = 0

        return target_match_cond
    
    else:
        mask = np.zeros_like(cond)
        for i in range(cond.shape[0]):
            for j in range(cond.shape[1]):
                if cond[i][j] == 0 or target[i][j] == 0:
                    mask[i][j] = 1

        masked_cond = np.ma.array(cond, mask=mask)
        masked_target = np.ma.array(target, mask=mask) 

        matched_pixels = histogram_matching.match_histograms(masked_cond.compressed(), masked_target.compressed())
        
        output = target.copy()
        idx = 0
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                if mask[i][j] == 0:
                    output[i][j] = matched_pixels[idx]
                    idx += 1


        return output




class CT_sim_SR(Dataset):
    def __init__(self, size=None):
        self.size = size
        self.base = self._get_base()
     

    def _get_base(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.base["iv"])


    def __getitem__(self, i):
        base = self.base["iv"][i]
        cond = self.base["noiv"][i]
        img = Image.open(base)
        cond_img = Image.open(cond)


        if not img.mode == "RGB":
            img = img.convert("RGB")

        if not cond_img.mode == "RGB":
            cond_img = cond_img.convert("RGB")

        img = np.array(img).astype(np.uint8)
        cond_img = np.array(cond_img).astype(np.uint8)

        '''
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cond_img_gray = cv2.cvtColor(cond_img, cv2.COLOR_RGB2GRAY)
        cond_img_gray = HU_syn(cond_img_gray, img_gray)
        cond_img = cv2.cvtColor(cond_img_gray, cv2.COLOR_GRAY2RGB)
        In practice, to prevent the duplicate computation, we precompute
        the HU_syn result and save it at the new directory as the self.con_root
        in the Dataset_convert.py
        '''

        example = {}
        example["image"] = (img/127.5 - 1.0).astype(np.float32)
        example["LR_image"] = (cond_img/127.5 - 1.0).astype(np.float32)


        return example


class CT_sim_SRTrain(CT_sim_SR):
    def __init__(self, cond_path=None, target_path=None,  **kwargs):
        self.base_root = target_path
        self.cond_root = cond_path
        super().__init__(**kwargs)
        assert self.base_root is not None
        assert self.cond_root is not None
        

    def _get_base(self):
        #return the file path of the dataset
        path = {}
        path["iv"] = []
        path["noiv"] = []

        iv_files = os.listdir(self.base_root)
        noiv_files = os.listdir(self.cond_root)

        iv_files.sort()
        noiv_files.sort()
        for files in iv_files:
            path["iv"].append(os.path.join(self.base_root, files))
        for files in noiv_files:
            path["noiv"].append(os.path.join(self.cond_root, files))
        print(f'path_len: {len(path["iv"])}')
        return path


        


class CT_sim_SRValidation(CT_sim_SR):
    def __init__(self, cond_path=None, target_path=None, **kwargs):
        self.base_root = target_path
        self.cond_root = cond_path
        super().__init__(**kwargs)
        assert self.base_root is not None
        assert self.cond_root is not None
        

    def _get_base(self):
        #return the file path of the dataset
        path = {}
        path["iv"] = []
        path["noiv"] = []

        iv_files = os.listdir(self.base_root)
        noiv_files = os.listdir(self.cond_root)

        iv_files.sort()
        noiv_files.sort()

        for files in iv_files:
            path["iv"].append(os.path.join(self.base_root, files))
        for files in noiv_files:
            path["noiv"].append(os.path.join(self.cond_root, files))

        print(f'path_len: {len(path["iv"])}')
        return path
