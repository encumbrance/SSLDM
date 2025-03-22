import os
import cv2
import numpy as np
import  random
from PIL import Image
from torch.utils.data import Dataset
from scipy.signal import find_peaks

def label_partition(label_list, percentage):
    seed = 33
    random.seed(seed)
    percentlist_pre = random.sample(label_list, int(len(label_list)*percentage))
    percentlist_post = [x for x in label_list if x not in percentlist_pre]
    return percentlist_pre, percentlist_post


def HU_syn_b(cond_gray, win=25,  mode="aggresive"):

    cond_hist = cv2.calcHist([cond_gray], [0], None, [256], [0, 256])
    cond_hist = cond_hist.flatten()


    peaks, _ = find_peaks(cond_hist, distance=50)
    
    if len(peaks) > 1:
        peaks = sorted(peaks, key=lambda x: abs(x - 125))
    
    cond_peak = peaks[0]

    cond_syn = np.zeros_like(cond_gray)

    if mode == "aggresive":
        edge_region = cv2.Canny(cond_gray, 50, 150)
        strength = np.random.randint(80, 100)   
        thickness = np.random.randint(10, 30)
    else:
        edge_region = cv2.Canny(cond_gray, 150, 255)
        strength = np.random.randint(50, 80)
        thickness = np.random.randint(10, 30)

    binary = np.zeros_like(cond_gray)
    for i in range(cond_gray.shape[0]):
        for j in range(cond_gray.shape[0]):
            if cond_gray[i][j] != 0:
                binary[i][j] = 255
            else:
                binary[i][j] = 0
    #dilate the overlap region
    kernel = np.ones((5,5),np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    binary = cv2.erode(binary, kernel, iterations=1)
 
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cond_gray, connectivity=8)
    idx = np.argsort(stats[:, -1])[::-1]
    binary = np.zeros_like(cond_gray)
    mask_idx = idx[1]
    mask = labels == mask_idx
    binary[mask] = 255
    x_b, y_b, w, h = cv2.boundingRect(binary)


    for i in range(cond_gray.shape[0]):
        for j in range(cond_gray.shape[1]):
            if cond_gray[i][j] < (cond_peak - win) or cond_gray[i][j] > (cond_peak + win):
                cond_syn[i][j] = cond_gray[i][j]  
            #if the pixel is to close to the boundings of the box, do not change it  
            elif i < y_b + thickness or i > y_b + h - thickness or j < x_b + 2*thickness or j > x_b + w - 2*thickness:
                cond_syn[i][j] = cond_gray[i][j]
            else:
                kernel_w = 2
                x_start = max(0, i - kernel_w)
                x_end = min(cond_gray.shape[0], i + kernel_w)
                y_start = max(0, j - kernel_w)
                y_end = min(cond_gray.shape[1], j + kernel_w)

                break_inner_loop = False
                for x in range(x_start, x_end):
                    for y in range(y_start, y_end):
                        if edge_region[x][y] == 255:
                            cond_syn[i][j] = cond_gray[i][j]
                            break_inner_loop = True
                            break
                    if break_inner_loop:
                        break

                if not break_inner_loop:
                    cond_syn[i][j] = cond_gray[i][j] + strength
    
    cond_syn = cv2.cvtColor(cond_syn, cv2.COLOR_GRAY2RGB)
    
    return cond_syn

def get_image_dict(path, data_type="image", label_map=None, box=None):
    img = Image.open(path)
    # if not img.mode == "RGB":
    #     img = img.convert("RGB")
    img = np.array(img).astype(np.uint8)
    if data_type == "image":
        img = (img/127.5 - 1.0).astype(np.float32)

    elif data_type == "hint":
        detected_map = cv2.Canny(img, 85, 225)
        detected_map = np.concatenate([detected_map[:, :, np.newaxis]] * 3, axis=2)
        detected_map = (detected_map/127.5 - 1.0).astype(np.float32)
        img = detected_map

    elif data_type == "hybrid":
        detected_map = cv2.Canny(img, 50, 100)
        for i in range (detected_map.shape[0]):
            for j in range (detected_map.shape[1]):
                if detected_map[i][j] == 255:
                    img[i][j] = [255, 255, 255]
        img = (img/127.5 - 1.0).astype(np.float32)

    elif data_type == "label":
        img = img.astype(np.int64)

    elif data_type == "syn":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = HU_syn_b(img, mode="aggresive")
        img = (img/127.5 - 1.0).astype(np.float32)

    elif data_type == "syn_teach":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = HU_syn_b(img, mode="conservative")
        img = (img/127.5 - 1.0).astype(np.float32)

    return img


class CT_sim_SR_SSCL(Dataset):
    def __init__(self, size=None, batch_size=4):   
        self.batch_size = batch_size
        assert batch_size % 2 == 0
        self.data_count = batch_size // 2
        self.base = self._get_base()
        self.size = size


    def _get_base(self):
        raise NotImplementedError

    def __len__(self):
        count = self.data_count
        length = len(self.base["noiv_unlabel"]) // count
        return length

    def __getitem__(self, i):
        example = {}
        example["image"] = []
        example["image_teach"] = []
        example["hint"] = []
        example["label"] = []   

        for j in range(self.data_count):

            # iv_unlabel_path = self.base["iv_unlabel"][i * self.data_count + j]
            noiv_unlabel_path = self.base["noiv_unlabel"][i * self.data_count + j]
            label_path_u = self.base["label_u"][i * self.data_count + j]


            iv_unlabel = get_image_dict(noiv_unlabel_path, data_type="syn")
            iv_unlabel_teach = get_image_dict(noiv_unlabel_path, data_type="syn_teach")
            hint = get_image_dict(noiv_unlabel_path, data_type="image")
            label = get_image_dict(label_path_u, data_type="label")

            example["image"].append(iv_unlabel)
            example["image_teach"].append(iv_unlabel_teach)
            example["hint"].append(hint)
            example["label"].append(label)


        
        indices = random.sample(range(len(self.base["noiv_label"])), self.data_count)
        for idx in indices:

            iv_label_path = self.base["iv_label"][idx]
            noiv_label_path = self.base["noiv_label"][idx]
            label_path = self.base["label"][idx]

            iv_label = get_image_dict(iv_label_path, data_type="image")
            hint = get_image_dict(noiv_label_path, data_type="image")
            label = get_image_dict(label_path, data_type="label")
            
            example["image"].append(iv_label)
            example["hint"].append(hint)
            example["label"].append(label)
        
        example["image"] = np.array(example["image"])
        example["image_teach"] = np.array(example["image_teach"])
        example["hint"] = np.array(example["hint"])
        example["label"] = np.array(example["label"])
        return example



class CT_sim_SRTrain_SSCL(CT_sim_SR_SSCL):
    def __init__(self, 
            path_target=None, 
            path_cond=None, 
            path_cond_l=None,
            path_label=None, 
            path_label_u=None, 
            L_ratio=0.05, **kwargs):
        

        self.base_root = path_target
        self.cond_root = path_cond
        self.cond_root_l = path_cond_l
        self.label_root = path_label
        self.label_root_u = path_label_u


        super().__init__(**kwargs)
        self.L_ratio = L_ratio
        assert self.base_root is not None
        assert self.cond_root is not None
        assert self.cond_root_l is not None
        assert self.label_root is not None
        assert self.label_root_u is not None

        

    def _get_base(self):
        path = {}

        path["iv_label"] = []
        # path["iv_unlabel"] = []

        path["noiv_label"] = []
        path["noiv_unlabel"] = []

        path["label"] = []
        path["label_u"] = []



        iv_files = os.listdir(self.base_root)
        noiv_files = os.listdir(self.cond_root)
        label_files_u = os.listdir(self.label_root_u)
        label_files = os.listdir(self.label_root)
        noiv_files_label = os.listdir(self.cond_root_l)

        iv_files.sort(key=lambda x: int(x.split(".")[0]))
        noiv_files.sort(key=lambda x: int(x.split(".")[0]))
        label_files_u.sort(key=lambda x: int(x.split(".")[0]))
        label_files.sort(key=lambda x: int(x.split(".")[0]))
        noiv_files_label.sort(key=lambda x: int(x.split(".")[0]))


        iv_files_label, _ = label_partition(iv_files, 0.05)
        _, noiv_files_unlabel = label_partition(noiv_files, 0.05)
        _, label_files_u = label_partition(label_files_u, 0.05)


        for files in iv_files_label:
            path["iv_label"].append(os.path.join(self.base_root, files))
        # for files in iv_files_unlabel:
        #     path["iv_unlabel"].append(os.path.join(self.base_root, files))
        for files in noiv_files_unlabel:
            path["noiv_unlabel"].append(os.path.join(self.cond_root, files))
        for files in label_files_u:
            path["label_u"].append(os.path.join(self.label_root_u, files))
        for files in label_files:
            path["label"].append(os.path.join(self.label_root, files))
        for files in noiv_files_label:
            path["noiv_label"].append(os.path.join(self.cond_root_l, files))
        
        return path
    



class CT_sim_SRVal_SSCL(CT_sim_SR_SSCL):
    def __init__(self, 
            path_target=None, 
            path_cond=None, 
            path_cond_l=None,
            path_label=None, 
            path_label_u=None, 
            L_ratio=0.05, **kwargs):
        
        self.base_root = path_target
        self.cond_root = path_cond
        self.cond_root_l = path_cond_l
        self.label_root = path_label
        self.label_root_u = path_label_u

        super().__init__(**kwargs)
        self.L_ratio = L_ratio

        assert self.base_root is not None
        assert self.cond_root is not None
        assert self.cond_root_l is not None
        assert self.label_root is not None
        assert self.label_root_u is not None

        

    def _get_base(self):
        path = {}

        path["iv_label"] = []
        # path["iv_unlabel"] = []

        path["noiv_label"] = []
        path["noiv_unlabel"] = []

        path["label"] = []
        path["label_u"] = []


        iv_files = os.listdir(self.base_root)
        noiv_files = os.listdir(self.cond_root)
        label_files = os.listdir(self.label_root)
        label_files_u = os.listdir(self.label_root_u)
        noiv_files_label = os.listdir(self.cond_root_l)
        
        
  
        iv_files.sort(key=lambda x: int(x.split(".")[0]))
        noiv_files.sort(key=lambda x: int(x.split(".")[0]))
        label_files.sort(key=lambda x: int(x.split(".")[0]))
        label_files_u.sort(key=lambda x: int(x.split(".")[0]))
        noiv_files_label.sort(key=lambda x: int(x.split(".")[0]))
    
     

        iv_files_label, _ = label_partition(iv_files, 0.05)
        _, noiv_files_unlabel = label_partition(noiv_files, 0.05)
        _, label_files_u = label_partition(label_files_u, 0.05)


        for file in iv_files_label:
            path["iv_label"].append(os.path.join(self.base_root, file))
        # for file in iv_files_unlabel:
        #     path["iv_unlabel"].append(os.path.join(self.base_root, file))
        for files in noiv_files_label:
            path["noiv_label"].append(os.path.join(self.cond_root_l, files))
        for file in noiv_files_unlabel:
            path["noiv_unlabel"].append(os.path.join(self.cond_root, file))
        for files in label_files:
            path["label"].append(os.path.join(self.label_root, files))
        for file in label_files_u:
            path["label_u"].append(os.path.join(self.label_root_u, file))

        return path