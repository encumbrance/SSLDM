import os
import sys
import cv2
import random
import numpy as np
from PIL import Image
from CT_sim import HU_syn




def data_partition(src, ratio=0.05):
    '''
    partition the data into training and validation
    '''
    save_path = src + f"_{int(ratio*100)}"
    os.makedirs(save_path, exist_ok=True)
    
    files = os.listdir(src)
    files.sort(key=lambda x: int(x.split('.')[0]))

    seed = 33
    random.seed(seed)
    percent_list = random.sample(files, int(len(files)*ratio))


    for i in range(len(percent_list)):
        img = Image.open(os.path.join(src, percent_list[i]))
        img.save(os.path.join(save_path, f"{i}.png"))


def cond2label(src):
    save_path = src + "_cond2label_test"
    os.makedirs(save_path, exist_ok=True)

    files = os.listdir(src)
    files.sort(key=lambda x: int(x.split('.')[0]))

    for idx in range(len(files)):
        img = cv2.imread(os.path.join(src, files[idx]))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        label = np.zeros_like(img_gray)
        for i in range(img_gray.shape[0]):
            for j in range(img_gray.shape[1]):
                if img_gray[i][j] != 0:
                    label[i][j] = 1

        cv2.imwrite(os.path.join(save_path, f"{idx}.png"), label)
    
def HU_match(src, tar):
    save_path = src + "_match"
    os.makedirs(save_path, exist_ok=True)

    src_files = os.listdir(src)
    src_files.sort(key=lambda x: int(x.split('.')[0]))

    tar_files = os.listdir(tar)
    tar_files.sort(key=lambda x: int(x.split('.')[0]))

    for idx in range(len(src_files)):
        src_img = cv2.imread(os.path.join(src, src_files[idx]))
        tar_img = cv2.imread(os.path.join(tar, tar_files[idx]))

        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        tar_img = cv2.cvtColor(tar_img, cv2.COLOR_BGR2RGB)

        src_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
        tar_img = cv2.cvtColor(tar_img, cv2.COLOR_RGB2GRAY)

        match_img = HU_syn(src_img, tar_img, mode="global")
        match_img = match_img.astype(np.uint8)
        match_img = cv2.cvtColor(match_img, cv2.COLOR_GRAY2RGB)


        cv2.imwrite(os.path.join(save_path, f"{idx}.png"), match_img)

        

if __name__ == "__main__":
    type = sys.argv[1]

    if type == "p":
        src = sys.argv[2]
        ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
        data_partition(src, ratio)
        data_partition(src)

    elif type == "c2l":
        src = sys.argv[2]
        cond2label(src) 

    elif type == "HU":
        src = sys.argv[2]
        tar = sys.argv[3]
        HU_match(src, tar)

    elif type == "h":
        print("p: data partition <src> <ratio>")
        print("c2l: cond2label <src>")
        print("HU: HU_match <src> <tar>")

    else:
        raise ValueError(f"Invalid type: {type}")
