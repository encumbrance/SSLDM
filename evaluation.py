import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def _evaluation(cond_file, target_file, gen_file):
    cond = cv2.imread(cond_file)
    cond = cv2.cvtColor(cond, cv2.COLOR_BGR2RGB)
    cond_gray = cv2.cvtColor(cond, cv2.COLOR_RGB2GRAY)

    target = cv2.imread(target_file)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    target_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)


    gen = cv2.imread(gen_file)
    gen = cv2.cvtColor(gen, cv2.COLOR_BGR2RGB)
    gen_gray = cv2.cvtColor(gen, cv2.COLOR_RGB2GRAY)



    overlap_region = np.zeros_like(cond_gray)
    for i in range(cond_gray.shape[0]):
        for j in range(cond_gray.shape[1]):
            if cond_gray[i][j] != 0 or target_gray[i][j] != 0:
                overlap_region[i][j] = 255


    target_setting = np.zeros_like(target_gray) 
    cond_setting = np.zeros_like(cond_gray)
    gen_setting = np.zeros_like(gen_gray)


    for i in range(overlap_region.shape[0]):
        for j in range(overlap_region.shape[1]):
            if overlap_region[i][j] == 255:
                target_setting[i][j] = target_gray[i][j]
                cond_setting[i][j] = cond_gray[i][j]
                gen_setting[i][j] = gen_gray[i][j]


    threshold = 80
    invalid_count = 0
    valid_count = 0

    label_map = np.zeros_like(cond_setting)
    gen_pred_map = np.zeros_like(cond_setting)

    for i in range(gen_setting.shape[0]):
        for j in range(gen_setting.shape[1]):
            if cond_setting[i][j] > target_setting[i][j] :
                invalid_count += 1
            else:
                valid_count += 1   
                label_map[i][j] = 0 if target_setting[i][j] - cond_setting[i][j] < threshold else 1

                if gen_setting[i][j]  < cond_setting[i][j]:
                    gen_pred_map[i][j] = 1
                elif gen_setting[i][j] - cond_setting[i][j] < threshold:
                    gen_pred_map[i][j] = 0
                else:
                    gen_pred_map[i][j] = 1


    correct_count_inside = 0
    region_count = 0    
    for i in range(overlap_region.shape[0]):
        for j in range(overlap_region.shape[1]):
            if overlap_region[i][j] == 255:
                region_count += 1
                if label_map[i][j] == gen_pred_map[i][j]:
                    correct_count_inside += 1

                

    psnr_score = psnr(target_gray, gen_gray)
    ssim_score = ssim(target_gray, gen_gray)
    accuracy_inside = correct_count_inside/region_count


    valid_ratio = valid_count/(valid_count + invalid_count) 

    return  psnr_score, ssim_score, accuracy_inside, valid_ratio
    
    

def evaluation(cond_path, target_path, gen_path):
    cond_files = os.listdir(cond_path)
    target_files = os.listdir(target_path)
    gen_files = os.listdir(gen_path)

    cond_files.sort(key=lambda x: int(x.split('.')[0]))
    target_files.sort(key=lambda x: int(x.split('.')[0]))
    gen_files.sort(key=lambda x: int(x.split('.')[0]))

    total_psnr = 0
    total_ssim = 0
    total_accuracy = 0
    total_valid_ratio = 0

    for idx in tqdm(range(len(cond_files))):
        cond_file = os.path.join(cond_path, cond_files[idx])
        target_file = os.path.join(target_path, target_files[idx])
        gen_file = os.path.join(gen_path, gen_files[idx])

        psnr_score, ssim_score, accuracy_inside, valid_ratio= _evaluation(cond_file, target_file, gen_file)
        
        total_psnr += psnr_score
        total_ssim += ssim_score
        total_accuracy += accuracy_inside
        total_valid_ratio += valid_ratio

    total_psnr /= len(cond_files)
    total_ssim /= len(cond_files)
    total_accuracy /= len(cond_files)
    total_valid_ratio /= len(cond_files)    

    return total_psnr, total_ssim, total_accuracy, total_valid_ratio


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Usage: python evaluation.py <cond_path> <target_path> <gen_path>")
        sys.exit(1)
    
    cond_path = sys.argv[1]
    target_path = sys.argv[2]
    gen_path = sys.argv[3]

    assert len(os.listdir(cond_path)) == len(os.listdir(target_path)) == len(os.listdir(gen_path))

    psnr_score, ssim_score, accuracy_inside, valid_ratio = evaluation(cond_path, target_path, gen_path)

    print(f"average psnr: {psnr_score}")
    print(f"average ssim: {ssim_score}")
    print(f"average accuracy_inside: {accuracy_inside}")
    print(f"average valid_ratio: {valid_ratio}")

