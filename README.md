# SSLDM: Semi-Supervised Learning with Latent Diffusion Models for 2D CT Contrast Enhanced Slice Generation
The implementation of the College Student Research Scholarship, NSC plan.
The main goal of this project is to generate 2D CT images form images without intravenous contrast agent injection to images with intravenous contrast agent injection. We provide two settings of the evaluation in Section 4 of the report. The implementation based on the Pytorch framework of the work [latent-diffusion](<https://github.com/CompVis/latent-diffusion>).

## environment
```
conda env create -f environment.yml
conda activate SSLDM
```
## Dataset
In this task, paired source dataset with the same size is prepared, the first one contains images with intravenous contrast agent injection, and the second one contains images without intravenous contrast agent injection. The dataset should fowllow the format of naming as follows:
```
dataset
├──0.png
├──1.png
├──2.png
├──3.png
...
```
## HU-based trasnformation upon the  LDM
### LDM
Please refer to the referecd work [latent-diffusion](<https://github.com/CompVis/latent-diffusion>) to get familiar with the LDM model and get the pretrained weights of the first stage encoder.
### Dataset
In this setting, we only need the dataset converted under the HU-based transformation from the source non-iv dataset and iv-dataset and then use it to replace the original non-contrast dataset in the SSLDM model. You can use the `Dataset_convert.py`  and refer the config `configs/CT_sim.yaml` datasets postfix to get the details about converting dataset.
### Training
```
CUDA_VISIBLE_DEVICES=0 python main.py --base configs/CT_sim.yaml -t --gpus 0,
```
## SSLDM with  labeled data and unlabeled data
### Dataset
In the setting, we need the dataset with partition of labeled data and unlabeled data. We also need the label data to train the condtiton extractor. You can use the `Dataset_convert.py` and refer the config `configs/CT_sim_SSL.yaml` datasets postfix to get all the needed dataset converted from the original two datasets.

### Condiction exracter and pretrained weights
In the setting, we need the pretrained weights of the HU-based trasnformation upon the  LDM and the following  condiction extractor. Then copy them into the SSLDM model as the initial weights by specifying in the config `configs/CT_sim_SSL.yaml`. 
```
python train_cond_extra.py configs/CT_sim_cond_extracter.yaml <log_path>
```
### Training
```
CUDA_VISIBLE_DEVICES=0 python main.py --base configs/CT_sim_SSL.yaml -t --gpus 0,
```
## Inference
To generate the 2D CT slices from the trained model, you can use the following command:
```
python infer.py <src> <config_path> <model_path>
```
## Evaluation
To evaluate the generated images, you can use the following command. The evaluation metrics include PSNR, SSIM, and our proposed metric.
```
python evaluation.py <cond_path> <target_path> <gen_path>
````