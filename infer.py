#License
#--------------------------------------------------------------------------------
# MIT License

# Copyright (c) 2022 Machine Vision and Learning Group, LMU Munich

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#--------------------------------------------------------------------------------
# Modified from latent-diffusion (https://github.com/CompVis/latent-diffusion)
#--------------------------------------------------------------------------------



import os
import sys
import cv2
import time
import torch
import torchvision
import numpy as np
from tqdm import tqdm


from einops import rearrange, repeat
from PIL import Image
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config, ismap
from ldm.models.diffusion.ddim import DDIMSampler

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return {"model": model}, global_step


def get_model(path_conf, path_ckpt):
    # path_conf = "/mnt/sdb1/audi/diffusion/latent-diffusion/logs/2025-03-14T09-58-55_CT_sim_SSL/configs/2025-03-14T09-58-55-project.yaml"
    # path_ckpt = "/mnt/sdb1/audi/diffusion/latent-diffusion/logs/2025-03-14T09-58-55_CT_sim_SSL/checkpoints/epoch=000010.ckpt" 
    config = OmegaConf.load(path_conf)
    model, step = load_model_from_config(config, path_ckpt)
    return model



def get_cond(selected_path):
    example = dict()

    c = Image.open(selected_path)
    if c.mode != 'RGB':
        c = c.convert('RGB')
    c = np.array(c).astype(np.uint8)

    

    detected_map = cv2.Canny(c, 50, 150)


    detected_map = detected_map[:, :, None]
    detected_map = np.concatenate([detected_map, detected_map, detected_map], axis=2)
    detected_map = (detected_map/127.5 - 1.0).astype(np.float32)

    c_up = c

    c = (c/127.5 - 1.0).astype(np.float32)
    c_up = (c_up/127.5 - 1.0).astype(np.float32)
    c = torch.unsqueeze(torchvision.transforms.ToTensor()(c), 0)
    c_up = torch.unsqueeze(torchvision.transforms.ToTensor()(c_up), 0)
    c = rearrange(c, '1 c h w -> 1 h w c')
    c_up = rearrange(c_up, '1 c h w -> 1 h w c')



    c = c.to(torch.device("cuda"))
    c_up = c_up.to(torch.device("cuda"))    


    example["hint"] = c  #for testing HU_syn_b + SSL
    example["LR_image"] = c #for testing HU_syn
    example["image"] = c_up #not used in inference

    return example



def run(model, selected_path, custom_steps, resize_enabled=False, classifier_ckpt=None, global_step=None):

    example = get_cond(selected_path)
    save_intermediate_vid = False
    n_runs = 1
    masked = False
    guider = None
    ckwargs = None
    mode = 'ddim'
    ddim_use_x0_pred = False
    temperature = 1.
    eta = 1.
    make_progrow = True
    custom_shape = None

    height, width = example["image"].shape[1:3]
    print(height,width)
 

    invert_mask = False

    x_T = None
    for n in range(n_runs):
        if custom_shape is not None:
            x_T = torch.randn(1, custom_shape[1], custom_shape[2], custom_shape[3]).to(model.device)
            x_T = repeat(x_T, '1 c h w -> b c h w', b=custom_shape[0])

        
        logs = make_convolutional_sample(example, model,
                                         mode=mode, custom_steps=custom_steps,
                                         eta=eta, swap_mode=False , masked=masked,
                                         invert_mask=invert_mask, quantize_x0=False,
                                         custom_schedule=None, decode_interval=10,
                                         resize_enabled=resize_enabled, custom_shape=custom_shape,
                                         temperature=temperature, noise_dropout=0.,
                                         corrector=guider, corrector_kwargs=ckwargs, x_T=x_T, save_intermediate_vid=save_intermediate_vid,
                                         make_progrow=make_progrow,ddim_use_x0_pred=ddim_use_x0_pred
                                         )
    return logs


@torch.no_grad()
def convsample_ddim(model, cond, steps, shape, eta=1.0, callback=None, normals_sequence=None,
                    mask=None, x0=None, quantize_x0=False, img_callback=None,
                    temperature=1., noise_dropout=0., score_corrector=None,
                    corrector_kwargs=None, x_T=None, log_every_t=None
                    ):

    ddim = DDIMSampler(model)
    bs = shape[0]  # dont know where this comes from but wayne
    shape = shape[1:]  # cut batch dim
    print(f"Sampling with eta = {eta}; steps: {steps}")
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, conditioning=cond, callback=callback,
                                         normals_sequence=normals_sequence, quantize_x0=quantize_x0, eta=eta,
                                         mask=mask, x0=x0, temperature=temperature, verbose=False,
                                         score_corrector=score_corrector,
                                         corrector_kwargs=corrector_kwargs, x_T=x_T)

    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(batch, model, mode="vanilla", custom_steps=None, eta=1.0, swap_mode=False, masked=False,
                              invert_mask=True, quantize_x0=False, custom_schedule=None, decode_interval=1000,
                              resize_enabled=False, custom_shape=None, temperature=1., noise_dropout=0., corrector=None,
                              corrector_kwargs=None, x_T=None, save_intermediate_vid=False, make_progrow=True,ddim_use_x0_pred=False):
    log = dict()
    
    z, c, x, xrec, xc = model.get_input(batch, model.first_stage_key,
                                        return_first_stage_outputs=True,
                                        force_c_encode=not (hasattr(model, 'split_input_params')
                                                            and model.cond_stage_key == 'coordinates_bbox'),
                                        return_original_cond=True)


    log_every_t = 1 if save_intermediate_vid else None

    if custom_shape is not None:
        z = torch.randn(custom_shape)
        print(f"Generating {custom_shape[0]} samples of shape {custom_shape[1:]}")

    z0 = None

    log["input"] = x
    log["reconstruction"] = xrec

    if ismap(xc):
        log["original_conditioning"] = model.to_rgb(xc)
        if hasattr(model, 'cond_stage_key'):
            log[model.cond_stage_key] = model.to_rgb(xc)

    else:
        log["original_conditioning"] = xc if xc is not None else torch.zeros_like(x)
        if model.cond_stage_model:
            log[model.cond_stage_key] = xc if xc is not None else torch.zeros_like(x)
            if model.cond_stage_key =='class_label':
                log[model.cond_stage_key] = xc[model.cond_stage_key]


    t0 = time.time()
    img_cb = None
    sample, intermediates = convsample_ddim(model, c, steps=custom_steps, shape=z.shape,
                                            eta=eta,
                                            quantize_x0=quantize_x0, img_callback=img_cb, mask=None, x0=z0,
                                            temperature=temperature, noise_dropout=noise_dropout,
                                            score_corrector=corrector, corrector_kwargs=corrector_kwargs,
                                            x_T=x_T, log_every_t=log_every_t)
    t1 = time.time()

    if ddim_use_x0_pred:
        sample = intermediates['pred_x0'][-1]

    x_sample = model.decode_first_stage(sample)
   

    try:
        x_sample_noquant = model.decode_first_stage(sample, force_not_quantize=True)
        log["sample_noquant"] = x_sample_noquant
        log["sample_diff"] = torch.abs(x_sample_noquant - x_sample)
    except:
        pass

    log["sample"] = x_sample
    log["time"] = t1 - t0

    return log


def infer(src, config_path=None, model_path=None, custom_steps=200):
    if model_path is None:
        print("Model path is not provided")
        sys.exit(1)
    if config_path is None:
        print("Config path is not provided")
        sys.exit(1)
    model = get_model(config_path, model_path)
    
    save_path = src + "_infer" 
    os.makedirs(save_path, exist_ok=True)
     
    files = os.listdir(src)
    files.sort(key=lambda x: int(x.split('.')[0]))
    
    for idx in tqdm(range(len(files))):
        selected_path = os.path.join(src, files[idx])
        logs = run(model["model"], selected_path,  custom_steps)
        sample = logs["sample"]
        sample = sample.detach().cpu()
        sample = torch.clamp(sample, -1., 1.)
        sample = (sample + 1.) / 2. * 255
        sample = sample.numpy().astype(np.uint8)
        sample = np.transpose(sample, (0, 2, 3, 1))
        img = Image.fromarray(sample[0])
        img.save(os.path.join(save_path, f"{idx}.png"))

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python infer.py <src> <config_path> <model_path>")
        sys.exit(1)

    src = sys.argv[1]
    config_path = sys.argv[2]
    model_path = sys.argv[3]
    infer(src, config_path, model_path)