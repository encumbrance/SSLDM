import torch
import random
import numpy as np
import lpips


from einops import rearrange, repeat
from torchvision.utils import make_grid
from skimage.util import random_noise

from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import  default, instantiate_from_config
from ldm.modules.diffusionmodules.util import  extract_into_tensor



'''
implement cutout to the edge_map tensor with size b, c, h, w
'''
def cutout(edge_map, size=64):
    b, c, h, w = edge_map.shape
    mask_color = 0
    #mask_percentage between 0.25 and 0.75
    mask_percentage = torch.rand(1).item() * 0.5 + 0.5

    #mask percentage of the center of the image pixels
    x_idx_list = torch.randint(h//4, 3*h//4, (int(mask_percentage*h),))
    y_idx_list = torch.randint(w//4, 3*w//4, (int(mask_percentage*w),))
    for x_idx in x_idx_list:
        for y_idx in y_idx_list:
            edge_map[:, :, x_idx, y_idx] = mask_color


    return edge_map

def zoom_in(edge_map):
    #randomly crop the image to 0.75 ~ 1.0 of the original h and w
    #and then resize it back to the original size
    b, c, h, w = edge_map.shape
    crop_percentage = torch.rand(1).item() * 0 + 1.0
    crop_h = int(crop_percentage * h)
    crop_w = int(crop_percentage * w)
    x_start = torch.randint(0, h - crop_h, (1,)).item()
    y_start = torch.randint(0, w - crop_w, (1,)).item()

    edge_map = edge_map[:, :, x_start:x_start+crop_h, y_start:y_start+crop_w]
    edge_map = torch.nn.functional.interpolate(edge_map, (h, w), mode='nearest')
    return edge_map, crop_percentage, x_start, y_start


def speckle_noise(edge_map, scale=0.01):
    noise_edge_map = random_noise(edge_map, mode='speckle', var=scale)
    return noise_edge_map
    
 

class SSLDM(LatentDiffusion):
    def __init__(self, cond_extracter_config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cond_extracter = instantiate_from_config(cond_extracter_config)
        self.lpips = lpips.LPIPS(net='alex')
        self.lpips = self.lpips.to(self.device)
        self.lpips.eval()
        self.lpips.requires_grad_(False)
        self.trajectory = None
        self.recon_student = None
        self.recon_teacher = None
        self.label = None
        


   
    
    def p_losses(self, x_start, x_t, cond, t, label, xc, noise=None):
        """
        split the input x_start into first half and second half
        with b, c, h, w to b/2, c, h, w and b/2, c, h, w
        """

        x_start_u = x_start[:x_start.shape[0]//2]
        x_start_l = x_start[x_start.shape[0]//2:]

        t_u = t[:t.shape[0]//2]
        t_l = t[t.shape[0]//2:]

        cond_u = cond[:cond.shape[0]//2]
        cond_l = cond[cond.shape[0]//2:]

        label_u = label[:label.shape[0]//2]
        label_u = label_u.to(torch.long)
        label_l = label[label.shape[0]//2:]
        label_l = label_l.to(torch.long)


        self.label = repeat(label, 'b h w -> b c h w', c=1)


        noise = default(noise, lambda: torch.randn_like(x_start_l))
        x_noisy = self.q_sample(x_start=x_start_l, t=t_l, noise=noise)
        model_output_l = self.apply_model(x_noisy, t_l, cond_l)


        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()
        

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        loss_l = self.get_loss(model_output_l, target, mean=False).mean(dim=[1, 2, 3]).mean()
        loss_dict.update({f'{prefix}/loss_l': loss_l})

        noise = default(noise, lambda: torch.randn_like(x_start_u))
        x_noisy_s = self.q_sample(x_start=x_start_u, t=t_u, noise=noise)
        x_noisy_t = self.q_sample(x_start=x_t, t=t_u, noise=noise)
       

        model_output_u = self.apply_model(x_noisy_s, t_u, cond_u)
        model_output_t = self.teacher_pred(x_noisy_t, t_u, cond_u)

        loss_u = self.get_loss(model_output_u, model_output_t, mean=False).mean(dim=[1, 2, 3]).mean()
        loss_dict.update({f'{prefix}/loss_u': loss_u})

        loss_u_simple = self.get_loss(model_output_u, noise, mean=False).mean(dim=[1, 2, 3]).mean()
        loss_dict.update({f'{prefix}/loss_u_simple': loss_u_simple})


        recon = self.predict_start_from_noise(x_noisy_s, t_u, model_output_u)
        recon_teacher = self.predict_start_from_noise(x_noisy_t, t_u, model_output_t)


        loss_u = self.get_loss(recon, recon_teacher, mean=False).mean(dim=[1, 2, 3]).mean()
        loss_dict.update({f'{prefix}/loss_u': loss_u})

        teacher_decoded = self.decode_first_stage(recon_teacher)
        self.trajectory = teacher_decoded
       
        student_decoded = self.decode_first_stage(recon)
        self.recon_student = student_decoded

        
        weight = extract_into_tensor(self.sqrt_alphas_cumprod, t_u, student_decoded.shape)

        pred_label_u = self.cond_extracter(student_decoded)
        loss_structural_u = torch.nn.functional.cross_entropy(pred_label_u, label_u, reduction='none')
        loss_structural_u = (loss_structural_u * weight).mean()
        loss_dict.update({f'{prefix}/loss_structural_u': loss_structural_u})

        loss_lpips = self.lpips.forward(student_decoded, xc[:xc.shape[0]//2])
        loss_lpips = (loss_lpips * weight).mean()
        loss_dict.update({f'{prefix}/loss_lpips': loss_lpips})


        lambda_l = 1.0
        lambda_u = 1.0
        lambda_u_simple = 0.5
        lambda_s = 1e-2
        lambda_v = 1e-2


        loss = lambda_l * loss_l + lambda_u * loss_u + lambda_u_simple * loss_u_simple + lambda_s * loss_structural_u + lambda_v * loss_lpips
        loss_dict.update({f'{prefix}/loss': loss})
        return loss, loss_dict
        


    def shared_step(self, batch, **kwargs):
        x, c, xc = self.get_input(batch, self.first_stage_key, return_original_cond=True)
        _, label = self.get_input(batch, self.first_stage_key, cond_key="label")
        _, x_t = self.get_input(batch, self.first_stage_key, cond_key="image_teach")
        x_t = self.encode_first_stage(x_t)
        loss = self(x, x_t, c, label, xc)
        return loss
    
    def forward(self, x, x_t, c, label, xc, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
        return self.p_losses(x, x_t, c, t, label, xc, *args, **kwargs)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_val = self.shared_step(batch)
        self.log_dict(loss_dict_val, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    
    @torch.no_grad()
    def teacher_pred(self, x, t, c):
        with self.ema_scope("Teacher Prediction"):
            pred = self.apply_model(x, t, c)
        return pred
    

    @torch.no_grad()
    def log_images(self, batch,  N=4, n_row=2, sample=True, ddim_steps=50, ddim_eta=1.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=1., unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c, xc = self.get_input(batch, self.first_stage_key, bs=N, force_c_encode=True, return_original_cond=True)
        _, x_t = self.get_input(batch, self.first_stage_key, cond_key="image_teach")
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["reconstruction_teacher"] = x_t
        log["label"] = self.label * 2.0 - 1.0
        log["conditioning"] = (xc)
        log["ODE_sol"] = self.trajectory
        log["recon_student"] = self.recon_student
        

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond=c,
                                                    batch_size=N, ddim=use_ddim,
                                                    ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        return log


    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        params = params + list(self.cond_stage_model.parameters())
        params = params + list(self.cond_extracter.parameters())

        print(f"params: {len(params)}")
        opt = torch.optim.AdamW(params, lr=lr)

        return opt
        

    # def get_learned_conditioning(self, c):
    #     flag = False
    #     if c.shape[0] == 1:
    #         c = repeat(c, '1 c h w -> b c h w', b=2)
    #         flag = True
    #     pred = self.cond_extracter(c[:c.shape[0]//2])
    #     pred = torch.argmax(pred, dim=1)
    #     self.pred = pred - 1
    #     pred = (pred - 1) * 2
    #     pred = repeat(pred, 'b h w -> b c h w', c=c.shape[1])
    #     c[:c.shape[0]//2] =(pred + c[:c.shape[0]//2]).clamp(-1, 1)
    #     if flag:
    #         # get the first one with size 1 c h w
    #         c = c[0:1]
    #         print(f"get_learned_conditioning: {c.shape}")
    #     self.cond_extracted = c
    #     c = self.cond_stage_model(c)
    #     return c
       