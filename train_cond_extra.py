import torch
import sys
import os
import numpy as np
import einops
from tqdm import tqdm
from ldm.util import instantiate_from_config
from torch.utils.data import DataLoader
from data.CT_sim_cond import CT_cond_extra
from omegaconf import OmegaConf


@torch.no_grad()
def encode_first_stage(first_stage_model, cond):
    return first_stage_model.encode(cond)


@torch.no_grad()
def decode_first_stage(first_stage_model, z):
    return first_stage_model.decode(z, False)


def get_latent(model, cond):
    posterior = encode_first_stage(model, cond)
    z = posterior.detach().requires_grad_(True)
    return z

def get_pred(model, cond):
    posterior = decode_first_stage(model, cond)
    z = posterior.detach().requires_grad_(True)
    return z

def train(model, train_loader, optimizer, criterion):
    model.train()
    loss_epoch = 0
    for _, batch in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        cond = batch['image']
        target = batch['label']

    
        cond = einops.rearrange(cond, 'b h w c -> b c h w ')
        cond = cond.cuda()

        output = model(cond)
        target = target.cuda()
        loss = criterion(output, target)


        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()


    loss_epoch /= len(train_loader)
    return loss_epoch
    
def validate(model, val_loader, criterion):
    model.eval()
    loss_epoch = 0
    with torch.no_grad():
        for _, batch in enumerate(tqdm(val_loader)):
            cond = batch['image']
            target = batch['label']

            cond = einops.rearrange(cond, 'b h w c -> b c h w')
            cond = cond.cuda()

            output = model(cond)
            target = target.cuda()
            loss = criterion(output, target)

            loss_epoch += loss.item()

    loss_epoch /= len(val_loader)
    return loss_epoch

if __name__ == "__main__":

    """
    model config is input argument argc, argc
    """
    config_path = sys.argv[1]  
    log_path = sys.argv[2]
    if len(sys.argv) != 3:
        print("Usage: python train_cond_extra.py <config_path> <log_path>")
        sys.exit(1)


    '''
    produce train/val set and train/val label set
    based on the src noiv data set A
    1. Use Dataset convert -p to get A with r ratio, call A_r
    2. Use Dataset convert -to get label set of A_r, call A_r_cond2label
    3. Use Dataset convert to get generate result of A_r, call A_r_infer
    '''
    # trainset_path = "/mnt/sdb1/audi/diffusion/latent-diffusion/data/manual_heart_iv_4seg_train_match_5_gen"
    # trainset_label_path = "/mnt/sdb1/audi/diffusion/latent-diffusion/data/manual_heart_iv_4seg_train_5_cond2label"
    # valset_path = "/mnt/sdb1/audi/diffusion/latent-diffusion/data/manual_heart_iv_4seg_val_match_5_gen"
    # valset_label_path = "/mnt/sdb1/audi/diffusion/latent-diffusion/data/manual_heart_iv_4seg_val_5_cond2label"

    trainset_path = None
    trainset_label_path = None
    valset_path = None
    valset_label_path = None

    assert trainset_path is not None
    assert trainset_label_path is not None
    assert valset_path is not None
    assert valset_label_path is not None

    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config)


    print(model)


    model = model.to('cuda')

    trainset = CT_cond_extra(trainset_path, trainset_label_path)
    valset = CT_cond_extra(valset_path, valset_label_path)

    train_loader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(valset, batch_size=4, shuffle=False, num_workers=0)

    max_epochs = 1000
    learning_rate = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()


    loss_dict = {}
    loss_dict['train'] = []
    loss_dict['val'] = []

    min_val_loss = 1e6
    min_val_loss2 = 1e7


    ctr = 0
    for epoch in range(max_epochs):
        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss = validate(model, val_loader, criterion)

        print(f"Epoch {epoch}, Train Loss: {train_loss}, Validation Loss: {val_loss}")
        loss_dict['train'].append(train_loss)
        loss_dict['val'].append(val_loss)


        if val_loss < min_val_loss:
            torch.save(model.state_dict(), log_path + '/best_model.pth')
            min_val_loss = val_loss
            ctr = 0
        elif val_loss < min_val_loss2:
            torch.save(model.state_dict(), log_path + '/best_model2.pth')
            min_val_loss2 = val_loss
            ctr = 0
        else:
            ctr += 1
        if ctr > 25:
            break


    
