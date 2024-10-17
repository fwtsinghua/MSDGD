
from model.MLPSynthesizer import MLPSynthesizer
from model.BaseDiffusion import BaseDiffuser
from model.TransSynthesizer import TransSynthesizer
from model.UnetSynthesizer import UnetSynthesizer

from datetime import datetime
import numpy as np
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm



def  train_diffusion_model(dataloader,
                          synthesizer_model, diffuser_model, device='cpu',
                          epochs = 10, learning_rate = 1e-4):

    # Init optimizer, scheduler and loss function.
    parameters = filter(lambda p: p.requires_grad,
                        synthesizer_model.parameters())  # determine synthesizer model parameters
    optimizer = optim.Adam(parameters, lr=learning_rate)  # init Adam optimizer
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs, verbose=False)  # init learning rate scheduler
    loss_fnc = nn.MSELoss() # init mean-squared-error loss
    loss_fnc = loss_fnc.to(device)


    # Train the model
    train_epoch_losses = []  # init collection of training epoch losses
    synthesizer_model.train()  # set the model in training mode
    pbar = tqdm(iterable=range(epochs), position=0, leave=True)  # init the training progress bar
    for epoch in pbar:  # iterate over training epochs
        batch_losses = []  # init epoch training batch losses
        for rows_tensor, label_tensor in dataloader:  # iterate over epoch batches
            rows_tensor= rows_tensor.to(device)  # send batch to device
            label_tensor = label_tensor.to(device)  # send batch to device
            timesteps = diffuser_model.sample_random_timesteps(n=rows_tensor.shape[0])  # sample diffusion timestep


            # add diffuser gaussian noise
            batch_noise_t, noise_t = diffuser_model.add_gauss_noise(x_num=rows_tensor,  t=timesteps)

            # # spatial_dimensional_transformation
            if 'Unet' in synthesizer_model.model_type:
                from scripts.SDTR import spatial_dimensional_transformation
                noise_t = spatial_dimensional_transformation(noise_t, part_size=noise_t.size(0)//synthesizer_model.channels)


            predicted_noise = synthesizer_model(batch_noise_t, timesteps=timesteps, class_labels=label_tensor)

            batch_loss = loss_fnc(input=noise_t, target=predicted_noise)  # compute training batch loss
            optimizer.zero_grad()  # reset model gradients
            batch_loss.backward()  # run model backward pass
            optimizer.step()  # optimize model parameters

            batch_losses.append(batch_loss.detach().cpu().numpy())  # collect training batch losses
        batch_losses_mean = np.mean(np.array(batch_losses))  # determine mean training epoch loss
        lr_scheduler.step()  # update learning rate scheduler
        train_epoch_losses.append(batch_losses_mean)  # collect mean training epoch loss

        # prepare and set training epoch progress bar update
        now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        pbar.set_description(
            '[LOG {}] epoch: {}, train-loss: {}'.format(str(now), str(epoch).zfill(4), str(batch_losses_mean)))
    return train_epoch_losses


import json
import torch

# 保存模型超参数和训练后的模型权重到 JSON 和 pth 文件
def save_model_and_params(synthesizer_model, diffuser_model, model_params_file, model_weights_file):
    # 保存超参数到 JSON 文件
    model_params = {
        "synthesizer_params": synthesizer_model.get_params(),
        "diffuser_params": diffuser_model.get_params()
    }
    with open(model_params_file, 'w') as f:
        json.dump(model_params, f)

    # 保存模型权重到 pth 文件
    torch.save({
        'synthesizer_state_dict': synthesizer_model.state_dict(),
        'diffuser_state_dict': None
    }, model_weights_file)


# 重新加载超参数和创建模型
def load_model_and_params(model_params_file, model_weights_file, device='cpu'):
    with open(model_params_file, 'r') as f:
        model_params = json.load(f)

    if model_params["synthesizer_params"]["model_type"] == "MLP":
        synthesizer_model = MLPSynthesizer(**model_params["synthesizer_params"])
    elif model_params["synthesizer_params"]["model_type"] =="Transformer":
        synthesizer_model = TransSynthesizer(**model_params["synthesizer_params"])
    elif model_params["synthesizer_params"]["model_type"] =="Unet":
        synthesizer_model = UnetSynthesizer(**model_params["synthesizer_params"])

    diffuser_model = BaseDiffuser(**model_params["diffuser_params"], device=device)

    # 加载模型权重
    checkpoint = torch.load(model_weights_file, map_location=device)
    synthesizer_model.load_state_dict(checkpoint['synthesizer_state_dict'])

    return synthesizer_model, diffuser_model



