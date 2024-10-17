from datetime import datetime

import pandas as pd
import torch
from tqdm import tqdm


def generate_diffusion_model(num_columns, num_scaler,
                             synthesizer_model, diffuser_model, device='cpu', num_class_embeds=10,
                             n_samples=100,batch_size=96 ,sample_diffusion_steps=10, spatial_dimensional_transformation_inverse=False):
    """
    :param num_columns: number of columns
    :param num_scaler: QuantileTransformer Object
    :param column_names: names of columns
    :param synthesizer_model:
    :param diffuser_model:
    :param n_samples: number of samples to generate
    :param sample_diffusion_steps: number of diffusion steps
    :return:
    """
    samples_list = []
    for i in range(n_samples//batch_size):
        print(f"Generating {i*batch_size} to {(i+1)*batch_size} samples")

        samples = torch.randn((batch_size, num_columns))  # init samples to be generated
        samples = samples.to(device)  # send samples to device
        labels = torch.randint(0, num_class_embeds, (batch_size,))  # init class labels
        labels = labels.to(device)


        pbar = tqdm(iterable=reversed(range(0, sample_diffusion_steps)), position=0, leave=True)
        for diffusion_step in pbar:  # iterate over diffusion steps
            now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')  # prepare and set training epoch progress bar update
            pbar.set_description('[LOG {}] Sample Diffusion Step: {}'.format(str(now), str(diffusion_step).zfill(4)))
            timesteps = torch.full((batch_size,), diffusion_step, dtype=torch.long, device=device)  # init diffusion timesteps
            model_out = synthesizer_model(samples, timesteps=timesteps, class_labels=labels)  # run synthesizer model forward pass
            if 'Unet' in synthesizer_model.model_type:
                from scripts.SDTR import spatial_dimensional_transformation_inverse
                model_out = spatial_dimensional_transformation_inverse(model_out)
            samples = diffuser_model.p_sample_gauss(model_out, samples, timesteps)  # run diffuser model forward pass

        samples_list.append(samples.detach().cpu())

    # denormalize numeric attributes
    samples = torch.cat(samples_list, dim=0)
    z_norm_upscaled = num_scaler.inverse_transform(samples.numpy())  # denormalize generated samples



    return z_norm_upscaled