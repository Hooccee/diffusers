import math

import torch
from torch import Tensor

@torch.inference_mode()
def decode_imgs(latents, pipeline):
    # vae image encoding
    imgs = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    img_tensor = pipeline.vae.decode(imgs)[0]
    imgs = pipeline.image_processor.postprocess(img_tensor, output_type="pil")
    return imgs

@torch.inference_mode()
def encode_imgs(imgs, pipeline, DTYPE):
    # vae image decoding
    latents = pipeline.vae.encode(imgs).latent_dist.sample()
    latents = (latents - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor
    latents = latents.to(dtype=DTYPE)
    return latents


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list:
    # 生成时间步长调度表
    def get_lin_function(
        x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
    ):
        # 计算线性函数的斜率和截距
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b

    def time_shift(mu: float, sigma: float, t: Tensor):
        # 计算时间偏移
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
    
    # 生成从1到0的线性时间步长
    timesteps = torch.linspace(1, 0, num_steps + 1, dtype=torch.float32)

    if shift:
        # 基于图像序列长度估计mu
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


