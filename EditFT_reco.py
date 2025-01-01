import gc
from typing import Any, Dict, Optional, Tuple, Union
import torch
import math
import argparse
import os

from PIL import Image
from diffusers import FluxPipeline,RfSolverFluxPipeline
from torch import Tensor
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)   # pylint: disable=invalid-name

# 使用推理模式装饰器，禁用梯度计算以提高推理速度和节省内存
@torch.inference_mode()
def decode_imgs(latents, pipeline):
    # 解码潜变量为图像
    imgs = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    imgs = pipeline.vae.decode(imgs)[0]
    imgs = pipeline.image_processor.postprocess(imgs, output_type="pil")
    return imgs

@torch.inference_mode()
def encode_imgs(imgs, pipeline, DTYPE):
    # 编码图像为潜变量
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











#--------------------继承并重写 diffusers.models.attention_processor.FluxAttnProcessor2_0 的 __call__ 方法
from diffusers.models.attention_processor import FluxAttnProcessor2_0
import torch.nn.functional as F
import torch.nn as nn


#--------------------继承并重写 diffusers.models.attention_processor.Attention 类
from diffusers.models.attention_processor import Attention
import inspect


#--------------------继承并重写 diffusers.models.transformer.FluxSingleTransformerBlock 类-----------------
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel, FluxSingleTransformerBlock





#--------------------继承并重写 diffusers.models.transformer.FluxTransformerBlock 类-----------------
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock





#--------------------继承并重写 diffusers.models.transformer.FluxTransformer2DModel 类-------------------

#/data/chx/FLUX.1-dev/transformer/config.json 中"_class_name": "FluxTransformer2DModel", 
# 改为"_class_name": "CustomFluxTransformer2DModel"

#/data/chx/FLUX.1-dev/model_index.json 中"_class_name": "FluxPipeline" 
# 改为"_class_name": "CustomFluxPipeline" 
# "FluxTransformer2DModel" 改为 "CustomFluxTransformer2DModel"


import numpy as np
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.transformers.transformer_flux import RfSolverFluxTransformer2DModel


#--------------------继承并重写 diffusers.FluxPipeline 类-------------------------------------------------
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast














@torch.inference_mode()
def interpolated_inversion(
    pipeline, 
    latents,
    DTYPE,
    joint_attention_kwargs,
    num_steps=28,
    use_shift_t_sampling=True, 
    source_prompt="",
    guidance_scale = 1.0,
    epsilon_t_dict= {}
):


    # 源文本提示
    prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
        prompt=source_prompt, 
        prompt_2=source_prompt
    )
    #print("latents", latents.shape)
    # 准备潜变量图像ID
    latent_image_ids = pipeline._prepare_latent_image_ids(
        latents.shape[0],
        latents.shape[2],
        latents.shape[3],
        latents.device, 
        DTYPE,
    )
    #print("latent_image_ids", latent_image_ids.shape)

    # 打包潜变量
    packed_latents = pipeline._pack_latents(
        latents,
        batch_size=latents.shape[0],
        num_channels_latents=latents.shape[1],
        height=latents.shape[2],
        width=latents.shape[3],
    )
    # 获取时间步长调度表
    
    timesteps = get_schedule( 
                num_steps=num_steps,
                image_seq_len=(packed_latents.shape[1] ), # vae_scale_factor = 16
                shift=use_shift_t_sampling,
            )
    
    
    
    # 准备指导向量
    guidance_vec = torch.full((packed_latents.shape[0],), guidance_scale, device=packed_latents.device, dtype=packed_latents.dtype)

    inject_list = [True] * joint_attention_kwargs['inject_step'] + [False] * (len(timesteps[:-1]) - joint_attention_kwargs['inject_step'])
    #反演过程反转调度表和inject_list
    timesteps = timesteps[::-1]
    inject_list = inject_list[::-1]

#-------------------------EditTF算法1--------------------------------------------
    I = 3  # 设置迭代次数

    # 存储反演轨迹和误差项
    x_t_dict = {}
    x_t_dict[timesteps[0]] = packed_latents.clone()
    
    print("Stage I: Fixed-point Inversion")
    with pipeline.progress_bar(total=len(timesteps)-1) as progress_bar:
        for idx, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            t_vec = torch.full((packed_latents.shape[0],), t_curr, dtype=packed_latents.dtype, device=packed_latents.device)
            # 获取对应的 sigma_t 和 sigma_{t-1}
            sigma_t = t_curr
            sigma_t_prev = t_prev

            # 第一阶段：固定点反转
            # 初始化 x_{t-1}^{0} = x_t
            x_t_minus_1_i = packed_latents.clone()
            x_t_minus_1_i_list = []

            for i in range(I):
                # 计算速度 v_theta(x_{t-1}^{i-1}, t-1)
                t_vec_prev = torch.full((x_t_minus_1_i.shape[0],), t_prev, dtype=x_t_minus_1_i.dtype, device=x_t_minus_1_i.device)
                joint_attention_kwargs['t'] = t_prev
                joint_attention_kwargs['inverse'] = True
                joint_attention_kwargs['second_order'] = False
                joint_attention_kwargs['inject'] = inject_list[idx]

                pred, joint_attention_kwargs = pipeline.transformer(
                    hidden_states=x_t_minus_1_i,
                    timestep=t_vec_prev,
                    guidance=guidance_vec,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=pipeline,
                )
                pred = pred[0]

                # 防止精度问题
                x_t_minus_1_i = x_t_minus_1_i.to(torch.float32)
                pred = pred.to(torch.float32)
                packed_latents = packed_latents.to(torch.float32)               

                # 更新 x_{t-1}^{i}
                x_t_minus_1_i = packed_latents + (sigma_t_prev - sigma_t) * pred 
                x_t_minus_1_i_list.append(x_t_minus_1_i)
                x_t_minus_1_i = x_t_minus_1_i.to(DTYPE)

            # 计算 x_{t-1} 为平均值
            x_t_minus_1 = sum(x_t_minus_1_i_list) / I

            # 存储 x_{t-1}
            x_t_minus_1 = x_t_minus_1.to(DTYPE)
            x_t_dict[t_prev] = x_t_minus_1

            # 更新 packed_latents 为 x_{t-1}
            packed_latents = x_t_minus_1

            packed_latents = packed_latents.to(DTYPE)
            progress_bar.update()

    # 速度补偿前反转timesteps
    timesteps = timesteps[::-1]

    # 第二阶段：速度补偿
    print("Stage II: Velocity Compensation")
    with pipeline.progress_bar(total=len(timesteps)-1) as progress_bar:
        for idx in range(len(timesteps)-1):
            t_curr = timesteps[idx]
            t_next = timesteps[idx+1]
            x_t_current = x_t_dict[t_curr]
            x_t_next = x_t_dict[t_next]
            t_vec = torch.full((x_t_current.shape[0],), t_curr, dtype=x_t_current.dtype, device=x_t_current.device)
            # 获取对应的 sigma_t 和 sigma_{t+1}
            sigma_t = t_curr
            sigma_t_next = t_next

            # 计算速度 v_theta(x_t, t)
            joint_attention_kwargs['t'] = t_curr
            joint_attention_kwargs['inverse'] = True
            joint_attention_kwargs['second_order'] = False
            joint_attention_kwargs['inject'] = inject_list[idx]

            pred, joint_attention_kwargs = pipeline.transformer(
                hidden_states=x_t_current,
                timestep=t_vec,
                guidance=guidance_vec,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=joint_attention_kwargs,
                return_dict=pipeline,
            )
            pred = pred[0]

            # 计算 \hat{x}_{t+1}
            x_hat_t_plus_1 = x_t_current + (sigma_t_next - sigma_t) * pred

            # 计算 ε_t
            epsilon_t = x_t_next - x_hat_t_plus_1

            # 存储 ε_t
            epsilon_t_dict[t_next] = epsilon_t

            progress_bar.update()
            
     
    
    # 解包潜变量
    latents = pipeline._unpack_latents(
            packed_latents,
            height=1024,
            width=1024,
            vae_scale_factor=pipeline.vae_scale_factor,
    )
    latents = latents.to(DTYPE)
    return latents ,joint_attention_kwargs, epsilon_t_dict



@torch.inference_mode()
def interpolated_denoise(
    pipeline, 
    joint_attention_kwargs,
    inversed_latents,            # 如果不使用反转潜变量，可以为 None
    use_inversed_latents=True,
    guidance_scale=4.0,
    target_prompt='photo of a tiger',
    DTYPE=torch.bfloat16,
    num_steps=28,
    use_shift_t_sampling=True, 
    epsilon_t_dict=None
):


    # 编码提示文本
    prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
        prompt=target_prompt, 
        prompt_2=target_prompt
    )

    # 准备潜变量图像ID
    latent_image_ids = pipeline._prepare_latent_image_ids(
        inversed_latents.shape[0],
        inversed_latents.shape[2],
        inversed_latents.shape[3],
        inversed_latents.device,
        DTYPE,
    )

    if use_inversed_latents:
        # 使用反转潜变量
        packed_latents = pipeline._pack_latents(
            inversed_latents,
            batch_size=inversed_latents.shape[0],
            num_channels_latents=inversed_latents.shape[1],
            height=inversed_latents.shape[2],
            width=inversed_latents.shape[3],
        )
    else:
        # 生成随机潜变量
        tmp_latents = torch.randn_like(img_latents)
        packed_latents = pipeline._pack_latents(
            tmp_latents,
            batch_size=tmp_latents.shape[0],
            num_channels_latents=tmp_latents.shape[1],
            height=tmp_latents.shape[2],
            width=tmp_latents.shape[3],
        )


    # 获取时间步长调度表
    timesteps = get_schedule( 
                num_steps=num_steps,
                image_seq_len=(packed_latents.shape[1] ), # vae_scale_factor = 16
                shift=use_shift_t_sampling,
            )
    

    guidance_vec = torch.full((packed_latents.shape[0],), guidance_scale, device=packed_latents.device, dtype=packed_latents.dtype)
    inject_list = [True] * joint_attention_kwargs['inject_step'] + [False] * (len(timesteps[:-1]) - joint_attention_kwargs['inject_step'])


    # 进行去噪
    print("Denoising")
    with pipeline.progress_bar(total=len(timesteps)-1) as progress_bar:
        for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            t_vec = torch.full((packed_latents.shape[0],), t_curr, dtype=packed_latents.dtype, device=packed_latents.device)
            
            joint_attention_kwargs['t'] = t_curr 
            joint_attention_kwargs['inverse'] = False
            joint_attention_kwargs['second_order'] = False
            joint_attention_kwargs['inject'] = inject_list[i]

            # 计算速度
            pred,joint_attention_kwargs = pipeline.transformer(
                    hidden_states=packed_latents,
                    timestep=t_vec,
                    guidance=guidance_vec,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=joint_attention_kwargs,  #TODO:此处可以传递inject的相关参数，详细仍需再研究，考虑形如 joint_attention_kwargs['inject'] = inject_list[i]  24/11/19 修改到此
                    return_dict=pipeline,
                )
            pred=pred[0]
            
            # 获取对应的 epsilon_t
            epsilon_t = epsilon_t_dict.get(t_prev, torch.zeros_like(packed_latents))

            # 防止精度问题
            packed_latents = packed_latents.to(torch.float32)
            pred = pred.to(torch.float32)
            epsilon_t=epsilon_t.to(torch.float32)

            # 更新潜变量
            packed_latents = packed_latents + (t_prev - t_curr) * pred + epsilon_t

            packed_latents = packed_latents.to(DTYPE)
            progress_bar.update()
    
    # 解包潜变量
    latents = pipeline._unpack_latents(
            packed_latents,
            height=1024,
            width=1024,
            vae_scale_factor=pipeline.vae_scale_factor,
    )
    latents = latents.to(DTYPE)
    return latents ,joint_attention_kwargs

@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description='使用不同参数测试 interpolated_denoise。')
    parser.add_argument('--model_path', type=str, default='/root/autodl-tmp/Flux-dev', help='预训练模型的路径')
    parser.add_argument('--image_path', type=str, default='./example/cat.png', help='输入图像的路径')
    parser.add_argument('--output_dir', type=str, default='outputs', help='保存输出图像的目录')
    parser.add_argument('--use_inversed_latents', action='store_true', help='使用反转潜变量')
    parser.add_argument('--guidance_scale', type=float, default=3.5, help='interpolated_denoise 的引导比例')
    parser.add_argument('--num_steps', type=int, default=28, help='时间步长的数量')
    parser.add_argument('--shift', action='store_true', help='在 get_schedule 中使用 shift')

    parser.add_argument('--source_prompt', type=str,
                        help='describe the content of the source image (or leaves it as null)')
    parser.add_argument('--target_prompt', type=str,
                        help='describe the requirement of editing')

    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16', 'float32'], help='计算的数据类型')
    
    parser.add_argument('--feature_path', type=str, default='feature',
                        help='the path to save the feature ')
    parser.add_argument('--inject', type=int, default=5,
                        help='the number of timesteps which apply the feature sharing')
    
    args = parser.parse_args()

    if args.dtype == 'bfloat16':
        DTYPE = torch.bfloat16
    elif args.dtype == 'float16':
        DTYPE = torch.float16
    elif args.dtype == 'float32':
        DTYPE = torch.float32
    else:
        raise ValueError(f"不支持的数据类型: {args.dtype}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    #/data/chx/FLUX.1-dev/transformer/config.json 中"_class_name": "FluxTransformer2DModel", 
    # 改为"_class_name": "CustomFluxTransformer2DModel"

    #/data/chx/FLUX.1-dev/model_index.json 中"_class_name": "FluxPipeline" 
    # 改为"_class_name": "CustomFluxPipeline" 
    # "FluxTransformer2DModel" 改为 "CustomFluxTransformer2DModel"
    pipe = RfSolverFluxPipeline.from_pretrained(args.model_path, torch_dtype=DTYPE)
    pipe.enable_sequential_cpu_offload()
    #pipe.enable_model_cpu_offload()

    # 如果不存在则创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    joint_attention_kwargs = {}
    joint_attention_kwargs['feature_path'] = args.feature_path
    joint_attention_kwargs['feature'] = {}
    joint_attention_kwargs['inject_step'] = args.inject
    epsilon_t_dict ={}
    if not os.path.exists(args.feature_path):
        os.mkdir(args.feature_path)

    # 加载并预处理图像
    img = Image.open(args.image_path)

    train_transforms = transforms.Compose(
                [
                    transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(1024),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

    img = train_transforms(img).unsqueeze(0).to(device).to(DTYPE)
    #print("img shape", img.shape)
    # 将图像编码为潜变量
    img_latent = encode_imgs(img, pipe, DTYPE)
    #print("img_la", img_latent.shape)
    if args.use_inversed_latents:
        # 进行插值反演
        inversed_latent ,joint_attention_kwargs,epsilon_t_dict = interpolated_inversion(
            pipe, 
            img_latent, 
            DTYPE=DTYPE, 
            num_steps=args.num_steps, 
            use_shift_t_sampling=True,
            source_prompt=args.source_prompt,
            guidance_scale = args.guidance_scale,
            joint_attention_kwargs=joint_attention_kwargs,
            epsilon_t_dict=epsilon_t_dict)    
    else:
        inversed_latent = None

    # 进行去噪
    img_latents,joint_attention_kwargs = interpolated_denoise(
        pipe, 
        inversed_latents=inversed_latent,
        use_inversed_latents=args.use_inversed_latents,
        joint_attention_kwargs=joint_attention_kwargs,
        guidance_scale=args.guidance_scale,
        target_prompt=args.target_prompt,
        DTYPE=DTYPE,
        num_steps=args.num_steps,
        use_shift_t_sampling=True,
        epsilon_t_dict=epsilon_t_dict
    )

    # 将潜变量解码为图像
    out = decode_imgs(img_latents, pipe)[0]

    # 保存输出图像
    output_filename = f"num_steps{args.num_steps}_inject{args.inject}_inversed{args.use_inversed_latents}_guidance{args.guidance_scale}.png"
    output_path = os.path.join(args.output_dir, output_filename)
    
    # 如果文件重名则自动加序号
    base, ext = os.path.splitext(output_path)
    counter = 1
    while os.path.exists(output_path):
        output_path = f"{base}_{counter}{ext}"
        counter += 1
    out.save(output_path)
    print(f"已保存输出图像到 {output_path}，参数为:num_steps={args.num_steps} inject={args.inject} inversed={args.use_inversed_latents} guidance_scale={args.guidance_scale}")

    # 显式删除不再需要的变量
    del out, img_latent, img_latents, inversed_latent, pipe, img, joint_attention_kwargs

    # 强制进行垃圾收集
    gc.collect()

if __name__ == "__main__":
    main()