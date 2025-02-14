import gc
from typing import Any, Dict, Optional, Tuple, Union
import torch
import math
import argparse
import os

from PIL import Image
from diffusers import FluxPipeline,RfSolverFluxPipeline, RfSolverFluxTransformer2DModel
from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader
import logging

from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from datasets import get_dataloader

from utils.utils import *
from utils.metrics import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

logger = logging.getLogger(__name__)   # pylint: disable=invalid-name


@torch.inference_mode()
def interpolated_inversion(
    pipeline, 
    latents,
    DTYPE,
    joint_attention_kwargs,
    num_steps=1,
    use_shift_t_sampling=True, 
    source_prompt="",
    guidance_scale = 1.0
):

    # 源文本提示
    prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
        prompt=source_prompt, 
        prompt_2=source_prompt
    )
    # 准备潜变量图像ID
    latent_image_ids = pipeline._prepare_latent_image_ids(
        latents.shape[0],
        latents.shape[2],
        latents.shape[3],
        latents.device, 
        DTYPE,
    )
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
    
    # 使用插值速度场进行图像反演
    with pipeline.progress_bar(total=len(timesteps)-1) as progress_bar:
        for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            t_vec = torch.full((packed_latents.shape[0],), t_curr, dtype=packed_latents.dtype, device=packed_latents.device)

            joint_attention_kwargs['t'] = t_prev 
            joint_attention_kwargs['inverse'] = True
            joint_attention_kwargs['second_order'] = False
            joint_attention_kwargs['inject'] = inject_list[i]
            #print("joint_attention_kwargs id:",id(joint_attention_kwargs))
            # 计算速度
            pred ,joint_attention_kwargs = pipeline.transformer(
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

            packed_latents_mid = packed_latents + (t_prev - t_curr) / 2 * pred

            t_vec_mid = torch.full((packed_latents.shape[0],), (t_curr + (t_prev - t_curr) / 2), dtype=packed_latents.dtype, device=packed_latents.device)
            joint_attention_kwargs['second_order'] = True
            #计算2阶速度
            pred_mid ,joint_attention_kwargs= pipeline.transformer(
                    hidden_states=packed_latents_mid,
                    timestep=t_vec_mid,
                    guidance=guidance_vec,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=joint_attention_kwargs,  
                    return_dict=pipeline,
                )
            pred_mid=pred_mid[0]

            first_order = (pred_mid - pred) / ((t_prev - t_curr) / 2)

            # 防止精度问题
            packed_latents = packed_latents.to(torch.float32)
            packed_latents_mid = packed_latents_mid.to(torch.float32)
            pred = pred.to(torch.float32)
            pred_mid = pred_mid.to(torch.float32)

            # 更新潜变量
            packed_latents = packed_latents + (t_prev - t_curr) * pred +0.5* (t_prev - t_curr) ** 2 * first_order
            
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
def interpolated_denoise(
    pipeline, 
    joint_attention_kwargs,
    inversed_latents,            # 如果不使用反转潜变量，可以为 None
    use_inversed_latents=True,
    guidance_scale=4.0,
    target_prompt='photo of a tiger',
    DTYPE=torch.bfloat16,
    num_steps=1,
    use_shift_t_sampling=True, 
    img_latents=None,
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


    # 使用插值速度场进行去噪
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
            


            packed_latents_mid = packed_latents + (t_prev - t_curr) / 2 * pred

            t_vec_mid = torch.full((packed_latents.shape[0],), (t_curr + (t_prev - t_curr) / 2), dtype=packed_latents.dtype, device=packed_latents.device)
            joint_attention_kwargs['second_order'] = True
            #计算2阶速度
            pred_mid ,joint_attention_kwargs= pipeline.transformer(
                    hidden_states=packed_latents_mid,
                    timestep=t_vec_mid,
                    guidance=guidance_vec,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=joint_attention_kwargs,  
                    return_dict=pipeline,
                )
            pred_mid=pred_mid[0]
            
            first_order = (pred_mid - pred) / ((t_prev - t_curr) / 2)

            # 防止精度问题
            packed_latents = packed_latents.to(torch.float32)
            packed_latents_mid = packed_latents_mid.to(torch.float32)
            pred = pred.to(torch.float32)
            pred_mid = pred_mid.to(torch.float32)

            # 更新潜变量
            packed_latents = packed_latents + (t_prev - t_curr) * pred + 0.5*(t_prev - t_curr) ** 2 * first_order
            
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
def main(args):

    if args.dtype == 'bfloat16':
        DTYPE = torch.bfloat16
    elif args.dtype == 'float16':
        DTYPE = torch.float16
    elif args.dtype == 'float32':
        DTYPE = torch.float32
    else:
        raise ValueError(f"不支持的数据类型: {args.dtype}")

    joint_attention_kwargs = {}
    joint_attention_kwargs['feature_path'] = args.feature_path
    joint_attention_kwargs['feature'] = {}
    joint_attention_kwargs['inject_step'] = args.inject

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.feature_path, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu"

    metrics = metircs()

    # ******** Loading pipeline **********
    pipe = RfSolverFluxPipeline.from_pretrained(args.model_path, torch_dtype=DTYPE)
    pipe.to(device)
    print(pipe.hf_device_map)
    #pipe.enable_model_cpu_offload()
    #pipe.enable_sequential_cpu_offload()

    # ******** Input processing **********
    if args.eval_datasets == '':
        img = Image.open(args.image_path)
        train_transforms = transforms.Compose(
                    [
                        transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),
                        transforms.CenterCrop(1024),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                    ]
                )

        img = train_transforms(img).unsqueeze(0)
        dataloader = [img, args.source_prompt, args.target_prompt]
    else:
        dataset = get_dataloader(args.eval_datasets)
        dataloader = DataLoader(
            dataset,
            batch_size=1,          # 每批64个样本
            shuffle=False,           # 训练时打乱数据
            num_workers=8,          # 使用4个子进程加载数据
            pin_memory=True         # 如果使用GPU，可以加速数据传输
        )

    # ******** evaluation **********
    mean_clip_score = 0
    count = 0
    for img, source_prompt, target_prompt in dataloader:
        img = img.to(device).to(DTYPE)
        # vae encode
        img_latent = encode_imgs(img, pipe, DTYPE)

        if True:
            # 进行插值反演
            inversed_latent ,joint_attention_kwargs = interpolated_inversion(
                pipe, 
                img_latent, 
                DTYPE=DTYPE, 
                num_steps=args.num_steps, 
                use_shift_t_sampling=True,
                source_prompt=source_prompt,
                guidance_scale = 1,
                joint_attention_kwargs=joint_attention_kwargs)    
        else:
            inversed_latent = None

        # 进行去噪
        img_latents,joint_attention_kwargs = interpolated_denoise(
            pipe, 
            inversed_latents=inversed_latent,
            use_inversed_latents=args.use_inversed_latents,
            joint_attention_kwargs=joint_attention_kwargs,
            guidance_scale=args.guidance_scale,
            target_prompt=target_prompt,
            DTYPE=DTYPE,
            num_steps=args.num_steps,
            use_shift_t_sampling=True,
            img_latents=img_latent
        )

        # 将潜变量解码为图像
        out = decode_imgs(img_latents, pipe)[0]

        # evaluation
        clip_score = metrics.clip_scores(target_prompt, out)
        print(f"==> clip score: {clip_score:.4f}")
        mean_clip_score += clip_score

        count += 1

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

    print('######### Evaluation Results ###########')
    mean_clip_score = mean_clip_score / count
    print(f"==> clip score: {mean_clip_score:.4f}")

    # 显式删除不再需要的变量
    pipe.maybe_free_model_hooks()
    del out, img_latent, img_latents, inversed_latent, pipe, img, joint_attention_kwargs

    # 强制进行垃圾收集
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='使用不同参数测试 interpolated_denoise。')
    parser.add_argument('--model_path', type=str, default='/root/autodl-tmp/Flux-dev', help='预训练模型的路径')
    parser.add_argument('--image_path', type=str, default='./example/image.png', help='输入图像的路径')
    parser.add_argument('--eval-datasets', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='outputs', help='保存输出图像的目录')
    parser.add_argument('--use_inversed_latents', action='store_true', help='使用反转潜变量')
    parser.add_argument('--guidance_scale', type=float, default=3.5, help='interpolated_denoise 的引导比例')
    parser.add_argument('--num-steps', type=int, default=30, help='时间步长的数量')
    parser.add_argument('--shift', action='store_true', help='在 get_schedule 中使用 shift')

    parser.add_argument('--source-prompt', type=str,
                        help='describe the content of the source image (or leaves it as null)')
    parser.add_argument('--target-prompt', type=str,
                        help='describe the requirement of editing')

    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16', 'float32'], help='计算的数据类型')
    
    parser.add_argument('--feature_path', type=str, default='features',
                        help='the path to save the feature ')
    parser.add_argument('--inject', type=int, default=5,
                        help='the number of timesteps which apply the feature sharing')
    
    args = parser.parse_args()
    main(args)
