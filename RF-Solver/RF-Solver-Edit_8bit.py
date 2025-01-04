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
import logging

from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

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

# class CustomFluxAttnProcessor(FluxAttnProcessor2_0):
#     def __call__(
#         self,
#         attn,
#         hidden_states: torch.FloatTensor,
#         encoder_hidden_states: torch.FloatTensor = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         image_rotary_emb: Optional[torch.Tensor] = None,
#         id: Optional[int] = None,
#         inject: Optional[bool] = False,
#         feature: Optional[Dict[str, Any]] = None,
#         t: Optional[int] = None,
#         second_order: Optional[bool] = False,
#         inverse: Optional[bool] = False,

#     ) -> torch.FloatTensor:
#         batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

#         # `sample` projections.
#         query = attn.to_q(hidden_states)
#         key = attn.to_k(hidden_states)
#         value = attn.to_v(hidden_states)

#         inner_dim = key.shape[-1]
#         head_dim = inner_dim // attn.heads

#         query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#         key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#         value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

#         if attn.norm_q is not None:
#             query = attn.norm_q(query)
#         if attn.norm_k is not None:
#             key = attn.norm_k(key)

#         # Save the features in the memory 此处进行Value的特征保存和注入
#         if inject and id > 19:
#             feature_name = str(t) + '_' + str(second_order) + '_' + str(id) + '_' + type + '_' + 'V'
#             if inverse:
#                 feature[feature_name] = value.cpu()
#             else:
#                 value =feature[feature_name].cuda()           

#         # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
#         if encoder_hidden_states is not None:
#             # `context` projections.
#             encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
#             encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
#             encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

#             encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
#                 batch_size, -1, attn.heads, head_dim
#             ).transpose(1, 2)
#             encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
#                 batch_size, -1, attn.heads, head_dim
#             ).transpose(1, 2)
#             encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
#                 batch_size, -1, attn.heads, head_dim
#             ).transpose(1, 2)

#             if attn.norm_added_q is not None:
#                 encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
#             if attn.norm_added_k is not None:
#                 encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

#             # attention
#             query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
#             key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
#             value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

#         if image_rotary_emb is not None:
#             from diffusers.models.embeddings import apply_rotary_emb

#             query = apply_rotary_emb(query, image_rotary_emb)
#             key = apply_rotary_emb(key, image_rotary_emb)

#         hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
#         hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
#         hidden_states = hidden_states.to(query.dtype)

#         if encoder_hidden_states is not None:
#             encoder_hidden_states, hidden_states = (
#                 hidden_states[:, : encoder_hidden_states.shape[1]],
#                 hidden_states[:, encoder_hidden_states.shape[1] :],
#             )

#             # linear proj
#             hidden_states = attn.to_out[0](hidden_states)
#             # dropout
#             hidden_states = attn.to_out[1](hidden_states)
#             encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

#             return hidden_states, encoder_hidden_states
#         else:
#             return hidden_states

#---------------------------------------------------------------------------------------------------------

#--------------------继承并重写 diffusers.models.attention_processor.Attention 类
from diffusers.models.attention_processor import Attention
import inspect

# class CustomAttention(Attention):
#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         **cross_attention_kwargs,
#     ) -> torch.Tensor:
#         r"""
#         The forward method of the `Attention` class.

#         Args:
#             hidden_states (`torch.Tensor`):
#                 The hidden states of the query.
#             encoder_hidden_states (`torch.Tensor`, *optional*):
#                 The hidden states of the encoder.
#             attention_mask (`torch.Tensor`, *optional*):
#                 The attention mask to use. If `None`, no mask is applied.
#             **cross_attention_kwargs:
#                 Additional keyword arguments to pass along to the cross attention.

#         Returns:
#             `torch.Tensor`: The output of the attention layer.
#         """
#         # The `Attention` class can call different attention processors / attention functions
#         # here we simply pass along all tensors to the selected processor class
#         # For standard processors that are defined here, `**cross_attention_kwargs` is empty

#         attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
#         quiet_attn_parameters = {"ip_adapter_masks", "id", "inject", "feature", "t", "second_order", "inverse"}
#         unused_kwargs = [
#             k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters
#         ]
#         if len(unused_kwargs) > 0:
#             logger.warning(
#                 f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
#             )
#         cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}

#         return self.processor(
#             self,
#             hidden_states,
#             encoder_hidden_states=encoder_hidden_states,
#             attention_mask=attention_mask,
#             **cross_attention_kwargs,
#         )

#---------------------------------------------------------------------------------------------------------


#--------------------继承并重写 diffusers.models.transformer.FluxSingleTransformerBlock 类-----------------
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel, FluxSingleTransformerBlock

# class CustomFluxSingleTransformerBlock(FluxSingleTransformerBlock):
#     def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=4.0):
#         super().__init__(dim, num_attention_heads, attention_head_dim, mlp_ratio)
#         self.attn = CustomAttention(
#             query_dim=dim,
#             cross_attention_dim=None,
#             dim_head=attention_head_dim,
#             heads=num_attention_heads,
#             out_dim=dim,
#             bias=True,
#             processor=CustomFluxAttnProcessor(),
#             qk_norm="rms_norm",
#             eps=1e-6,
#             pre_only=True,
#         )
#--------------------------------------------------------------------------------------------------------



#--------------------继承并重写 diffusers.models.transformer.FluxTransformerBlock 类-----------------
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock
# class CustomFluxTransformerBlock(FluxTransformerBlock):
#     def __init__(self, dim, num_attention_heads, attention_head_dim, qk_norm="rms_norm", eps=1e-6):
#         super().__init__(dim, num_attention_heads, attention_head_dim, qk_norm, eps)
#         processor = FluxAttnProcessor2_0()
#         self.attn = CustomAttention2(
#             query_dim=dim,
#             cross_attention_dim=None,
#             added_kv_proj_dim=dim,
#             dim_head=attention_head_dim,
#             heads=num_attention_heads,
#             out_dim=dim,
#             context_pre_only=False,
#             bias=True,
#             processor=processor,
#             qk_norm=qk_norm,
#             eps=eps,
#         )

#--------------------------------------------------------------------------------------------------------




#--------------------继承并重写 diffusers.models.transformer.FluxTransformer2DModel 类-------------------

#/data/chx/FLUX.1-dev/transformer/config.json 中"_class_name": "FluxTransformer2DModel", 
# 改为"_class_name": "CustomFluxTransformer2DModel"

#/data/chx/FLUX.1-dev/model_index.json 中"_class_name": "FluxPipeline" 
# 改为"_class_name": "CustomFluxPipeline" 
# "FluxTransformer2DModel" 改为 "CustomFluxTransformer2DModel"


import numpy as np
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers

# class CustomFluxTransformer2DModel(FluxTransformer2DModel):
#     _no_split_modules = ["CustomFluxTransformerBlock", "CustomFluxSingleTransformerBlock"]
#     def __init__(
#         self,
#         patch_size: int = 1,
#         in_channels: int = 64,
#         num_layers: int = 19,
#         num_single_layers: int = 38,
#         attention_head_dim: int = 128,
#         num_attention_heads: int = 24,
#         joint_attention_dim: int = 4096,
#         pooled_projection_dim: int = 768,
#         guidance_embeds: bool = False,
#         axes_dims_rope: Tuple[int] = (16, 56, 56),
#     ):
#         super().__init__(
#             patch_size,
#             in_channels,
#             num_layers,
#             num_single_layers,
#             attention_head_dim,
#             num_attention_heads,
#             joint_attention_dim,
#             pooled_projection_dim,
#             guidance_embeds,
#             axes_dims_rope,
#         )

#         self.transformer_blocks = nn.ModuleList(
#             [
#                 CustomFluxTransformerBlock(
#                     dim=self.inner_dim,
#                     num_attention_heads=self.config.num_attention_heads,
#                     attention_head_dim=self.config.attention_head_dim,
#                 )
#                 for i in range(self.config.num_layers)
#             ]
#         )

#         self.single_transformer_blocks = nn.ModuleList(
#             [
#                 CustomFluxSingleTransformerBlock(
#                     dim=self.inner_dim,
#                     num_attention_heads=self.config.num_attention_heads,
#                     attention_head_dim=self.config.attention_head_dim,
#                 )
#                 for _ in range(self.config.num_single_layers)
#             ]
#         )





#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         encoder_hidden_states: torch.Tensor = None,
#         pooled_projections: torch.Tensor = None,
#         timestep: torch.LongTensor = None,
#         img_ids: torch.Tensor = None,
#         txt_ids: torch.Tensor = None,
#         guidance: torch.Tensor = None,
#         joint_attention_kwargs: Optional[Dict[str, Any]] = None,
#         controlnet_block_samples=None,
#         controlnet_single_block_samples=None,
#         return_dict: bool = True,
#         controlnet_blocks_repeat: bool = False,
#     ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
#         """
#         The [`FluxTransformer2DModel`] forward method.

#         Args:
#             hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
#                 Input `hidden_states`.
#             encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
#                 Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
#             pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
#                 from the embeddings of input conditions.
#             timestep ( `torch.LongTensor`):
#                 Used to indicate denoising step.
#             block_controlnet_hidden_states: (`list` of `torch.Tensor`):
#                 A list of tensors that if specified are added to the residuals of transformer blocks.
#             joint_attention_kwargs (`dict`, *optional*):
#                 A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
#                 `self.processor` in
#                 [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
#             return_dict (`bool`, *optional*, defaults to `True`):
#                 Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
#                 tuple.

#         Returns:
#             If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
#             `tuple` where the first element is the sample tensor.
#         """
#         if joint_attention_kwargs is not None:
#             logger.warning(
#                     "test1"
#                 )
#             #joint_attention_kwargs = joint_attention_kwargs.copy() 此处传递副本改为传递引用

#             lora_scale = joint_attention_kwargs.pop("scale", 1.0)
#         else:
#             lora_scale = 1.0

#         if USE_PEFT_BACKEND:
#             # weight the lora layers by setting `lora_scale` for each PEFT layer
#             scale_lora_layers(self, lora_scale)
#         else:
#             if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
#                 logger.warning(
#                     "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
#                 )
#         hidden_states = self.x_embedder(hidden_states)

#         timestep = timestep.to(hidden_states.dtype) * 1000
#         if guidance is not None:
#             guidance = guidance.to(hidden_states.dtype) * 1000
#         else:
#             guidance = None
#         temb = (
#             self.time_text_embed(timestep, pooled_projections)
#             if guidance is None
#             else self.time_text_embed(timestep, guidance, pooled_projections)
#         )
#         encoder_hidden_states = self.context_embedder(encoder_hidden_states)

#         if txt_ids.ndim == 3:
#             logger.warning(
#                 "Passing `txt_ids` 3d torch.Tensor is deprecated."
#                 "Please remove the batch dimension and pass it as a 2d torch Tensor"
#             )
#             txt_ids = txt_ids[0]
#         if img_ids.ndim == 3:
#             logger.warning(
#                 "Passing `img_ids` 3d torch.Tensor is deprecated."
#                 "Please remove the batch dimension and pass it as a 2d torch Tensor"
#             )
#             img_ids = img_ids[0]

#         ids = torch.cat((txt_ids, img_ids), dim=0)
#         image_rotary_emb = self.pos_embed(ids)

#         for index_block, block in enumerate(self.transformer_blocks):
#             if self.training and self.gradient_checkpointing:

#                 def create_custom_forward(module, return_dict=None):
#                     def custom_forward(*inputs):
#                         if return_dict is not None:
#                             return module(*inputs, return_dict=return_dict)
#                         else:
#                             return module(*inputs)

#                     return custom_forward

#                 ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
#                 encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
#                     create_custom_forward(block),
#                     hidden_states,
#                     encoder_hidden_states,
#                     temb,
#                     image_rotary_emb,
#                     **ckpt_kwargs,
#                 )

#             else:
#                 encoder_hidden_states, hidden_states = block(
#                     hidden_states=hidden_states,
#                     encoder_hidden_states=encoder_hidden_states,
#                     temb=temb,
#                     image_rotary_emb=image_rotary_emb,
#                     joint_attention_kwargs=joint_attention_kwargs,
#                 )

#             # controlnet residual
#             if controlnet_block_samples is not None:
#                 interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
#                 interval_control = int(np.ceil(interval_control))
#                 # For Xlabs ControlNet.
#                 if controlnet_blocks_repeat:
#                     hidden_states = (
#                         hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
#                     )
#                 else:
#                     hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
        
#         cnt = 0

#         hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

#         joint_attention_kwargs['type'] = 'single'

#         for index_block, block in enumerate(self.single_transformer_blocks):
#             if self.training and self.gradient_checkpointing:

#                 def create_custom_forward(module, return_dict=None):
#                     def custom_forward(*inputs):
#                         if return_dict is not None:
#                             return module(*inputs, return_dict=return_dict)
#                         else:
#                             return module(*inputs)

#                     return custom_forward

#                 ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
#                 hidden_states = torch.utils.checkpoint.checkpoint(
#                     create_custom_forward(block),
#                     hidden_states,
#                     temb,
#                     image_rotary_emb,
#                     **ckpt_kwargs,
#                 )

#             else:
#                 joint_attention_kwargs['id'] = cnt
#                 hidden_states = block(
#                     hidden_states=hidden_states,
#                     temb=temb,
#                     image_rotary_emb=image_rotary_emb,
#                     joint_attention_kwargs=joint_attention_kwargs,
#                 )
#                 cnt += 1

#             # controlnet residual
#             if controlnet_single_block_samples is not None:
#                 interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
#                 interval_control = int(np.ceil(interval_control))
#                 hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
#                     hidden_states[:, encoder_hidden_states.shape[1] :, ...]
#                     + controlnet_single_block_samples[index_block // interval_control]
#                 )

#         hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

#         hidden_states = self.norm_out(hidden_states, temb)
#         output = self.proj_out(hidden_states)

#         if USE_PEFT_BACKEND:
#             # remove `lora_scale` from each PEFT layer
#             unscale_lora_layers(self, lora_scale)

#         if not return_dict:
#             return (output,)

#         return Transformer2DModelOutput(sample=output)
#--------------------------------------------------------------------------------------------------------

#--------------------继承并重写 diffusers.FluxPipeline 类-------------------------------------------------
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

# class CustomFluxPipeline(FluxPipeline):
#     def __init__(
#         self,
#         scheduler: FlowMatchEulerDiscreteScheduler,
#         vae: AutoencoderKL,
#         text_encoder: CLIPTextModel,
#         tokenizer: CLIPTokenizer,
#         text_encoder_2: T5EncoderModel,
#         tokenizer_2: T5TokenizerFast,
#         transformer: CustomFluxTransformer2DModel,
#     ):
#         super().__init__(
#             scheduler=scheduler,
#             vae=vae,
#             text_encoder=text_encoder,
#             tokenizer=tokenizer,
#             text_encoder_2=text_encoder_2,
#             tokenizer_2=tokenizer_2,
#             transformer=transformer,
#         )
    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
    #     # 调用父类的 from_pretrained 方法
    #     pipeline = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        
    #     # 替换 transformer 为 CustomFluxTransformer2DModel
    #     custom_transformer = CustomFluxTransformer2DModel(
    #         patch_size=pipeline.transformer.config.patch_size,
    #         in_channels=pipeline.transformer.config.in_channels,
    #         num_layers=pipeline.transformer.config.num_layers,
    #         num_single_layers=pipeline.transformer.config.num_single_layers,
    #         attention_head_dim=pipeline.transformer.config.attention_head_dim,
    #         num_attention_heads=pipeline.transformer.config.num_attention_heads,
    #         joint_attention_dim=pipeline.transformer.config.joint_attention_dim,
    #         pooled_projection_dim=pipeline.transformer.config.pooled_projection_dim,
    #         guidance_embeds=pipeline.transformer.config.guidance_embeds,
    #         axes_dims_rope=pipeline.transformer.config.axes_dims_rope,
    #     )
    #     pipeline.transformer = custom_transformer
        
    #     return pipeline

#--------------------------------------------------------------------------------------------------------















@torch.inference_mode()
def interpolated_inversion(
    pipeline, 
    latents,
    DTYPE,
    joint_attention_kwargs,
    num_steps=28,
    use_shift_t_sampling=True, 
    source_prompt="",
    guidance_scale = 1.0
):


    # 源文本提示
    prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
        prompt=source_prompt, 
        prompt_2=source_prompt
    )
    print("latents", latents.shape)
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

    #pipeline.to('cpu')
    
    
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
            pred ,joint_attention_kwargs= pipeline.transformer(
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
            #print("joint_attention_kwargs id:",id(joint_attention_kwargs))


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

    #pipeline.transformer.to('cpu')        
    
    
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
    num_steps=28,
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
    #device = "cpu"


    quant_config = TransformersBitsAndBytesConfig(load_in_8bit=True,)
    transformer_8bit = RfSolverFluxTransformer2DModel.from_pretrained(
        args.model_path,
        subfolder="transformer",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
    )

    pipe = RfSolverFluxPipeline.from_pretrained(args.model_path, torch_dtype=DTYPE,transformer=transformer_8bit)
    print(pipe.hf_device_map)
    pipe.enable_model_cpu_offload()
    #pipe.enable_sequential_cpu_offload()

    # 如果不存在则创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    joint_attention_kwargs = {}
    joint_attention_kwargs['feature_path'] = args.feature_path
    joint_attention_kwargs['feature'] = {}
    joint_attention_kwargs['inject_step'] = args.inject
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
    #if args.use_inversed_latents:
    if True:
        # 进行插值反演
        inversed_latent ,joint_attention_kwargs = interpolated_inversion(
            pipe, 
            img_latent, 
            DTYPE=DTYPE, 
            num_steps=args.num_steps, 
            use_shift_t_sampling=True,
            source_prompt=args.source_prompt,
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
        target_prompt=args.target_prompt,
        DTYPE=DTYPE,
        num_steps=args.num_steps,
        use_shift_t_sampling=True,
        img_latents=img_latent
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
    pipe.maybe_free_model_hooks()
    del out, img_latent, img_latents, inversed_latent, pipe, img, joint_attention_kwargs

    # 强制进行垃圾收集
    gc.collect()

if __name__ == "__main__":
    main()