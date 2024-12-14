# coding=utf-8
# Copyright 2024 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import random
import unittest

import torch

from diffusers import IFImg2ImgSuperResolutionPipeline
from diffusers.models.attention_processor import AttnAddedKVProcessor
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.testing_utils import (
    floats_tensor,
    load_numpy,
    require_accelerator,
    require_torch_gpu,
    skip_mps,
    slow,
    torch_device,
)

from ..pipeline_params import (
    TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_PARAMS,
)
from ..test_pipelines_common import PipelineTesterMixin, assert_mean_pixel_difference
from . import IFPipelineTesterMixin


@skip_mps
class IFImg2ImgSuperResolutionPipelineFastTests(PipelineTesterMixin, IFPipelineTesterMixin, unittest.TestCase):
    pipeline_class = IFImg2ImgSuperResolutionPipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS - {"width", "height"}
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS.union({"original_image"})
    required_optional_params = PipelineTesterMixin.required_optional_params - {"latents"}

    def get_dummy_components(self):
        return self._get_superresolution_dummy_components()

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        original_image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
        image = floats_tensor((1, 3, 16, 16), rng=random.Random(seed)).to(device)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "original_image": original_image,
            "generator": generator,
            "num_inference_steps": 2,
            "output_type": "np",
        }

        return inputs

    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )
    def test_xformers_attention_forwardGenerator_pass(self):
        self._test_xformers_attention_forwardGenerator_pass(expected_max_diff=1e-3)

    def test_save_load_optional_components(self):
        self._test_save_load_optional_components()

    @unittest.skipIf(torch_device not in ["cuda", "xpu"], reason="float16 requires CUDA or XPU")
    @require_accelerator
    def test_save_load_float16(self):
        # Due to non-determinism in save load of the hf-internal-testing/tiny-random-t5 text encoder
        super().test_save_load_float16(expected_max_diff=1e-1)

    def test_attention_slicing_forward_pass(self):
        self._test_attention_slicing_forward_pass(expected_max_diff=1e-2)

    def test_save_load_local(self):
        self._test_save_load_local()

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(
            expected_max_diff=1e-2,
        )


@slow
@require_torch_gpu
class IFImg2ImgSuperResolutionPipelineSlowTests(unittest.TestCase):
    def setUp(self):
        # clean up the VRAM before each test
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_if_img2img_superresolution(self):
        pipe = IFImg2ImgSuperResolutionPipeline.from_pretrained(
            "DeepFloyd/IF-II-L-v1.0",
            variant="fp16",
            torch_dtype=torch.float16,
        )
        pipe.unet.set_attn_processor(AttnAddedKVProcessor())
        pipe.enable_model_cpu_offload()

        torch.cuda.reset_max_memory_allocated()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        generator = torch.Generator(device="cpu").manual_seed(0)

        original_image = floats_tensor((1, 3, 256, 256), rng=random.Random(0)).to(torch_device)
        image = floats_tensor((1, 3, 64, 64), rng=random.Random(0)).to(torch_device)

        output = pipe(
            prompt="anime turtle",
            image=image,
            original_image=original_image,
            generator=generator,
            num_inference_steps=2,
            output_type="np",
        )

        image = output.images[0]

        assert image.shape == (256, 256, 3)

        mem_bytes = torch.cuda.max_memory_allocated()
        assert mem_bytes < 12 * 10**9

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/if/test_if_img2img_superresolution_stage_II.npy"
        )
        assert_mean_pixel_difference(image, expected_image)

        pipe.remove_all_hooks()
