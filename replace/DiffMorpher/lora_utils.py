from timeit import default_timer as timer
from datetime import timedelta
from PIL import Image
import os
import numpy as np
from einops import rearrange
import torch
import torch.nn.functional as F
from torchvision import transforms
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from packaging import version
from PIL import Image
import tqdm
import lpips
import sys
from transformers import AutoTokenizer, PretrainedConfig
from scipy.stats import norm
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers, LoraLoaderMixin
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    SlicedAttnAddedKVProcessor,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
import random
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.17.0")


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")

def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs

def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=False):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds




# model_path: path of the model
# images: input image or list of images, have not been pre-processed
# save_lora_dir: the path to save the lora
# prompt: the user input prompt
# lora_steps: number of lora training step
# lora_lr: learning rate of lora training
# lora_rank: the rank of lora
# train_batch_size: batch size for training to avoid OOM, None means process all images at once
def train_lora(images, prompt, prompt1, search_0, search_1, train_batch_size=10 ,save_lora_dir="./lora", model_path=None, tokenizer=None, text_encoder=None, vae=None, unet=None, noise_scheduler=None, lora_steps=200, lora_lr=2e-4, lora_rank=16, weight_name=None, safe_serialization=False, progress=tqdm):
    # initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        # mixed_precision='fp16'
    )
    set_seed(0)

    # Load the tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )
    # initialize the model
    if noise_scheduler is None:
        noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    if text_encoder is None:
        text_encoder_cls = import_model_class_from_model_name_or_path(model_path, revision=None)
        text_encoder = text_encoder_cls.from_pretrained(
            model_path, subfolder="text_encoder", revision=None
        )
    if vae is None:
        vae = AutoencoderKL.from_pretrained(
            model_path, subfolder="vae", revision=None
        )
    if unet is None:
        unet = UNet2DConditionModel.from_pretrained(
            model_path, subfolder="unet", revision=None
        )

    # set device and dtype
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    unet.to(device)
    vae.to(device)
    text_encoder.to(device)

    # initialize UNet LoRA
    unet_lora_attn_procs = {}
    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            raise NotImplementedError("name must start with up_blocks, mid_blocks, or down_blocks")

        if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
            lora_attn_processor_class = LoRAAttnAddedKVProcessor
        else:
            lora_attn_processor_class = (
                LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
            )
        unet_lora_attn_procs[name] = lora_attn_processor_class(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank
        )
    unet.set_attn_processor(unet_lora_attn_procs)
    unet_lora_layers = AttnProcsLayers(unet.attn_processors)

    # Optimizer creation
    params_to_optimize = (unet_lora_layers.parameters())
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=lora_lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=lora_steps,
        num_cycles=1,
        power=1.0,
    )

    # prepare accelerator
    unet_lora_layers = accelerator.prepare_model(unet_lora_layers)
    optimizer = accelerator.prepare_optimizer(optimizer)
    lr_scheduler = accelerator.prepare_scheduler(lr_scheduler)

    # initialize text embeddings
    with torch.no_grad():
        text_inputs = tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None)
        text_embedding = encode_prompt(
            text_encoder,
            text_inputs.input_ids,
            text_inputs.attention_mask,
            text_encoder_use_attention_mask=False
        )
        if prompt1 is not None:
            text_inputs1 = tokenize_prompt(tokenizer, prompt1, tokenizer_max_length=None)
            text_embedding1 = encode_prompt(
                text_encoder,
                text_inputs1.input_ids,
                text_inputs1.attention_mask,
                text_encoder_use_attention_mask=False
            )

    if not isinstance(images, list):
        images = [images]
    
    processed_images = []
    for img in images:
        if type(img) == np.ndarray:
            img = Image.fromarray(img)
        processed_images.append(img)
        
    # initialize latent distribution
    image_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.RandomCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    transformed_images = []
    for img in processed_images:
        transformed_img = image_transforms(img).to(device)
        transformed_images.append(transformed_img)
    
    # 编码所有图片得到 latent distributions
    # 获取图片数量
    if isinstance(search_0, list) and isinstance(search_1, list):
        print(f"search_0: {search_0}, search_1: {search_1}")
        print(f"transformed_images length: {len(transformed_images)}")
        img1_list_zoom = transformed_images[:len(transformed_images)//4] # 80
        img1_list_rotate = transformed_images[len(transformed_images)//4:len(transformed_images)//2]
        img2_list_zoom = transformed_images[len(transformed_images)//2:len(transformed_images)//4*3]
        img2_list_rotate = transformed_images[len(transformed_images)//4*3:]
        sampled = []
        search0_zoom = search_0[0]
        search0_rotate = search_0[1]
        search1_zoom = search_1[0]
        search1_rotate = search_1[1]
        for i in range(search0_zoom+1):
            sampled.append((img1_list_zoom[i], text_embedding))
        for i in range(search0_rotate+1):
            sampled.append((img1_list_rotate[i], text_embedding))
        for i in range(search1_zoom+1):
            sampled.append((img2_list_zoom[i], text_embedding1))
        for i in range(search1_rotate+1):
            sampled.append((img2_list_rotate[i], text_embedding1))
        
        sampled_0_zoom = sampled[:search0_zoom+1]
        sampled_0_rotate = sampled[search0_zoom+1:search0_zoom+search0_rotate+1]
        sampled_1_zoom = sampled[search0_zoom+search0_rotate+1:search0_zoom+search0_rotate+search1_zoom+1]
        sampled_1_rotate = sampled[search0_zoom+search0_rotate+search1_zoom+1:]
        sigma_ratio = 0.15
        
        scale_0_zoom = sigma_ratio * len(sampled_0_zoom)
        scale_0_rotate = sigma_ratio * len(sampled_0_rotate)
        scale_1_zoom = sigma_ratio * len(sampled_1_zoom)
        scale_1_rotate = sigma_ratio * len(sampled_1_rotate)
        
        gaussian_distribution_0_zoom = norm.pdf(
            np.arange(len(sampled_0_zoom)),
            loc=len(sampled_0_zoom) // 2,
            scale=scale_0_zoom
        )
        gaussian_distribution_0_zoom = gaussian_distribution_0_zoom / gaussian_distribution_0_zoom.sum()
        
        gaussian_distribution_0_rotate = norm.pdf(
            np.arange(len(sampled_0_rotate)),
            loc=len(sampled_0_rotate) // 2,
            scale=scale_0_rotate
        )
        gaussian_distribution_0_rotate = gaussian_distribution_0_rotate / gaussian_distribution_0_rotate.sum()
        
        gaussian_distribution_1_zoom = norm.pdf(
            np.arange(len(sampled_1_zoom)),
            loc=len(sampled_1_zoom) // 2,
            scale=scale_1_zoom
        )
        gaussian_distribution_1_zoom = gaussian_distribution_1_zoom / gaussian_distribution_1_zoom.sum()
        
        gaussian_distribution_1_rotate = norm.pdf(
            np.arange(len(sampled_1_rotate)),
            loc=len(sampled_1_rotate) // 2,
            scale=scale_1_rotate
        )
        gaussian_distribution_1_rotate = gaussian_distribution_1_rotate / gaussian_distribution_1_rotate.sum()
            
    else:
        img1_list = transformed_images[:len(transformed_images)//2] # 80
        img2_list = transformed_images[len(transformed_images)//2:]

        sampled = []
        for i in range(search_0+1):
            sampled.append((img1_list[i], text_embedding))
        for i in range(search_1+1):
            sampled.append((img2_list[i], text_embedding1))
            
        sampled_0 = sampled[:search_0+1]
        sampled_1 = sampled[search_0+1:]
            
        sigma_ratio = 0.15   
        scale_0 = sigma_ratio * len(sampled_0)
        scale_1 = sigma_ratio * len(sampled_1)
        
        gaussian_distribution_0 = norm.pdf(
            np.arange(len(sampled_0)),
            loc=len(sampled_0) // 2,
            scale=scale_0
        )
        gaussian_distribution_0 = gaussian_distribution_0 / gaussian_distribution_0.sum()
        gaussian_distribution_1 = norm.pdf(
            np.arange(len(sampled_1)),
            loc=len(sampled_1) // 2,
            scale=scale_1
        )
        gaussian_distribution_1 = gaussian_distribution_1 / gaussian_distribution_1.sum()
    for step in progress.tqdm(range(lora_steps), desc="Training LoRA..."):
        unet.train()
        
        # random select train_batch_size images from sampled
        if len(sampled) >= train_batch_size:
            if isinstance(search_0, list) and isinstance(search_1, list):
                batch_samples = []
                quarter = train_batch_size // 4
                idxs_0_zoom = np.random.choice(
                    len(sampled_0_zoom),
                    size=quarter,
                    p=gaussian_distribution_0_zoom,
                    replace=True
                )
                idxs_0_rotate = np.random.choice(
                    len(sampled_0_rotate),
                    size=quarter,
                    p=gaussian_distribution_0_rotate,
                    replace=True
                )
                idxs_1_zoom = np.random.choice(
                    len(sampled_1_zoom),
                    size=quarter,
                    p=gaussian_distribution_1_zoom,
                    replace=True
                )
                idxs_1_rotate = np.random.choice(
                    len(sampled_1_rotate),
                    size=quarter,
                    p=gaussian_distribution_1_rotate,
                    replace=True
                )
                for idx in idxs_0_zoom:
                    batch_samples.append(sampled_0_zoom[idx])
                for idx in idxs_0_rotate:
                    batch_samples.append(sampled_0_rotate[idx])
                for idx in idxs_1_zoom:
                    batch_samples.append(sampled_1_zoom[idx])
                for idx in idxs_1_rotate:
                    batch_samples.append(sampled_1_rotate[idx])
            else:
                # TODO: 采用高斯分布来采样train_batch_size个图片，中心为sampled 的 一半
                batch_samples = []
                half = train_batch_size // 2
                rest = train_batch_size - half
                idxs_0 = np.random.choice(
                    len(sampled_0),
                    size=half,
                    p=gaussian_distribution_0,
                    replace=True
                )
                idxs_1 = np.random.choice(
                    len(sampled_1),
                    size=rest,
                    p=gaussian_distribution_1,
                    replace=True
                )
                for idx in idxs_0:
                    batch_samples.append(sampled_0[idx])
                for idx in idxs_1:
                    batch_samples.append(sampled_1[idx])
        else:
            batch_samples = sampled
            
        batch_imgs = torch.stack([img for (img, _) in batch_samples], dim=0).to(device)
        batch_text = torch.cat([emb for (_, emb) in batch_samples], dim=0).to(device)
        latents_dist = vae.encode(batch_imgs).latent_dist
        model_input = latents_dist.sample() * vae.config.scaling_factor
        
        noise = torch.randn_like(model_input)
        bsz, channels, height, width = model_input.shape
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
        )
        timesteps = timesteps.long()

        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

        # Predict the noise residual
        model_pred = unet(noisy_model_input, timesteps, batch_text).sample

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(model_input, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    # save the trained lora
    # unet = unet.to(torch.float32)
    # vae = vae.to(torch.float32)
    # text_encoder = text_encoder.to(torch.float32)

    # unwrap_model is used to remove all special modules added when doing distributed training
    # so here, there is no need to call unwrap_model
    # unet_lora_layers = accelerator.unwrap_model(unet_lora_layers)
    LoraLoaderMixin.save_lora_weights(
        save_directory=save_lora_dir,
        unet_lora_layers=unet_lora_layers,
        text_encoder_lora_layers=None,
        weight_name=weight_name,
        safe_serialization=safe_serialization
    )
    
# def load_lora(unet, lora_0, lora_1, alpha):
#     lora = {}
#     for key in lora_0:
#         lora[key] = (1 - alpha) * lora_0[key] + alpha * lora_1[key]
#     unet.load_attn_procs(lora)
#     return unet

def load_lora(unet, lora):
    unet.load_attn_procs(lora)
    return unet