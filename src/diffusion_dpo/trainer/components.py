"""Module for initializing and managing system components like UNet, VAE, and text encoders."""

import logging
from packaging import version
from typing import List, Optional, Tuple, Union

import torch
from accelerate.utils import ContextManagers
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

from diffusion_dpo.text_encoder.model import import_text_encoder_class_from_name_or_path

logger = logging.getLogger(__name__)


def get_system_components(
    args,
    tokenizer_one: Optional[CLIPTokenizer] = None,
    tokenizer_two: Optional[CLIPTokenizer] = None,
    tokenizer_and_encoder_name: Optional[str] = None,
) -> dict:
    """Initialize and configure system components for training.
    
    Args:
        args: Training arguments
        tokenizer_one: First tokenizer for SDXL
        tokenizer_two: Second tokenizer for SDXL
        tokenizer_and_encoder_name: Name or path for tokenizer and encoder
        
    Returns:
        dict: Dictionary containing initialized components
    """
    components = {}
    
    # Initialize text encoders
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        if args.sdxl:
            # SDXL has two text encoders
            text_encoder_cls_one = import_text_encoder_class_from_name_or_path(
                tokenizer_and_encoder_name, args.revision
            )
            text_encoder_cls_two = import_text_encoder_class_from_name_or_path(
                tokenizer_and_encoder_name, args.revision, subfolder="text_encoder_2"
            )
            text_encoder_one = text_encoder_cls_one.from_pretrained(
                tokenizer_and_encoder_name, subfolder="text_encoder", revision=args.revision
            )
            text_encoder_two = text_encoder_cls_two.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision
            )
            
            if args.pretrained_model_name_or_path == "stabilityai/stable-diffusion-xl-refiner-1.0":
                components["text_encoders"] = [text_encoder_two]
                components["tokenizers"] = [tokenizer_two]
            else:
                components["text_encoders"] = [text_encoder_one, text_encoder_two]
                components["tokenizers"] = [tokenizer_one, tokenizer_two]
        else:
            text_encoder = CLIPTextModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
            )
            components["text_encoder"] = text_encoder

        # Initialize VAE
        vae_path = (
            args.pretrained_model_name_or_path
            if args.pretrained_vae_model_name_or_path is None
            else args.pretrained_vae_model_name_or_path
        )
        vae = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
            revision=args.revision
        )
        components["vae"] = vae

        # Initialize reference UNet
        ref_unet = UNet2DConditionModel.from_pretrained(
            args.unet_init if args.unet_init else args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=args.revision
        )
        components["ref_unet"] = ref_unet

    # Initialize main UNet
    if args.unet_init:
        logger.info(f"Initializing unet from {args.unet_init}")
    unet = UNet2DConditionModel.from_pretrained(
        args.unet_init if args.unet_init else args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision
    )
    components["unet"] = unet

    # Freeze components
    vae.requires_grad_(False)
    if args.sdxl:
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
    else:
        text_encoder.requires_grad_(False)
    if args.train_method == 'dpo':
        ref_unet.requires_grad_(False)

    # Enable xformers
    if is_xformers_available():
        import xformers
        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            logger.warning(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, "
                "please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
        unet.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError("xformers is not available. Make sure it is installed correctly")

    return components
