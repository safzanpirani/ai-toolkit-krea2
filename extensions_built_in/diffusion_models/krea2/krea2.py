"""Krea2Model -- ai-toolkit adapter for Krea 2 (K2), enabling LoRA training.

K2 is a single-stream MMDiT (~13B) with a Qwen3-VL-4B multi-layer text tap and
the Qwen-Image VAE. diffusers has no K2 transformer class, so the network is
vendored under ./src (mmdit.py), along with the text conditioner (encoder.py)
and a preview sampler (pipeline.py).

Notable, K2-specific pieces of this adapter:

  * Conditioning is 4D. The text tap returns 12 stacked Qwen3-VL hidden-state
    layers -> per-prompt ``(L, 12, 2560)``. ``AdvancedPromptEmbeds`` requires 2D
    per-prompt tensors, so get_prompt_embeds flattens the layer axis to
    ``(L, 12*2560)`` and get_noise_prediction restores ``(B, Lt, 12, 2560)``
    before the model call (the pattern the example README documents).

  * The transformer eats patchified tokens + 3-axis positions + a combined
    text/image key-padding mask, not ``(B, C, h, w)``. get_noise_prediction
    patchifies via ``prepare()`` and unpatchifies the prediction back to
    ``(B, C, h, w)`` so the rest of the trainer is none the wiser.

  * Velocity convention already matches ai-toolkit: K2 trains ``v = noise -
    clean`` with t=1 == pure noise, so timesteps map by ``t/1000`` with no flip
    and get_loss_target is the standard ``noise - clean``.

  * Loading is Windows-mmap-safe and memory-bounded: the 26.6 GB bf16 checkpoint
    is streamed with plain buffered I/O (safetensors' mmap access-violates on a
    26 GB file on Windows), and the 28 main blocks are quanto-quantized one at a
    time on the GPU then parked on CPU, so peak host RAM stays well under the
    bf16 size. The sensitive non-block layers (first/last/tmlp/tproj/txtfusion/
    txtmlp) are kept in bf16 -- negligible FLOPs, and the txtfusion projector is
    a Linear over a 4D tensor that quanto's matmul would reject.
"""

import json
import os
import struct
from typing import List, Optional

import torch
import yaml
from einops import rearrange
from safetensors.torch import save_file

from diffusers import AutoencoderKLQwenImage
from optimum.quanto import freeze

from toolkit.accelerator import unwrap_model
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.models.base_model import BaseModel
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.util.quantize import get_qtype, quantize

from .src.mmdit import SingleMMDiTConfig, SingleStreamDiT
from .src.encoder import Qwen3VLConditioner
from .src.pipeline import Krea2Pipeline, pad_prompt_embeds
from .src.sampling_utils import prepare


# K2 "single_mmdit_large_wide" (from the reference inference.py).
K2_CONFIG = SingleMMDiTConfig(
    features=6144,
    tdim=256,
    txtdim=2560,
    heads=48,
    kvheads=12,
    multiplier=4,
    layers=28,
    patch=2,
    channels=16,
    txtheads=20,
    txtkvheads=20,
    txtlayers=12,
)

DEFAULT_TE = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_VAE = "Qwen/Qwen-Image"

# Training/sampling flow-match scheduler. shift warps timestep sampling toward
# the high-noise end; ~3.0 is a sane high-res default and close to K2's
# effective resolution-aware mu at 1024px.
scheduler_config = {
    "num_train_timesteps": 1000,
    "use_dynamic_shifting": False,
    "shift": 3.0,
}

_ST_DTYPE = {
    "F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32,
    "F64": torch.float64, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
}


def _st_header(path):
    """Parse a safetensors header without mmap. Returns (header_dict, data_base)."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


def _read_tensor(fh, base, meta, dev, cast=torch.bfloat16):
    """Plain buffered read of one tensor (no mmap) -> dev. Floats cast to `cast`."""
    dtype = _ST_DTYPE[meta["dtype"]]
    start, end = meta["data_offsets"]
    fh.seek(base + start)
    buf = bytearray(end - start)
    fh.readinto(buf)
    t = torch.frombuffer(buf, dtype=dtype).reshape(meta["shape"])
    if cast is not None and t.is_floating_point():
        t = t.to(cast)
    return t.to(dev)


def _load_subtree(fh, base, header, module, prefix, dtype, device, seen):
    """Buffered-load every param/buffer of `module` (and descendants) onto `device`."""
    for sub_name, sub in module.named_modules():
        p2 = (prefix + sub_name + ".") if sub_name else prefix
        for pname, p in list(sub._parameters.items()):
            if p is None:
                continue
            key = p2 + pname
            if key not in header:
                raise KeyError(f"checkpoint missing param {key}")
            sub._parameters[pname] = torch.nn.Parameter(
                _read_tensor(fh, base, header[key], device, cast=dtype),
                requires_grad=False,
            )
            seen.add(key)
        for bname, b in list(sub._buffers.items()):
            if b is None:
                continue
            key = p2 + bname
            if key in header:
                sub._buffers[bname] = _read_tensor(
                    fh, base, header[key], device, cast=dtype
                )
                seen.add(key)


class Krea2Model(BaseModel):
    arch = "krea2"
    use_old_lokr_format = False

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs
        )
        self.is_flow_matching = True
        self.is_transformer = True
        # Recursively LoRA every Linear under the DiT (attention + MLP across the
        # 28 single-stream blocks and the text-fusion blocks) -- the standard
        # "all linear" target other ai-toolkit DiT adapters use.
        self.target_lora_modules = ["SingleStreamDiT"]

        self.patch_size = K2_CONFIG.patch
        self.vae_scale_factor = 8
        self.num_latent_channels = K2_CONFIG.channels
        self.num_txt_layers = K2_CONFIG.txtlayers
        self.txt_dim = K2_CONFIG.txtdim
        self.max_text_length = 512

    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return self.vae_scale_factor * self.patch_size

    # ------------------------------------------------------------------ load
    def load_model(self):
        dtype = self.torch_dtype
        device = self.device_torch
        self.print_and_status_update("Loading Krea 2 (K2)")

        mk = getattr(self.model_config, "model_kwargs", None) or {}
        te_id = mk.get("text_encoder", DEFAULT_TE)
        vae_id = mk.get("vae", DEFAULT_VAE)

        # Resolve the transformer checkpoint (a .safetensors file, or a dir
        # containing raw/turbo/model.safetensors).
        path = self.model_config.name_or_path
        if os.path.isdir(path):
            for cand in ("raw.safetensors", "turbo.safetensors", "model.safetensors"):
                if os.path.exists(os.path.join(path, cand)):
                    path = os.path.join(path, cand)
                    break

        self.print_and_status_update("Loading transformer (streaming, mmap-safe)")
        with torch.device("meta"):
            transformer = SingleStreamDiT(K2_CONFIG)

        do_quant = bool(self.model_config.quantize)
        qtype = get_qtype(self.model_config.qtype or "qfloat8") if do_quant else None

        header, base = _st_header(path)
        keys, seen = set(header), set()
        n_blocks = len(transformer.blocks)
        with open(path, "rb") as fh:
            for child_name, child in transformer.named_children():
                if child_name == "blocks":
                    for i, block in enumerate(child):
                        # Stream a block's weights straight onto the GPU and
                        # quantize it there, then park it on CPU as fp8.
                        blk_dev = device if do_quant else "cpu"
                        _load_subtree(
                            fh, base, header, block, f"blocks.{i}.",
                            dtype, blk_dev, seen,
                        )
                        if do_quant:
                            quantize(block, weights=qtype)
                            freeze(block)
                            block.to("cpu")
                        if (i + 1) % 7 == 0 or i + 1 == n_blocks:
                            self.print_and_status_update(
                                f"  blocks {i + 1}/{n_blocks}"
                            )
                        flush()
                else:
                    # Sensitive non-block layers stay bf16 on CPU for now.
                    _load_subtree(
                        fh, base, header, child, f"{child_name}.",
                        dtype, "cpu", seen,
                    )

        missing = keys - seen
        if missing:
            raise KeyError(
                f"unused checkpoint keys ({len(missing)}): {sorted(missing)[:8]}"
            )
        del header
        flush()

        # ---- text encoder ----
        # Load + quantize the TE while the transformer is still on CPU (avoids a
        # GPU spike from the ~7 GB bf16 TE colliding with the 14 GB transformer),
        # then PARK THE TE ON CPU. It is only needed briefly to encode prompts
        # (get_prompt_embeds brings it up and back down), so keeping it off the
        # GPU lets the transformer run fully on-GPU / un-offloaded within 24 GB.
        self.print_and_status_update("Loading text encoder (Qwen3-VL-4B)")
        text_encoder = Qwen3VLConditioner(
            te_id, max_length=self.max_text_length, torch_dtype=dtype
        )
        text_encoder.drop_visual()
        text_encoder = text_encoder.eval().requires_grad_(False)
        text_encoder.to(self.device_torch, dtype=dtype)
        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing text encoder")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
            freeze(text_encoder)
        text_encoder.to("cpu")
        flush()

        # now hand the GPU to the transformer
        if not self.model_config.low_vram:
            transformer.to(device, dtype=dtype)
        flush()

        # ---- VAE ----
        self.print_and_status_update("Loading VAE (Qwen-Image)")
        vae = AutoencoderKLQwenImage.from_pretrained(
            vae_id, subfolder="vae", torch_dtype=self.vae_torch_dtype
        )
        vae.to(self.vae_device_torch, dtype=self.vae_torch_dtype)
        vae.eval()
        vae.requires_grad_(False)
        flush()

        self.noise_scheduler = Krea2Model.get_train_scheduler()
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = text_encoder.tokenizer
        self.model = transformer
        self.pipeline = Krea2Pipeline(self)
        self.print_and_status_update("Model Loaded")

    # ------------------------------------------------------------- sampling
    def get_generation_pipeline(self):
        return Krea2Pipeline(self)

    def generate_single_image(
        self,
        pipeline: Krea2Pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: AdvancedPromptEmbeds,
        unconditional_embeds: AdvancedPromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        sc = self.get_bucket_divisibility()
        gen_config.width = int(gen_config.width // sc * sc)
        gen_config.height = int(gen_config.height // sc * sc)

        img = pipeline(
            conditional_embeds=conditional_embeds,
            unconditional_embeds=unconditional_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
        )[0]
        return img

    # ------------------------------------------------------------- training
    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0..1000, 1000 = pure noise
        text_embeddings: AdvancedPromptEmbeds,
        **kwargs,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)
        device = self.device_torch
        dtype = self.torch_dtype
        patch = self.patch_size

        b, c, h, w = latent_model_input.shape
        t01 = timestep.to(device, dtype=torch.float32) / 1000.0

        # rebuild K2's 4D text conditioning + key-padding mask
        feats, text_mask = pad_prompt_embeds(
            text_embeddings.text_embeds, device, dtype
        )
        lt = feats.shape[1]
        context = feats.reshape(b, lt, self.num_txt_layers, self.txt_dim)

        # patchify latent + combined text/image positions & mask
        img_tokens, pos, full_mask = prepare(
            latent_model_input.to(device, dtype), lt, patch, text_mask
        )

        out = self.model(
            img=img_tokens, context=context, t=t01, pos=pos, mask=full_mask
        )

        noise_pred = rearrange(
            out,
            "b (gh gw) (c ph pw) -> b c (gh ph) (gw pw)",
            gh=h // patch, gw=w // patch, ph=patch, pw=patch, c=c,
        )
        return noise_pred

    def get_prompt_embeds(self, prompt) -> AdvancedPromptEmbeds:
        if isinstance(prompt, str):
            prompt = [prompt]

        # The TE lives on CPU between calls (see load_model). Bring it up to the
        # GPU just for the forward, then send it back so the transformer keeps
        # the VRAM. Embeds are parked on CPU; the trainer moves them as needed.
        te = self.text_encoder
        moved = te.device == torch.device("cpu")
        if moved:
            te.to(self.device_torch)

        hiddens, mask = te(prompt)  # (B, L, 12, 2560), (B, L) bool
        mask = mask.bool()

        embeds_list = []
        for i in range(len(prompt)):
            valid = mask[i]
            h = hiddens[i][valid]            # (L_real, 12, 2560)
            l_real = h.shape[0]
            # flatten the 12-layer axis into features so the per-item tensor is 2D
            h = h.reshape(l_real, self.num_txt_layers * self.txt_dim)
            embeds_list.append(h.to(self.torch_dtype).cpu())

        if moved:
            te.to("cpu")
            flush()

        return AdvancedPromptEmbeds(text_embeds=embeds_list)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def condition_noisy_latents(self, latents: torch.Tensor, batch):
        return latents

    # ------------------------------------------------------------- VAE I/O
    # Qwen-Image VAE is a Wan-style 3D VAE: add/remove a frame dim and normalize
    # with per-channel latents_mean/std. Identical handling to the qwen_image
    # adapter (K2 uses the exact same VAE).
    def encode_images(self, image_list: List[torch.Tensor], device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype
        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)
        self.vae.eval()
        self.vae.requires_grad_(False)

        image_list = [image.to(device, dtype=dtype) for image in image_list]
        images = torch.stack(image_list).to(device, dtype=dtype)
        images = images.unsqueeze(2)  # frame dim
        latents = self.vae.encode(images).latent_dist.sample()

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)

        latents = (latents - latents_mean) * latents_std
        latents = latents.to(device, dtype=dtype).squeeze(2)
        return latents

    def decode_latents(self, latents: torch.Tensor, device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype
        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)

        latents = latents.to(device, dtype=dtype).unsqueeze(2)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents * latents_std + latents_mean
        images = self.vae.decode(latents).sample
        return images.squeeze(2).to(device, dtype=dtype)

    # ------------------------------------------------------------- saving
    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        transformer = unwrap_model(self.model)
        os.makedirs(os.path.join(output_path, "transformer"), exist_ok=True)
        state_dict = {
            k: v.clone().to("cpu", dtype=save_dtype)
            for k, v in transformer.state_dict().items()
        }
        save_file(
            state_dict,
            os.path.join(output_path, "transformer", "model.safetensors"),
        )
        with open(os.path.join(output_path, "aitk_meta.yaml"), "w") as f:
            yaml.dump(meta, f)

    def get_base_model_version(self):
        return "krea2"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["blocks"]

    def convert_lora_weights_before_save(self, state_dict):
        return {
            k.replace("transformer.", "diffusion_model."): v
            for k, v in state_dict.items()
        }

    def convert_lora_weights_before_load(self, state_dict):
        return {
            k.replace("diffusion_model.", "transformer."): v
            for k, v in state_dict.items()
        }
