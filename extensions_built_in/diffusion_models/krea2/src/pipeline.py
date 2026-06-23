"""Minimal K2 preview sampler for ai-toolkit training previews.

ai-toolkit encodes prompts itself and only ever hands the pipeline
``AdvancedPromptEmbeds`` (never text). This pipeline reconstructs K2's 4D text
conditioning from the stored 2D embeds, runs the reference Euler+CFG flow-match
loop, and decodes with the model's VAE. It is faithful to the reference
``sampling.py`` (same CFG form ``cond + g*(cond - uncond)`` and ``mu`` schedule).
"""

from typing import List, Optional

import torch
from einops import rearrange
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor

from .sampling_utils import prepare, roundup, timesteps


def pad_prompt_embeds(embeds_list, device, dtype):
    """Right-pad per-item (L_i, D) text features into (B, L_max, D) + (B, L_max) mask."""
    lengths = [e.shape[0] for e in embeds_list]
    max_len = max(lengths)
    dim = embeds_list[0].shape[-1]
    batch_size = len(embeds_list)

    features = torch.zeros(batch_size, max_len, dim, device=device, dtype=dtype)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
    for i, e in enumerate(embeds_list):
        n = e.shape[0]
        features[i, :n] = e.to(device, dtype)
        mask[i, :n] = True
    return features, mask


class Krea2Pipeline:
    def __init__(self, model):
        self.model = model

    @property
    def device(self):
        return self.model.device_torch

    def to(self, *args, **kwargs):
        return self

    def set_progress_bar_config(self, **kwargs):
        pass

    def _context(self, embeds, device, dtype):
        feats, mask = pad_prompt_embeds(embeds.text_embeds, device, dtype)
        b, lt, _ = feats.shape
        ctx = feats.reshape(b, lt, self.model.num_txt_layers, self.model.txt_dim)
        return ctx, mask

    @torch.no_grad()
    def __call__(
        self,
        conditional_embeds,
        unconditional_embeds,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 4.5,
        latents: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> List[Image.Image]:
        m = self.model
        device = m.device_torch
        dtype = m.torch_dtype
        patch = m.patch_size
        comp = m.vae_scale_factor

        align = comp * patch
        width, height = roundup(width, align), roundup(height, align)
        gh, gw = height // comp, width // comp

        if latents is None:
            shape = (1, m.num_latent_channels, gh, gw)
            # randn_tensor handles a CPU generator targeting a CUDA tensor
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        latents = latents.to(device, dtype=dtype)

        ctx, cond_mask = self._context(conditional_embeds, device, dtype)
        img, pos, mask = prepare(latents, ctx.shape[1], patch, cond_mask)

        do_cfg = unconditional_embeds is not None and guidance_scale and guidance_scale > 0
        if do_cfg:
            unctx, un_mask = self._context(unconditional_embeds, device, dtype)
            _, unpos, unmask = prepare(latents, unctx.shape[1], patch, un_mask)

        # mu interpolation endpoints (same as reference sampling.py)
        x1 = (256 // align) ** 2
        x2 = (1280 // align) ** 2
        ts = timesteps(img.shape[1], num_inference_steps, x1, x2)

        for tcurr, tprev in zip(ts[:-1], ts[1:]):
            t = torch.full((img.shape[0],), tcurr, device=device, dtype=dtype)
            cond = m.model(img=img, context=ctx, t=t, pos=pos, mask=mask)
            if do_cfg:
                uncond = m.model(img=img, context=unctx, t=t, pos=unpos, mask=unmask)
                v = cond + guidance_scale * (cond - uncond)
            else:
                v = cond
            img = img + (tprev - tcurr) * v

        latent = rearrange(
            img,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            ph=patch, pw=patch,
            h=gh // patch, w=gw // patch,
        )
        images = m.decode_latents(latent.to(dtype), device=device, dtype=dtype)
        images = images.float().clamp(-1.0, 1.0)
        images = ((images + 1.0) * 127.5).round().to(torch.uint8)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(arr) for arr in images]
