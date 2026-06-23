"""K2 flow-matching helpers shared by the adapter and the preview pipeline.

Vendored from the K2 reference ``sampling.py`` (the parts that are pure
geometry/scheduling, no model or VAE). ``prepare`` patchifies the latent and
builds the combined text+image position / key-padding tensors; ``timesteps``
is the resolution-aware shifted 1->0 schedule.
"""

import math

import torch
from einops import rearrange, repeat


def roundup(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


def prepare(img, txtlen, patch, txtmask):
    """Patchify latent + build combined position/mask. Returns (img, pos, mask).

    in:  img      (B, C, h, w) latent
         txtlen   number of text tokens (Lt)
         patch    patch size
         txtmask  (B, Lt) bool text key-padding mask
    out: img      (B, h/patch * w/patch, C*patch*patch) image tokens
         pos      (B, Lt + Limg, 3) 3-axis position ids (text rows are 0)
         mask     (B, Lt + Limg) bool combined key-padding mask
    """
    b, _, h, w = img.shape
    h_, w_ = h // patch, w // patch
    imgids = torch.zeros((h_, w_, 3), device=img.device)
    imgids[..., 1] = torch.arange(h_, device=img.device)[:, None]
    imgids[..., 2] = torch.arange(w_, device=img.device)[None, :]
    imgpos = repeat(imgids, "h w three -> b (h w) three", b=b, three=3)
    imgmask = torch.ones(b, h_ * w_, device=img.device, dtype=torch.bool)
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)

    txtpos = torch.zeros(b, txtlen, 3, device=img.device)
    mask = torch.cat((txtmask, imgmask), dim=1)
    pos = torch.cat((txtpos, imgpos), dim=1)
    return img, pos, mask


def timesteps(seq_len, steps, x1, x2, y1=0.5, y2=1.15, sigma=1.0, mu=None):
    """Resolution-aware flow-matching timestep schedule (t: 1 -> 0)."""
    ts = torch.linspace(1, 0, steps + 1)
    if mu is None:
        slope = (y2 - y1) / (x2 - x1)
        mu = slope * seq_len + (y1 - slope * x1)
    ts = math.exp(mu) / (math.exp(mu) + (1.0 / ts - 1.0) ** sigma)
    return ts.tolist()
