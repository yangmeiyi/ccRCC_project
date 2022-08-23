from torch import nn
import PIL
import torchvision
from sklearn import metrics
import torch
import numpy as np
import torch.nn.functional as F
from functools import partial
from einops.layers.torch import Rearrange, Reduce
from TransRestNet_model.utils import Bar, Logger, AverageMeter, accuracy
from dataloader import CCRFolder
import pandas as pd
import os
import random
import time



class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + F.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )

def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor=4, dropout=0.):
    assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_size // patch_size) ** 2
    # chan_first = nn.Conv1d, chan_last = nn.Linear
    chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
    print(chan_first, chan_last)

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
        nn.Linear((patch_size ** 2) * channels, dim), #
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )



def Base_MLPMixer():
    model = MLPMixer(
        image_size=64,
        channels=3,
        patch_size=16,
        dim=512,
        depth=12,
        num_classes=2
    )
    return model




