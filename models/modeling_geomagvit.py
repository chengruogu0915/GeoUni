from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
import math
from einops import rearrange, reduce, pack, unpack
from collections import namedtuple
from math import log2, ceil
from .modeling_utils import ConfigMixin, ModelMixin, register_to_config
from typing import List


class Encoder(ModelMixin, ConfigMixin):
    @dataclass
    class Config:
        ch: int = 128
        out_ch: int = 3
        in_channels: int = 3
        num_res_blocks: int = 4
        z_channels: int = 13
        ch_mult: List[int] = field(default_factory=lambda: [1, 2, 2, 2, 4, 4])  
        resolution: int = 512
        double_z: bool = False
        
        
    def __init__(self, 
                 ch: int = 128,
                 out_ch: int = 3, 
                 in_channels: int = 3, 
                 num_res_blocks: int = 4, 
                 z_channels: int = 13, 
                 ch_mult:List[int]=[1, 2, 2, 2, 4, 4], 
                 resolution=512,
                 double_z=False,
                ):
        super().__init__()
        
        self.in_channels = in_channels
        self.z_channels = z_channels
        self.resolution = resolution

        self.num_res_blocks = num_res_blocks
        self.num_blocks = len(ch_mult)
        
        self.conv_in = nn.Conv2d(in_channels,
                                 ch,
                                 kernel_size=(3, 3),
                                 padding=1,
                                 bias=False
        )

        ## construct the model
        self.down = nn.ModuleList()

        in_ch_mult = (1,)+tuple(ch_mult)
        for i_level in range(self.num_blocks):
            block = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level] #[1, 1, 2, 2, 4]
            block_out = ch*ch_mult[i_level] #[1, 2, 2, 4]
            for _ in range(self.num_res_blocks):
                block.append(ResBlock(block_in, block_out))
                block_in = block_out
            
            down = nn.Module()
            down.block = block
            if i_level < self.num_blocks - 1:
                down.downsample = nn.Conv2d(block_out, block_out, kernel_size=(3, 3), stride=(2, 2), padding=1)

            self.down.append(down)
        
        ### mid
        self.mid_block = nn.ModuleList()
        for res_idx in range(self.num_res_blocks):
            self.mid_block.append(ResBlock(block_in, block_in))
        
        ### end
        self.norm_out = nn.GroupNorm(32, block_out, eps=1e-6)
        self.conv_out = nn.Conv2d(block_out, z_channels, kernel_size=(1, 1))
            
    def forward(self, x):

        ## down
        x = self.conv_in(x)
        for i_level in range(self.num_blocks):
            for i_block in range(self.num_res_blocks):
                x = self.down[i_level].block[i_block](x)
            
            if i_level <  self.num_blocks - 1:
                x = self.down[i_level].downsample(x)
        
        ## mid 
        for res in range(self.num_res_blocks):
            x = self.mid_block[res](x)
        

        x = self.norm_out(x)
        x = swish(x)
        x = self.conv_out(x)

        return x

class LFQ(nn.Module):
    def __init__(self,
                 dim: int = 13,
                 codebook_size: int = 2**13,
                 num_codebooks: int = 1,):
        super().__init__()
        
        self.codebook_size = codebook_size
        self.codebook_dim = dim
        self.dim = dim
        self.num_codebooks = num_codebooks
        
        codebook_dims = self.codebook_dim * num_codebooks
        
        has_projections = dim != codebook_dims
        self.has_projections = has_projections
        self.register_buffer('mask', 2 ** torch.arange(self.codebook_dim), persistent=False)
        self.register_buffer('zero', torch.tensor(0.), persistent = False)
        
        # codes

        all_codes = torch.arange(codebook_size)
        bits = self.indices_to_bits(all_codes)
        codebook = bits * 2.0 - 1.0

        self.register_buffer('codebook', codebook, persistent = False)
    
    def indices_to_bits(self, x):
        """
        x: long tensor of indices 

        returns big endian bits
        eg: 3(010) then output:[False, True, False]
        """
        mask = 2 ** torch.arange(self.codebook_dim, device=x.device, dtype=torch.long)
        # x is now big endian bits, the last dimension being the bits
        x = (x.unsqueeze(-1) & mask) != 0
        return x
    
        
    def get_codebook_entry(self, x, bhwc): #0610

        mask = 2 ** torch.arange(self.codebook_dim, device=x.device, dtype=torch.long)
        
        x = (x.unsqueeze(-1) & mask) != 0
        x = x * 2.0 - 1.0 #back to the float
        ## scale back to the 
        b, h, w, c = bhwc
        x = rearrange(x, "b (h w) c -> b h w c", h=h, w=w, c=c)
        x = rearrange(x, "b h w c -> b c h w")
        return x
    
    def bits_to_indices(self, bits):
        """
        bits: bool tensor of big endian bits, where the last dimension is the bit dimension

        returns indices, which are long integers from 0 to self.codebook_size
        """
        assert bits.shape[-1] == self.codebook_dim
        indices = 2 ** torch.arange(
            0,
            self.codebook_dim,
            1,
            dtype=torch.long,
            device=bits.device,
        )
        return (bits * indices).sum(-1)
    
    def decode(self, x):
        """
        x: ... NH
            where NH is number of codebook heads
            A longtensor of codebook indices, containing values from
            0 to self.codebook_size
        """
        x = self.indices_to_bits(x)
        # to some sort of float
        #x = x.to(self.dtype)
        x = x.to(torch.float32)
        # -1 or 1
        x = x * 2 - 1
        x = rearrange(x, "... NC Z-> ... (NC Z)")
        return x
    
    
    def forward(
        self, 
        x,
    ):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim 
        """

        x = rearrange(x, 'b d ... -> b ... d')
        x, ps = pack_one(x, 'b * d')
        # split out number of codebooks

        x = rearrange(x, 'b n (c d) -> b n c d', c = self.num_codebooks)


        codebook_value = torch.Tensor([1.0]).to(device=x.device, dtype=x.dtype)
        quantized = torch.where(x > 0, codebook_value, -codebook_value) # higher than 0 filled 

        # calculate indices
        indices = reduce((quantized > 0).int() * self.mask.int(), 'b n c d -> b n c', 'sum')

        


        # use straight-through gradients (optionally with custom activation fn) if training

        quantized = x + (quantized - x).detach() #transfer to quantized

        # merge back codebook dim

        quantized = rearrange(quantized, 'b n c d -> b n (c d)')

        # reconstitute image or video dimensions

        quantized = unpack_one(quantized, ps, 'b * d')
        quantized = rearrange(quantized, 'b ... d -> b d ...')

        
        
        indices = unpack_one(indices, ps, 'b * c')
        indices = indices.flatten()

        ret = (quantized, indices)
        return ret


class Decoder(ModelMixin, ConfigMixin):
    def __init__(self, 
                 ch: int = 128, 
                 out_ch: int = 3,
                 in_channels: int = 3,
                 num_res_blocks:int = 4,
                 z_channels:int = 13, 
                 ch_mult:List[int]=[1, 2, 2, 2, 4, 4], 
                 resolution:int = 512, 
                 double_z: bool = False):
        super().__init__()

        self.ch = ch
        self.num_blocks = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        block_in = ch*ch_mult[self.num_blocks-1]

        self.conv_in = nn.Conv2d(
            z_channels, block_in, kernel_size=(3, 3), padding=1, bias=True
        )

        self.mid_block = nn.ModuleList()
        for res_idx in range(self.num_res_blocks):
            self.mid_block.append(ResBlock(block_in, block_in))
        
        self.up = nn.ModuleList()

        self.adaptive = nn.ModuleList()

        for i_level in reversed(range(self.num_blocks)):
            block = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            self.adaptive.insert(0, AdaptiveGroupNorm(z_channels, block_in))
            for i_block in range(self.num_res_blocks):
                # if i_block == 0:
                #     block.append(ResBlock(block_in, block_out, use_agn=True))
                # else:
                block.append(ResBlock(block_in, block_out))
                block_in = block_out
            
            up = nn.Module()
            up.block = block
            if i_level > 0:
                up.upsample = Upsampler(block_in)
            self.up.insert(0, up)
        
        self.norm_out = nn.GroupNorm(32, block_in, eps=1e-6)

        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=(3, 3), padding=1)
    
    def forward(self, z):
        
        style = z.clone() #for adaptive groupnorm

        z = self.conv_in(z)

        ## mid
        for res in range(self.num_res_blocks):
            z = self.mid_block[res](z)
        
        ## upsample
        for i_level in reversed(range(self.num_blocks)):
            ### pass in each resblock first adaGN
            z = self.adaptive[i_level](z, style)
            for i_block in range(self.num_res_blocks):
                z = self.up[i_level].block[i_block](z)
            
            if i_level > 0:
                z = self.up[i_level].upsample(z)
        
        z = self.norm_out(z)
        z = swish(z)
        z = self.conv_out(z)

        return z

class GeoMAGVIT(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
            self,
        ):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quantize = LFQ()
        self.z_channels = 13
                    
    def encode(self, x):
        h = self.encoder(x)
        quant, indices = self.quantize(h)
        return quant, indices

    def decode(self, quant):
        #quant:batch * dim 11  * 32 * 32 （+-1）
        #dec:batch * 3 * 512 * 512
        dec = self.decoder(quant)
        return dec
    
    def get_code(self, pixel_values):
        hidden_states = self.encoder(pixel_values)
        quant, indices = self.quantize(hidden_states)
        indices = indices.reshape(pixel_values.shape[0], -1)
        
        return indices

    def decode_code(self, codebook_indices, shape=None):
        
        
        z_q = self.quantize.get_codebook_entry(codebook_indices, 
                                               (codebook_indices.shape[0],int(math.sqrt(codebook_indices.shape[-1])),int(math.sqrt(codebook_indices.shape[-1])), self.z_channels)) 
        
        
        dec = self.decoder(z_q)
        
        return dec

    def forward(self, input):
        quant, indices = self.encode(input)
        dec = self.decode(quant)
        return dec


# swish:x*sigmoid(x)
def swish(x):
    # swish
    return x*torch.sigmoid(x)   

class ResBlock(nn.Module):
    def __init__(self, 
                 in_filters,
                 out_filters,
                 use_conv_shortcut = False,
                 use_agn = False,
                 ) -> None:
        super().__init__()

        self.in_filters = in_filters # in_ch
        self.out_filters = out_filters # out_ch
        self.use_conv_shortcut = use_conv_shortcut
        self.use_agn = use_agn

        if not use_agn: ## agn is GroupNorm likewise skip it if has agn before
            self.norm1 = nn.GroupNorm(32, in_filters, eps=1e-6)
        self.norm2 = nn.GroupNorm(32, out_filters, eps=1e-6)

        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False)

        if in_filters != out_filters:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False)
            else:
                self.nin_shortcut = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), padding=0, bias=False)
    

    def forward(self, x, **kwargs):
        residual = x

        if not self.use_agn:
            x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_filters != self.out_filters:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)

        return x + residual


class AdaptiveGroupNorm(nn.Module):
    def __init__(self, z_channel, in_filters, num_groups=32, eps=1e-6):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=in_filters, eps=eps, affine=False)
        # self.lin = nn.Linear(z_channels, in_filters * 2)
        self.gamma = nn.Linear(z_channel, in_filters)
        self.beta = nn.Linear(z_channel, in_filters)
        self.eps = eps
    
    def forward(self, x, quantizer):
        B, C, _, _ = x.shape
        # quantizer = F.adaptive_avg_pool2d(quantizer, (1, 1))
        ### calcuate var for scale
        scale = rearrange(quantizer, "b c h w -> b c (h w)")
        scale = scale.var(dim=-1) + self.eps #not unbias
        scale = scale.sqrt()
        scale = self.gamma(scale).view(B, C, 1, 1)

        ### calculate mean for bias
        bias = rearrange(quantizer, "b c h w -> b c (h w)")
        bias = bias.mean(dim=-1)
        bias = self.beta(bias).view(B, C, 1, 1)
       
        x = self.gn(x)
        x = scale * x + bias

        return x
  

class Upsampler(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None
    ):
        super().__init__()
        dim_out = dim * 4
        self.conv1 = nn.Conv2d(dim, dim_out, (3, 3), padding=1)
        self.depth2space = depth_to_space

    def forward(self, x):
        """
        input_image: [B C H W]
        """
        out = self.conv1(x)
        out = self.depth2space(out, block_size=2)
        return out


def depth_to_space(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """ Depth-to-Space DCR mode (depth-column-row) core implementation.

        Args:
            x (torch.Tensor): input tensor. The channels-first (*CHW) layout is supported.
            block_size (int): block side size
    """
    # check inputs
    if x.dim() < 3:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor of at least 3 dimensions"
        )
    c, h, w = x.shape[-3:]

    s = block_size**2
    if c % s != 0:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor with C divisible by {s}, but got C={c} channels"
        )

    outer_dims = x.shape[:-3]

    # splitting two additional dimensions from the channel dimension
    x = x.view(-1, block_size, block_size, c // s, h, w)

    # putting the two new dimensions along H and W
    x = x.permute(0, 3, 4, 1, 5, 2)

    # merging the two new dimensions with H and W
    x = x.contiguous().view(*outer_dims, c // s, h * block_size,
                            w * block_size)

    return x

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def pack_one(t, pattern):
    return pack([t], pattern)

if __name__ == '__main__':
    encoder = Encoder()
    import ipdb
    ipdb.set_trace()
    print()

