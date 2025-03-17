import numpy as np
import math
import torch
import torch.nn as nn
from copy import deepcopy
from typing import List
from .embedders import SMEEmbedder, SymFormerEmbedder
from odeformerplus.envs.utils import load_config


config = load_config('odeformerplus/envs/config.yaml')

MAX_XDIM = config.expression_tree.max_xdim


class ParameterModule(nn.Module):
    "Simple wrapper of nn.Parameter"
    def __init__(self, data, requires_grad=True):
        super().__init__()
        self.data = nn.Parameter(data, requires_grad=requires_grad)
    def __repr__(self): return f"Size={list(self.data.shape)}"
    def forward(self):
        return self.data

#########################
## TRANSFORMER MODULES ##
#########################

class MultiHeadAttention(nn.Module):
    """
    Multi-head scaled dot product attention.

    Args
    ----
    q: Tensor, shape [B, ..., Lq, d_model]
    k,v: Tensor, shape [B, ..., Lk, d_model]
    mask: BoolTensor, shape [B, ..., Lq,Lk], where True indicates no contribution
    return_weights: bool, whether return attention weights
        
    Returns
    -------
    output: Tensor, shape [B, ..., Lq, d_model]
    weights: NDArray, shape [B, ..., Lq, Lk], if return_weights=True, else None
    """
    def __init__(self, num_heads, d_model):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert (d_model % num_heads == 0), "`d_model` must be a multiple of `num_heads`"
        self.head_dim = d_model // num_heads
        
        # Linear projection layers
        self.fc_q = torch.nn.Linear(d_model, d_model)
        self.fc_k = torch.nn.Linear(d_model, d_model)
        self.fc_v = torch.nn.Linear(d_model, d_model)
        self.fc_o = torch.nn.Linear(d_model, d_model)
    
    def scaled_dot_product_attention(self, q, k, v, mask=None, return_weights=False):
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim) # [B, ..., Lq, Lk]
        
        if mask is not None:
            # check mask shape
            assert (q.shape[0] == mask.shape[0]        # batch
                    and q.shape[-2] == mask.shape[-2]  # Lq
                    and k.shape[-2] == mask.shape[-1]) # Lk
            if q.ndim != mask.ndim:
                mask = mask.view(mask.shape[0], *[1]*(q.ndim-3), mask.shape[-2], mask.shape[-1])
                mask = mask.repeat(1, *q.shape[1:-2], 1, 1) # [B, ...,Lq, Lk]
            
            scores = scores.masked_fill(mask, float('-1e20')) # places are True have no contributions to scores
        
        weights = torch.softmax(scores, dim=-1)
        if return_weights:
            return torch.matmul(weights, v), weights.detach().cpu().numpy()
        return torch.matmul(weights, v), None
    
    def forward(self, q, k, v, mask=None, return_weights=False):
        _q = self.fc_q(q).reshape(*q.shape[:-1], self.num_heads, self.head_dim) # [B, ..., Lq, num_heads, head_dim]
        _k = self.fc_k(k).reshape(*k.shape[:-1], self.num_heads, self.head_dim) # [B, ..., Lk, num_heads, head_dim]
        _v = self.fc_v(v).reshape(*v.shape[:-1], self.num_heads, self.head_dim) # [B, ..., Lk, num_heads, head_dim]

        _q = _q.transpose(-2,-3) # [B, ..., num_heads, Lq, head_dim]
        _k = _k.transpose(-2,-3) # [B, ..., num_heads, Lk, head_dim]
        _v = _v.transpose(-2,-3) # [B, ..., num_heads, Lk, head_dim]
        
        attn,weights = self.scaled_dot_product_attention(_q, _k, _v, mask, return_weights)
        attn = attn.transpose(-2,-3) # [B, ..., Lq, num_heads, head_dim]
        attn = attn.reshape(*attn.shape[:-2], self.d_model)
        output = self.fc_o(attn)
        return output, weights

class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0., actn=nn.ReLU()):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            actn,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.ffn(x)

class AddAndNorm(nn.Module):
    def __init__(self, d_model, dropout=0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, y):
        return self.dropout(self.norm(x + y))
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0, actn=nn.ReLU()):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads, d_model)
        self.ffn = FFN(d_model, d_ff, dropout, actn)
        self.add_and_norm1 = AddAndNorm(d_model, dropout)
        self.add_and_norm2 = AddAndNorm(d_model, dropout)
    
    def forward(self, q, k, v, mask=None, return_weights=False):
        attn, weights = self.mha(q, k, v, mask, return_weights) # attn: [B, ..., Lq, d_model]
        x = self.add_and_norm1(q, attn)
        ff = self.ffn(x)
        output = self.add_and_norm2(x, ff)
        if return_weights:
            return output, weights
        return output, None

class SinePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, device=None):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp((-torch.arange(0, d_model, 2) / d_model) * math.log(10000.0)) # div = 1/10000**(2i/d_model)
        pe = torch.zeros(max_len, d_model) # shape [L, d_model]
        pe[:,0::2] = torch.sin(pos*div) # PE(pos, 2i) = sin(pos/10000**(2i/d_model))
        pe[:,1::2] = torch.cos(pos*div) # PE(pos, 2i+1) = cos(pos/10000**(2i/d_model))
        self.pe = ParameterModule(pe, False)
        self.to(device)

    def forward(self, x, dim: int=-2):
        # x: [..., d_model]
        shape = [1]*x.ndim
        shape[dim] = x.shape[dim]
        shape[-1] = x.shape[-1] # [..., dim, ..., d_model]
        pe = (self.pe()[:x.shape[dim],:]).reshape(shape)
        return x + pe

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, device=None):
        super().__init__()
        self.pe = ParameterModule(torch.FloatTensor(max_len, d_model), True)
        nn.init.xavier_uniform_(self.pe.data)
        self.to(device)
        
    def forward(self, x, dim: int=-2):
        # x: [..., d_model]
        shape = [1]*x.ndim
        shape[dim] = x.shape[dim]
        shape[-1] = x.shape[-1] # [..., dim, ..., d_model]
        pe = (self.pe()[:x.shape[dim],:]).reshape(shape)
        return x + pe

#############################
## SET TRANSFORMER MODULES ##
#############################

# Adapted from https://github.com/juho-lee/set_transformer/blob/master/modules.py
class ISAB(nn.Module):
    "Induced-set attention block."
    def __init__(
            self,
            num_inds, # size of the induced set
            d_model=256,
            num_heads=8,
            dim_feedforward=1024,
            dropout=0.,
            actn=nn.SiLU(),
        ):
        super().__init__()
        self.induced_set = ParameterModule(torch.FloatTensor(num_inds, d_model),True)
        nn.init.xavier_uniform_(self.induced_set.data)
        
        self.mab1 = TransformerBlock(d_model, num_heads, dim_feedforward, dropout, actn)
        self.mab2 = TransformerBlock(d_model, num_heads, dim_feedforward, dropout, actn)

    def forward(self, x, mask=None):
        s = self.induced_set().view(*[1]*(x.ndim-2), *self.induced_set().shape)
        s = s.repeat(*x.shape[:-2], 1, 1) # [B, ..., Ls, d_model]
        h,_ = self.mab1(s, x, x, mask, False) # [B, ..., Ls, d_model]
        output,_ = self.mab2(x, h, h, mask, False) # [B, ..., Lx, d_model]
        return output
    
class PMA(nn.Module):
    "Pooling by  multihead attention"
    def __init__(
            self,
            num_seeds, # output sequence length
            d_model=256,
            num_heads=8,
            dim_feedforward=1024,
            dropout=0.,
            actn=nn.SiLU(),
        ):
        super().__init__()
        self.seeds = ParameterModule(torch.FloatTensor(num_seeds, d_model), True) # seed querys
        nn.init.xavier_uniform_(self.seeds.data)

        self.mab = TransformerBlock(d_model, num_heads, dim_feedforward, dropout, actn)
        
    def forward(self, x):
        seeds = self.seeds().view(*[1]*(x.ndim-2), *self.seeds().shape)
        seeds = seeds.repeat(*x.shape[:-2], 1, 1) # [B, ..., Ls, d_model]
        output,_ = self.mab(seeds, x, x)
        return output

class AP(nn.Module):
    "Attention-based pooling"
    def __init__(
            self,
            num_seeds,
            d_model=256
        ):
        super().__init__()
        self.seeds = ParameterModule(torch.FloatTensor(num_seeds, d_model), True) # seed querys
        nn.init.xavier_uniform_(self.seeds.data)

    def forward(self, x):
        seeds = self.seeds().view(*[1]*(x.ndim-2), *self.seeds().shape)
        seeds = seeds.repeat(*x.shape[:-2], 1, 1) # [B, ..., Ls, d_model]
        weights = torch.softmax(seeds.bmm(x.transpose(-1,-2)), dim=-1) # [B, ..., Ls, Lx]
        return weights.bmm(x) # [B, ..., Ls, d_model]
    
############################
## ODEFORMER-PLUS MODULES ##
############################

class ODEFormerPlusDecoder(nn.Module):
    def __init__(
            self,
            codebook_sz=1024,
            d_model=256,
            num_heads=8,
            dim_feedforward=1024,
            dropout=0.,
            actn=nn.SiLU(),
            num_layers=6,
            embedder_type='symformer', # 'sme', 'symformer'
            device=None
    ):
        super().__init__()
        # Codebook
        self.codebook = nn.Embedding(codebook_sz, d_model)
        nn.init.xavier_uniform_(self.codebook.weight.data)

        # Symbol decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=actn,
                batch_first=True
            ),
            num_layers=num_layers
        )

        # Embedder
        self.embedder_type = embedder_type
        if embedder_type == 'sme':
            self.embedder = SMEEmbedder(d_model, device=device)
        else:
            self.embedder = SymFormerEmbedder(d_model, device=device)

        # Positional encoding
        self.PE = SinePositionalEncoding(d_model, device=device)

        # Prediction head
        if self.embedder_type == 'symformer':
            self.num_head = nn.Linear(d_model, 1)

        self.sym_head = nn.Linear(d_model, self.embedder.vocab_sz)

        self.device = device
        self.to(device)

    def codebook_forward(self, x: torch.Tensor, beta: float=0.25):
        # x: [B, T, d_model]
        info = {'idx': None, 'code': None, 'loss': None}
        # find the nearest codebook vector
        codes = self.codebook.weight.data.detach() # [codebook_sz, d_model]
        dists = torch.cdist(x.detach(), codes[None,...]) # [B, T, codebook_sz]
        info['idx'] = torch.argmin(dists, dim=-1).long() # [B, T]
        info['code'] = x + (self.codebook(info['idx']) - x).detach() # straight-through estimator, [B, T, d_model]
        
        mse = nn.MSELoss()
        c = self.codebook(info['idx'])
        info['loss'] = beta * mse(x, c.detach()) + mse(x.detach(), c)
        return info

    def forward(self, x: torch.Tensor, y: List[List[str]]):
        # Embed target
        y_input = [['<sos>'] + yi for yi in y]
        y_target = [yi + ['<eos>'] for yi in y]

        y_embed = self.embedder.embed_symbols(y_input, False)[0] # [B, Ty, embed_dim]
        y_embed = self.PE(y_embed)
        
        y_info = self.embedder.embed_symbols(y_target, True) # y_info = {'idx': [...], 'num': [...], 'mask': [...]}

        # Decoder
        tgt_msk = nn.Transformer.generate_square_subsequent_mask(y_embed.shape[1]).to(self.device)
        dec_embed = self.decoder(tgt=y_embed, memory=x, tgt_mask=tgt_msk, tgt_is_causal=True) # [B, Ty, embed_dim]

        # Prediction head
        if self.embedder_type == 'symformer':
            pred_num = self.num_head(dec_embed) # [B, Ty, 1]
            pred_sym = self.sym_head(dec_embed) # [B, Ty, vocab_sz]
            return pred_num, pred_sym, y_info
        else:
            pred_sym = self.sym_head(dec_embed) # [B, Ty, vocab_sz]
            return pred_sym, y_info
   
    
class ODEFormerPlusEncoder(nn.Module):
    def __init__(
            self,
            code_len=100,
            d_model=256,
            num_heads=8,
            dim_feedforward=1024,
            dropout=0,
            actn=nn.SiLU(),
            num_layers_sym=4,
            num_layers_num=4,
            embedder_type_sym='symformer',
            embedder_type_num='symformer',
            pool_type='pma',
            with_codebook_transformer=False,
            codebook_sz=1024,
            num_layers_code=4,
            device=None
    ):
        super().__init__()
        # Symbol encoder
        _encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=actn,
            batch_first=True
        )
        self.encoder_sym = nn.TransformerEncoder(
            encoder_layer=_encoder_layer,
            num_layers=num_layers_sym
        )
        del _encoder_layer

        # Number encoder
        _encoder_layer = ISAB(
            num_inds=100,
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            actn=actn)
        self.encoder_num = nn.ModuleList(
            [
                deepcopy(_encoder_layer) for _ in range(num_layers_num)
            ]
        )
        del _encoder_layer
        
        # Encoder pooling
        assert pool_type in ['ap', 'pma']
        if pool_type == 'ap':
            self.pool_sym = AP(code_len, d_model)
            self.pool_num = AP(code_len, d_model)
        else:
            self.pool_sym = PMA(
                code_len, d_model, num_heads, dim_feedforward, dropout, actn
            )
            self.pool_num = PMA(
                code_len, d_model, num_heads, dim_feedforward, dropout, actn
            )

        # Embedder
        assert embedder_type_sym in ['sme', 'symformer']
        assert embedder_type_num in ['sme', 'symformer']
        if embedder_type_sym == 'sme':
            self.embedder_sym = SMEEmbedder(d_model, device=device)
        else:
            self.embedder_sym = SymFormerEmbedder(d_model, device=device)
    
        if embedder_type_num == 'sme':
            self.embedder_num = SMEEmbedder(d_model, device=device)
        else:
            self.embedder_num = SymFormerEmbedder(d_model, device=device)

        # Positional encoding
        self.PE = SinePositionalEncoding(d_model, device=device)

        # FFN
        if embedder_type_num == 'sme':
            self.ffn = nn.Sequential(
                nn.Linear(d_model * 3 * (MAX_XDIM + 1), d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model)
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model * (MAX_XDIM + 1), d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model)
            )
        
        # Codebook transformer
        if with_codebook_transformer:
            self.codebook_transformer = nn.TransformerEncoder(
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=actn,
                    batch_first=True
                ),
                num_layers = num_layers_code
            )
            self.codebook_ffn = nn.Linear(d_model, codebook_sz)
    
        self.device = device
        self.to(device)

    def forward(self, x: List[List[np.ndarray]], y: List[List[str]]):
        """
        Args
        ----
        x: List[List[np.ndarray]]
            Input data with shape [B, [N, [L, D]]], where:
            - B is the batch size
            - N is the number of trajectories per sample (variable)
            - L is the length of each trajectory (variable) 
            - D is the dimension of each point (variable)

        y: List[List[str]]
            Target expressions with shape [B, [M]], where:
            - B is the batch size
            - M is the length of each expression (variable)
        """
        # Embed input
        x_embed = self.embedder_num.embed_numbers(x, False)[0] # [B, Tx, X, embed_dim], X = 3D' or D', D' = MAX_XDIM + 1
        x_embed = self.ffn(x_embed.flatten(2)) # [B, Tx, embed_dim]

        for layer in self.encoder_num:
            x_embed = layer(x_embed)
        x_embed = self.pool_num(x_embed) # [B, code_len, embed_dim]
        
        # Embed target
        y_embed = self.embedder_sym.embed_symbols(y, False)[0] # [B, Ty, embed_dim]
        y_embed = self.PE(y_embed)
        y_embed = self.pool_sym(y_embed) # [B, code_len, embed_dim]

        # Fusion
        xy_embed = x_embed + y_embed

        info = {'embed_sym': x_embed, 'embed_num': y_embed, 'embed_fuse': xy_embed}
        # code transformer
        if hasattr(self, 'codebook_transformer'):
            code_logits = self.codebook_transformer(xy_embed)
            code_logits = self.codebook_ffn(code_logits) # [B, code_len, codebook_sz]
            info['code_logits'] = code_logits
        return info
