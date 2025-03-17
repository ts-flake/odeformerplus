import torch
import torch.nn as nn
from odeformerplus.envs.symbolic import UNARY_OPERATORS, BINARY_OPERATORS
from odeformerplus.envs.tokenizers import SMETokenizer, SymFormerTokenizer
from typing import List
import numpy as np

from odeformerplus.envs.utils import load_config

config = load_config('odeformerplus/envs/config.yaml')

MAX_XDIM = config.expression_tree.max_xdim

SPECIAL_VOCABS = [
    '<pad>',
    '<null>',
    '<mask>',
    '|',
    ';',
    '<sos>',
    '<eos>',
    '<args>',
    '</args>'
    '<odes>',
    '</odes>',
    '<const>'
]

SYMBOL_VOCABS = [f'x{i}' for i in range(1, config.expression_tree.max_xdim+1)] + ['t']

OPERATOR_VOCABS = list(UNARY_OPERATORS.keys()) + list(BINARY_OPERATORS.keys())


class SMEEmbedder(nn.Module):
    """
    Sign-mantissa-exponent embedder.
    """
    def __init__(self, embed_dim=256, device=None):
        super().__init__()
        self.tokenizer = SMETokenizer()
        self.vocabs = SPECIAL_VOCABS + SYMBOL_VOCABS + OPERATOR_VOCABS + self.tokenizer.vocabs
        self.vocab_sz = len(self.vocabs)
        self.vocab2idx = dict(zip(self.vocabs, range(self.vocab_sz)))
        self.idx2vocab = dict(zip(range(self.vocab_sz), self.vocabs))

        self.embed = nn.Embedding(self.vocab_sz, embed_dim, padding_idx=self.vocab2idx['<pad>'], device=device)
        self.device = device

    def embed_numbers(self, batch: List[List[np.ndarray]], return_info_only=False):
        """
        Embed pure numeric inputs.

        Args
        ----
        batch: List[List[NDArray]], shape [B, [N, [L, D]]], where:
            - B is the batch size
            - N is the number of trajectories per sample (variable)
            - L is the length of each trajectory (variable)
            - D is the dimension of each point (variable)

        Returns
        -------
        If return_info_only=True:
        Dict containing:
        - 'idx': LongTensor(token indices), shape [B, T, 3D']

        Else:
        Tuple containing:
        - Tensor(embeddings), shape [B, T, 3D', embed_dim] where:
            - T is the max length after padding
            - D' = MAX_XDIM + 1
            - embed_dim is the embedding dimension
        - Dict containing the same info as above
        """
        T = max(sum(bi.shape[0] for bi in b) for b in batch)
        Dmax = MAX_XDIM + 1

        batch_pad = []
        for b in batch:
            b_stack = np.vstack(b) # [NL, D]
            tokens = np.stack(
                [np.vectorize((lambda x: self.tokenizer.encode(x)[i]))(b_stack) for i in range(3)],
                axis=2
            )
            idxs = np.vectorize(self.vocab2idx.get)(tokens) # [NL, D, 3]

            # padding
            T_pad = np.ones((T-idxs.shape[0], idxs.shape[1], idxs.shape[2])) * self.vocab2idx['<pad>']
            D_pad = np.ones((T, Dmax-idxs.shape[1], idxs.shape[2])) * self.vocab2idx['<pad>']
            idxs = np.concatenate([idxs, T_pad], axis=0)
            idxs = np.concatenate([idxs, D_pad], axis=1) # [T, D', 3]

            batch_pad.append(idxs)
        
        # embedding
        batch_pad = torch.LongTensor(np.stack(batch_pad)) # [B, T, D', 3]
        batch_pad = batch_pad.flatten(2).to(self.device) # [B, T, 3D']
        if return_info_only:
            return {'idx': batch_pad}
        return self.embed(batch_pad), {'idx': batch_pad} # [B, T, 3D', embed_dim]
    
    def embed_symbols(self, batch: List[List[str]], return_info_only=False):
        """
        Embed numeric and symbolic inputs.

        Args
        ----
        batch: List[List[str]], shape [B, [L]], where:
            - B is the batch size
            - L is the length of the expression (variable)

        Returns
        -------
        If return_info_only=True:
        Dict containing:
        - 'idx': LongTensor(token indices), shape [B, T]
        
        Else:
        Tuple containing:
        - Tensor(embeddings), shape [B, T, embed_dim], where:
            - T is the max length after padding
            - embed_dim is the embedding dimension
        - Dict containing the same info as above
        """
        def is_number(x):
            try: float(x); return True
            except: return False
        
        # tokenize any number (str) in the expression
        # and find the max sequence length T
        T = 0
        batch_tokenized = []
        for b in batch:
            _tok= []
            for bi in b:
                if is_number(bi):
                    _tok.extend(self.tokenizer.encode(float(bi)))
                else:
                    _tok.append(bi)
            batch_tokenized.append(_tok)
            T = max(T, len(_tok))
        
        # padding
        batch_pad = []
        for b in batch_tokenized:
            idxs = np.vectorize(self.vocab2idx.get)(b)
            T_pad = np.ones(T-idxs.shape[0]) * self.vocab2idx['<pad>']
            idxs = np.concatenate([idxs, T_pad]) # [T,]
            batch_pad.append(idxs)
        
        # embedding
        batch_pad = torch.LongTensor(np.stack(batch_pad)).to(self.device) # [B, T]
        
        if return_info_only:
            return {'idx': batch_pad}
        else:
            return self.embed(batch_pad), {'idx': batch_pad} # [B, T, embed_dim]
    

class SymFormerEmbedder(nn.Module):
    """
    SymFormer embedder.
    """
    def __init__(self, embed_dim=256, device=None):
        super().__init__()
        self.tokenizer = SymFormerTokenizer()
        self.vocabs = SPECIAL_VOCABS + SYMBOL_VOCABS + OPERATOR_VOCABS + self.tokenizer.vocabs
        self.vocab_sz = len(self.vocabs)
        self.vocab2idx = dict(zip(self.vocabs, range(self.vocab_sz)))
        self.idx2vocab = dict(zip(range(self.vocab_sz), self.vocabs))

        self.embed = nn.Embedding(self.vocab_sz, embed_dim, padding_idx=self.vocab2idx['<pad>'], device=device)
        self.device = device
    
    def embed_numbers(self, batch: List[List[np.ndarray]], return_info_only=False):
        """
        Embed pure numeric inputs.

        Args
        ----
        batch: List[List[NDArray]], shape [B, [N, [L, D]]], where:
            - B is the batch size
            - N is the number of trajectories per sample (variable)
            - L is the length of each trajectory (variable)
            - D is the dimension of each point (variable)

        Returns
        -------
        If return_info_only=True:
        Dict containing:
        - 'idx': LongTensor(token indices), shape [B, T, D']
        - 'num': FloatTensor(numeric values), shape [B, T, D']
        
        Else:
        Tuple containing:
        - Tensor(embeddings), shape [B, T, D', embed_dim], where:
            - T is the max length after padding
            - D' = MAX_XDIM + 1
            - embed_dim is the embedding dimension
        - Dict containing the same info as above
        """
        T = max(sum(bi.shape[0] for bi in b) for b in batch)
        Dmax = MAX_XDIM + 1

        batch_pad = {'idx': [], 'num': []}
        for b in batch:
            b_stack = np.vstack(b) # [NL, D]
            nums = np.vectorize((lambda x: self.tokenizer.encode(x)[0]))(b_stack) # [NL, D]
            tokens = np.vectorize((lambda x: self.tokenizer.encode(x)[1]))(b_stack)
            idxs = np.vectorize(self.vocab2idx.get)(tokens) # [NL, D]
            
            # padding
            T_pad = np.ones((T-idxs.shape[0], idxs.shape[1])) * self.vocab2idx['<pad>']
            D_pad = np.ones((T, Dmax-idxs.shape[1])) * self.vocab2idx['<pad>']
            idxs = np.concatenate([idxs, T_pad], axis=0)
            idxs = np.concatenate([idxs, D_pad], axis=1) # [T, D']

            nums = np.concatenate([nums, np.ones_like(T_pad)], axis=0)
            nums = np.concatenate([nums, np.ones_like(D_pad)], axis=1) # [T, D']

            batch_pad['idx'].append(idxs)
            batch_pad['num'].append(nums)
        
        # embedding
        batch_pad['idx'] = torch.LongTensor(np.stack(batch_pad['idx'])).to(self.device) # [B, T, D']
        batch_pad['num'] = torch.FloatTensor(np.stack(batch_pad['num'])).to(self.device) # [B, T, D']

        if return_info_only:
            return {'idx': batch_pad['idx'], 'num': batch_pad['num']}
        else:
            return self.embed(batch_pad['idx']) * batch_pad['num'][..., None], {'idx': batch_pad['idx'], 'num': batch_pad['num']} # [B, T, D', embed_dim]
    
    def embed_symbols(self, batch: List[List[str]], return_info_only=False):
        """
        Embed numeric and symbolic inputs.

        Args
        ----
        batch: List[List[str]], shape [B, [L]], where:
            - B is the batch size
            - L is the length of the expression (variable)

        Returns
        -------
        If return_info_only=True:
        Dict containing:
            - 'idx': LongTensor(token indices), shape [B, T]
            - 'num': FloatTensor(numeric values), shape [N], where N is the total number of numeric tokens
            - 'mask': BoolTensor(numeric token mask), shape [B, T]
        
        Else:
        Tuple containing:
            - Tensor(embeddings), shape [B, T, embed_dim], where:
                - T is the max length after padding
                - embed_dim is the embedding dimension
            - Dict containing the same info as above
        """
        def is_number(x):
            try: float(x); return True
            except: return False
        
        # tokenize any number (str) in the expression
        # and find the max sequence length T
        T = 0
        batch_tokenized = {'token': [], 'num': [], 'mask': []}
        for b in batch:
            _tok = []
            _num = []
            _msk = []
            for bi in b:
                if is_number(bi):
                    res = self.tokenizer.encode(float(bi))
                    _tok.append(res[1])
                    _num.append(res[0])
                    _msk.append(True)
                else:
                    _tok.append(bi)
                    _msk.append(False)
            batch_tokenized['token'].append(_tok)
            batch_tokenized['num'].append(_num)
            batch_tokenized['mask'].append(_msk)
            T = max(T, len(b))
        
        # padding
        batch_pad = {'idx': [], 'mask': []}
        for bt,bm in zip(batch_tokenized['token'], batch_tokenized['mask']):
            idxs = np.vectorize(self.vocab2idx.get)(bt)
            T_pad = np.ones(T-idxs.shape[0]) * self.vocab2idx['<pad>']
            idxs = np.concatenate([idxs, T_pad])       # [T,]
            msks = np.concatenate([bm, np.zeros_like(T_pad, dtype=bool)]) # [T,]
            batch_pad['idx'].append(idxs)
            batch_pad['mask'].append(msks)
        
        # embedding
        batch_pad['idx'] = torch.LongTensor(np.stack(batch_pad['idx'])).to(self.device)   # [B, T]
        batch_pad['mask'] = torch.BoolTensor(np.stack(batch_pad['mask'])).to(self.device) # [B, T], True indicates is_number
        nums = torch.FloatTensor(np.concatenate(batch_tokenized['num'])).to(self.device)  # [L,]
        
        if return_info_only:
            return {'num': nums, 'idx': batch_pad['idx'], 'mask': batch_pad['mask']}
        else:
            emb = self.embed(batch_pad['idx']) # [B, T, embed_dim]
            emb[batch_pad['mask'], :] = emb[batch_pad['mask'], :] * nums[:, None]
            return emb, {'num': nums, 'idx': batch_pad['idx'], 'mask': batch_pad['mask']}
        





        

