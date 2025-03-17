import numpy as np
import math
import torch
import torch.nn as nn

from copy import deepcopy

from .modules import *
from .embedders import *
from odeformerplus.envs.dataset import add_noise_to_trajs, subsample_trajs, rescale_trajs, subsample_exprs
from odeformerplus.envs.utils import load_config
from .utils import freeze_params, unfreeze_params

config = load_config('odeformerplus/envs/config.yaml')

MAX_XDIM = config.expression_tree.max_xdim
TIME_RANGR_MIN = config.dataset_generator.traj_time_range_l
TIME_RANGR_MAX = config.dataset_generator.traj_time_range_u
X0_RADIUS = config.dataset_generator.x0_range_traj

############################
## ODEFORMER-PLUS-VANILLA ##
############################

class ODEFormerPlus_vanilla(nn.Module):
    def __init__(
            self,
            num_inds=100,
            num_seeds=100,
            d_model=256,
            num_heads=8,
            dim_feedforward=1024,
            dropout=0.,
            actn=nn.GELU(),
            num_layers_enc=4,
            num_layers_dec=8,
            enc_pool_type='pma', # 'ap', 'pma'
            embedder_type_enc='symformer', # 'sme', 'symformer'
            embedder_type_dec='symformer',
            share_embedder=False, # sharing embedder with encoder and decoder
            device=None
    ):
        super().__init__()
        self.d_model = d_model
        self.embedder_type_enc = embedder_type_enc
        self.embedder_type_dec = embedder_type_dec

        # Encoder
        _encoder_layer = ISAB(
            num_inds, d_model, num_heads, dim_feedforward, dropout, actn
        )
        self.encoder = nn.ModuleList([
            deepcopy(_encoder_layer) for _ in range(num_layers_enc)
        ])
        del _encoder_layer

        # Encoder pooling layer
        assert enc_pool_type in ['ap', 'pma']
        if enc_pool_type == 'ap':
            self.encoder_pool = AP(num_seeds, d_model)
        else:
            self.encoder_pool = PMA(
                num_seeds, d_model, num_heads, dim_feedforward, dropout, actn
            )

        # Decoder
        _decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=actn,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=_decoder_layer,
            num_layers=num_layers_dec
        )
        del _decoder_layer

        # Positional encoder
        self.PE = SinePositionalEncoding(d_model, device=device)

        # Embedder
        assert embedder_type_enc in ['sme', 'symformer']
        assert embedder_type_dec in ['sme', 'symformer']
       
        if not share_embedder:
            if embedder_type_enc == 'sme':
                self.embedder_enc = SMEEmbedder(d_model, device=device)
            else:
                self.embedder_enc = SymFormerEmbedder(d_model, device=device)
        
            if embedder_type_dec == 'sme':
                self.embedder_dec = SMEEmbedder(d_model, device=device)
            else:
                self.embedder_dec = SymFormerEmbedder(d_model, device=device)
        else:
            assert embedder_type_enc == embedder_type_dec
            if embedder_type_enc == 'sme':
                self.embedder_enc = self.embedder_dec = SMEEmbedder(d_model, device=device)
            else:
                self.embedder_enc = self.embedder_dec = SymFormerEmbedder(d_model, device=device)

        # FFN
        if embedder_type_enc == 'sme':
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
        
        # Prediction head
        if embedder_type_dec == 'symformer':
            self.num_head = nn.Linear(d_model, 1)
        
        self.sym_head = nn.Linear(d_model, self.embedder_dec.vocab_sz)
        
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
            - D is the dimension of each point including time (variable)

        y: List[List[str]]
            Target expressions with shape [B, [M]], where:
            - B is the batch size
            - M is the length of each expression (variable)
        """
        # Embed input
        x_embed = self.embedder_enc.embed_numbers(x, False)[0] # [B, Tx, X, embed_dim], X = 3D' or D', D' = MAX_XDIM + 1

        # Reduce dimension
        x_embed = self.ffn(x_embed.flatten(2)) # [B, Tx, embed_dim]

        # Encoder
        for layer in self.encoder:
            x_embed = layer(x_embed)
        
        # Encoder pooling
        x_embed = self.encoder_pool(x_embed) # [B, num_seeds, embed_dim]
        
        # Embed target
        y_input = [['<sos>'] + yi for yi in y]
        y_target = [yi + ['<eos>'] for yi in y]

        y_embed = self.embedder_dec.embed_symbols(y_input, False)[0] # [B, Ty, embed_dim]
        y_embed = self.PE(y_embed)
        
        y_info = self.embedder_dec.embed_symbols(y_target, True) # y_info = {'idx': [...], ['num': [...], 'mask': [...]]}

        # Decoder
        tgt_msk = nn.Transformer.generate_square_subsequent_mask(y_embed.shape[1]).to(self.device)
        dec_embed = self.decoder(tgt=y_embed, memory=x_embed, tgt_mask=tgt_msk, tgt_is_causal=True) # [B, Ty, embed_dim]

        # Prediction head
        if self.embedder_type_dec == 'symformer':
            pred_num = self.num_head(dec_embed) # [B, Ty, 1]
            pred_sym = self.sym_head(dec_embed) # [B, Ty, vocab_sz]
            return pred_num, pred_sym, y_info
        else:
            pred_sym = self.sym_head(dec_embed) # [B, Ty, vocab_sz]
            return pred_sym, y_info

    def train_step(self, x: List[List[np.ndarray]], y: List[List[str]], optimizer, noise_sig=0.05, drop_rate=0.2):
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

        optimizer: Optimizer
            Optimizer
        noise_sig: float
            Noise standard deviation
        drop_rate: float
            Drop rate
        """
        self.train()
        optimizer.zero_grad()
        
        _x = []
        # Add noise and subsample input
        for xi in x:
            if noise_sig > 0: xi = add_noise_to_trajs(xi, noise_sig, deterministic=False)
            if drop_rate > 0: xi = subsample_trajs(xi, drop_rate, deterministic=False)
            _x.append(xi)
        
        # Forward
        if self.embedder_type_dec == 'symformer':
            pred_num, pred_sym, y_info = self.forward(_x, y)
            loss_ce = nn.CrossEntropyLoss(ignore_index=self.embedder_dec.vocab2idx['<pad>'])(pred_sym.transpose(1,2), y_info['idx'])
            loss_mse = nn.MSELoss()(pred_num[y_info['mask'], :], y_info['num'][:, None])
            loss = loss_ce + loss_mse
        else:
            pred_sym, y_info = self.forward(_x, y)
            loss = nn.CrossEntropyLoss(ignore_index=self.embedder_dec.vocab2idx['<pad>'])(pred_sym.transpose(1,2), y_info['idx'])
        
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.inference_mode()
    def inference(
            self, 
            x: List[np.ndarray], 
            noise_sig=0, 
            drop_rate=0, 
            decode_strategy='greedy', 
            rescale_input=True,
            max_seq_len=200, 
            num_beams=20, 
            temperature=0.1, 
            top_k=50
    ):
        """
        Infer for a list of trajectories (unbatched).
        """        
        # Rescale input
        if rescale_input:
            _x, info = rescale_trajs(x, X0_RADIUS, [TIME_RANGR_MIN, TIME_RANGR_MAX])
        else:
            _x = deepcopy(x)
            info = None

        # Add noise and subsample input
        if noise_sig > 0: _x = add_noise_to_trajs(_x, noise_sig, deterministic=True)
        if drop_rate > 0: _x = subsample_trajs(_x, drop_rate, deterministic=True)

        self.eval()
        # Embed input
        x_embed = self.embedder_enc.embed_numbers([_x], False)[0] # [1, Tx, X, embed_dim], X = 3D' or D', D' = MAX_XDIM + 1
        x_embed = self.ffn(x_embed.flatten(2)) # [1, Tx, embed_dim]

        # Encoder
        for layer in self.encoder:
            x_embed = layer(x_embed)
        
        # Encoder pooling
        x_embed = self.encoder_pool(x_embed) # [1, num_seeds, embed_dim]

        # Decode
        if decode_strategy == 'greedy':
            output = self.greedy_decode(x_embed, max_seq_len)
            return {
                'output': output,
                'info': info
            }
        elif decode_strategy == 'beam':
            from .utils import integrate_ode, calculate_R2
            outputs = self.beam_decode(x_embed, num_beams, max_seq_len, temperature, top_k)
            # Pick the best output
            best_output = outputs[0]
            best_r2 = -np.inf
            for output in outputs:
                try:
                    ode_trajs = integrate_ode(output, [xi[0,1:] for xi in _x], [xi[:,0] for xi in _x])
                except Exception:
                    continue
                try:
                    r2s = calculate_R2(ode_trajs, _x)
                    r2 = np.mean(r2s)
                except Exception:
                    continue
                if r2 > best_r2:
                    best_r2 = r2
                    best_output = output
            return {
                'output': best_output,
                'beam_outputs': outputs,
                'info': info
            }

        else:
            raise ValueError(f'Invalid decode strategy: {decode_strategy}')

    
    @torch.inference_mode()
    def greedy_decode(self, x_embed, max_seq_len=200):
        assert x_embed.shape[0] == 1  # batch size must be 1
        # Initialize decoding sequences
        dec_input = [['<sos>']]

        # Greedy decoding
        for _ in range(max_seq_len):
            # Create causal mask
            tgt_len = len(dec_input[0])
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(self.device)
            
            # Get decoder input embeddings
            dec_embed = self.embedder_dec.embed_symbols(dec_input, False)[0]
            dec_embed = self.PE(dec_embed)
            
            # Forward pass through decoder
            dec_embed = self.decoder(tgt=dec_embed, memory=x_embed, tgt_mask=tgt_mask, tgt_is_causal=True)
                
            # Predict next token
            if self.embedder_type_dec == 'symformer':
                num_pred = self.num_head(dec_embed[:,-1:,:]).item() # [1,]
                sym_pred = self.sym_head(dec_embed[:,-1:,:])[0,0,:]  # [vocab_sz,]
                sym_pred = torch.argmax(sym_pred, dim=-1).item()    # [1,]

                sym_token = self.embedder_dec.idx2vocab[sym_pred] # [1,]
                
                # Decode number tokens
                if sym_token in self.embedder_dec.tokenizer.vocabs: # is a number
                    sym_token = str(round(self.embedder_dec.tokenizer.decode([num_pred, sym_token]), 4))

                # Append to decoder input
                dec_input[0].append(sym_token)
            else:
                # TODO: fix this
                sym_pred = self.sym_head(dec_embed[:,-1:,:])[0,0,:]  # [vocab_sz,]
                sym_pred = torch.argmax(sym_pred, dim=-1).item()    # [1,]
                
                sym_token = self.embedder_dec.idx2vocab[sym_pred] # [1]
                dec_input[0].append(sym_token)
                
            # Check if finished
            if '<eos>' in dec_input[0]:
                break

        # Truncate at <eos> token
        output = dec_input[0]
        try:
            eos_idx = output.index('<eos>')
            output = output[1:eos_idx] # exclude <sos> and <eos>
        except ValueError:
            output = output[1:] # exclude <sos>
        
        if self.embedder_type_dec == 'sme':
            _output = []
            for idx,tok in enumerate(output):
                if tok in ['+', '-']:
                    try:
                        num = self.embedder_dec.tokenizer.decode(output[idx:idx+3])
                        _output.append(str(round(float(num), 4)))
                    except:
                        raise Exception(f'SME decoding error at: {output[idx:idx+3]}')
                else:
                    _output.append(tok)
            output = _output

        return output
    
    @torch.inference_mode()
    def beam_decode(self, x_embed, num_beams=20, max_len=200, temperature=0.1, top_k=50):
        """
        Beam sampling.
        """
        assert x_embed.shape[0] == 1  # batch size must be 1
        
        # Initialize beams
        x_embed = x_embed.repeat(num_beams, 1, 1) # [num_beams, Tx, embed_dim]

        dec_input = [['<sos>'] for _ in range(num_beams)]  # list of sequences, one per beam
        beam_scores = [0 for _ in range(num_beams)]  # for each beam
        
        finished_beams = []
        finished_scores = []

        for _ in range(max_len):
            if len(dec_input) == 0:
                break

            # Forward pass through decoder
            dec_embed = self.embedder_dec.embed_symbols(dec_input, False)[0] # [num_beams, L, embed_dim]
            dec_embed = self.PE(dec_embed)

            tgt_len = dec_embed.shape[1]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(self.device)
            dec_embed = self.decoder(tgt=dec_embed, memory=x_embed, tgt_mask=tgt_mask, tgt_is_causal=True) # [num_beams, L, embed_dim]

            # Get logits
            if self.embedder_type_dec == 'symformer':
                num_pred = self.num_head(dec_embed[:,-1:,:])  # [num_beams, 1, 1]
                sym_pred = self.sym_head(dec_embed[:,-1:,:])  # [num_beams, 1, vocab_sz]
                logits = sym_pred / temperature
            else:
                sym_pred = self.sym_head(dec_embed[:,-1:,:])  # [num_beams, 1, vocab_sz]
                logits = sym_pred / temperature
                
            # Select top-k
            top_k_logits, top_k_idxs = torch.topk(logits[:,0,:], top_k, dim=-1)  # [num_beams, top_k]
            probs = torch.nn.functional.softmax(top_k_logits, dim=-1)  # [num_beams, top_k]

            # Sample from top-k
            sampled_idxs = torch.multinomial(probs, 1)  # [num_beams, 1]
            sampled_probs = torch.gather(probs, -1, sampled_idxs).squeeze(-1).cpu().numpy()  # [num_beams]
            sampled_token_idxs = torch.gather(top_k_idxs, -1, sampled_idxs).squeeze(-1).cpu().numpy()  # [num_beams]

            # Update decoder input
            if self.embedder_type_dec == 'symformer':
                for beam,idx in enumerate(sampled_token_idxs):
                    tok = self.embedder_dec.idx2vocab[idx]
                    if tok in self.embedder_dec.tokenizer.vocabs:
                        num = self.embedder_dec.tokenizer.decode([num_pred[beam,0,0].item(), tok])
                        dec_input[beam].append(str(round(float(num), 4)))
                    else:
                        dec_input[beam].append(tok)
            else:
                for beam, idx in enumerate(sampled_token_idxs):
                    tok = self.embedder_dec.idx2vocab[idx]
                    dec_input[beam].append(tok)

            # Update beam scores
            beam_scores = [score + np.log(prob) for score, prob in zip(beam_scores, sampled_probs)]

            # Check if any sequences are finished
            # Note: after this, num_beams is changed in the outer loop
            for beam, sequence in enumerate(dec_input):
                if '<eos>' in sequence:
                    finished_beams.append(sequence)
                    finished_scores.append(beam_scores[beam])
                    # remove finished beam
                    dec_input.pop(beam)
                    beam_scores.pop(beam)
                    x_embed = x_embed[:-1,:,:]

        # Add any unfinished sequences to the finished list
        finished_beams.extend(dec_input)
        finished_scores.extend(beam_scores)
            
        # Sort beams by scores
        sorted_idxs = np.argsort(finished_scores)[::-1] # descending
        best_beams = [finished_beams[i] for i in sorted_idxs]
        # Process sequences to remove <sos> and truncate at <eos>
        outputs = []
        for beam in best_beams:
            try:
                eos_idx = beam.index('<eos>')
                outputs.append(beam[1:eos_idx])  # exclude <sos> and <eos>
            except ValueError:
                outputs.append(beam[1:])  # exclude <sos>
            
        # Check for SMEs
        if self.embedder_type_dec == 'sme':
            _outputs = []
            for beam in outputs:
                _beam = []
                for idx,tok in enumerate(beam):
                    if tok in ['+', '-']:
                        try:
                            num = self.embedder_dec.tokenizer.decode(beam[idx:idx+3])
                            _beam.append(str(round(float(num), 4)))
                        except:
                            raise Exception(f'SME decoding error at: {beam[idx:idx+3]}')
                    else:
                        _beam.append(tok)
                _outputs.append(_beam)
            outputs = _outputs

        return outputs

####################
## ODEFORMER-PLUS ##
####################

class ODEFormerPlusStage1(nn.Module):
    def __init__(
            self,
            code_len=100,
            codebook_sz=1024,
            d_model=256,
            num_heads=8,
            dim_feedforward=1024,
            dropout=0,
            actn=nn.GELU(),
            num_layers_enc_sym=4,
            num_layers_enc_num=4,
            num_layers_dec=8,
            embedder_type_enc_sym='symformer',
            embedder_type_enc_num='symformer',
            embedder_type_dec='symformer',
            pool_type='pma',
            device=None
    ):
        super().__init__()
        # Encoder
        self.encoder = ODEFormerPlusEncoder(
            code_len=code_len,
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            actn=actn,
            num_layers_sym=num_layers_enc_sym,
            num_layers_num=num_layers_enc_num,
            embedder_type_sym=embedder_type_enc_sym,
            embedder_type_num=embedder_type_enc_num,
            with_codebook_transformer=False,
            pool_type=pool_type,
            device=device
        )

        # Codebook and decoder
        self.decoder = ODEFormerPlusDecoder(
            codebook_sz=codebook_sz,
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            actn=actn,
            num_layers=num_layers_dec,
            embedder_type=embedder_type_dec,
            device=device
        )
        self.embedder_type_dec = embedder_type_dec

        self.device = device
        self.to(device)
    
    def contrastive_loss(self, embed_sym, embed_num, temperature=0.2):
        # embed_sym: [B, code_len, d_model]
        # embed_num: [B, code_len, d_model]
        assert embed_num.shape == embed_sym.shape
        
        # normalize
        _sym = torch.nn.functional.normalize(embed_sym, p=2, dim=-1) # [B, d_model]
        _num = torch.nn.functional.normalize(embed_num, p=2, dim=-1) # [B, d_model]

        loss = []
        # compute contrastive loss over sequence
        for i in range(_sym.shape[1]):
            # cosine similarity
            sim = torch.matmul(_sym[:,i,:], _num[:,i,:].T) / temperature  # [B, B]
            
            # labels
            labels = torch.arange(sim.size(0), device=sim.device)
            
            # contrastive loss
            loss_s2n = nn.CrossEntropyLoss()(sim, labels)
            loss_n2s = nn.CrossEntropyLoss()(sim.T, labels)
            loss.append((loss_s2n + loss_n2s) / 2.0)
        return sum(loss) / len(loss)

    def train_step(self, x: List[List[np.ndarray]], y: List[List[str]], optimzer, code_loss_beta=0.25, contrastive_loss_weight=0.2):
        self.train()
        optimzer.zero_grad()

        # Symbol and number encoding
        info = self.encoder(x, y)
        loss_cl = self.contrastive_loss(info['embed_sym'], info['embed_num'])

        # Codebook forward
        code_info = self.decoder.codebook_forward(info['embed_fuse'], code_loss_beta)
        loss_code = code_info['loss']

        # Symbol decoding
        if self.embedder_type_dec == 'symformer':
            pred_num, pred_sym, y_info = self.decoder.forward(code_info['code'], y)
            loss_ce = nn.CrossEntropyLoss(ignore_index=self.decoder.embedder.vocab2idx['<pad>'])(pred_sym.transpose(1,2), y_info['idx'])
            loss_mse = nn.MSELoss()(pred_num[y_info['mask'], :], y_info['num'][:, None])
            loss_dec = loss_ce + loss_mse
        else:
            pred_sym, y_info = self.decoder.forward(code_info['code'], y)
            loss_dec = nn.CrossEntropyLoss(ignore_index=self.decoder.embedder.vocab2idx['<pad>'])(pred_sym.transpose(1,2), y_info['idx'])

        loss = contrastive_loss_weight * loss_cl + loss_code + loss_dec
        loss.backward()
        optimzer.step()
        return {
            'loss': loss.item(),
            'loss_cl': loss_cl.item(),
            'loss_code': loss_code.item(),
            'loss_dec': loss_dec.item()
        }

class ODEFormerPlusStage2(nn.Module):
    def __init__(
            self,
            stage1,
            code_len=100,
            codebook_sz=1024,
            d_model=256,
            num_heads=8,
            dim_feedforward=1024,
            dropout=0,
            actn=nn.GELU(),
            num_layers_enc_sym=4,
            num_layers_enc_num=4,
            num_layers_code=4,
            embedder_type_enc_sym='symformer',
            embedder_type_enc_num='symformer',
            pool_type='pma',
            device=None
        ):
        super().__init__()
        self.stage1 = stage1
        # freeze stage 1 parameters
        freeze_params(self.stage1)

        self.encoder = ODEFormerPlusEncoder(
            code_len=code_len,
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            actn=actn,
            num_layers_sym=num_layers_enc_sym,
            num_layers_num=num_layers_enc_num,
            num_layers_code=num_layers_code,
            with_codebook_transformer=True,
            codebook_sz=codebook_sz,
            embedder_type_sym=embedder_type_enc_sym,
            embedder_type_num=embedder_type_enc_num,
            pool_type=pool_type,
            device=device
        )

        self.device = device
        self.to(device)


    def train_step(self, x_line: List[List[np.ndarray]], x_traj: List[List[np.ndarray]], y: List[List[str]], optimzer, noise_sig=0.05, drop_rate=0.2):
        self.train()
        optimzer.zero_grad()

        _x = []
        # Add noise and subsample input
        for xi in x_traj:
            if noise_sig > 0: xi = add_noise_to_trajs(xi, noise_sig, deterministic=False)
            if drop_rate > 0: xi = subsample_trajs(xi, drop_rate, deterministic=False)
            _x.append(xi)
        
        _y = []
        # Extract subexpression
        for yi in y:
            _y.append(subsample_exprs(yi))

        # Symbol and number encoding
        info = self.encoder(_x, _y)

        # Targets from stage 1
        with torch.no_grad():
            xy_embed_targ = self.stage1.encoder(x_line, y)['embed_fuse'].detach()
            targ_idx = self.stage1.decoder.codebook_forward(xy_embed_targ)['idx'].detach()

        loss_ce = nn.CrossEntropyLoss()(info['code_logits'].transpose(1,2), targ_idx)
        loss_mse = nn.MSELoss()(info['embed_fuse'], xy_embed_targ)
        loss = loss_ce + loss_mse
        loss.backward()
        optimzer.step()
        return {
            'loss': loss.item(),
            'loss_ce': loss_ce.item(),
            'loss_mse': loss_mse.item()
        }

class ODEFormerPlus(nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            device=None
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        freeze_params(self.encoder)
        freeze_params(self.decoder)

        self.device = device
        self.to(device)

    @torch.inference_mode()
    def inference(
            self, 
            x: List[np.ndarray],
            y: List[str],
            noise_sig=0, 
            drop_rate=0, 
            decode_strategy='greedy', 
            rescale_input=True,
            max_seq_len=200, 
            num_beams=20, 
            temperature=0.1, 
            top_k=50
    ):
        """
        Infer for a list of trajectories (unbatched).
        """        
        # Rescale input
        if rescale_input:
            _x, info = rescale_trajs(x, X0_RADIUS, [TIME_RANGR_MIN, TIME_RANGR_MAX])
        else:
            _x = deepcopy(x)
            info = None

        # Add noise and subsample input
        if noise_sig > 0: _x = add_noise_to_trajs(_x, noise_sig, deterministic=True)
        if drop_rate > 0: _x = subsample_trajs(_x, drop_rate, deterministic=True)

        self.eval()
        # Symbol and number encoding
        enc_info = self.encoder([_x], [y])

        # Codes
        idx = torch.argmax(enc_info['code_logits'], dim=-1).long()
        code = self.decoder.codebook(idx)


        # Decode
        if decode_strategy == 'greedy':
            output = self.greedy_decode(code, max_seq_len)
            return {
                'output': output,
                'info': info
            }
        elif decode_strategy == 'beam':
            from .utils import integrate_ode, calculate_R2
            outputs = self.beam_decode(code, num_beams, max_seq_len, temperature, top_k)
            # Pick the best output
            best_output = outputs[0]
            best_r2 = -np.inf
            for output in outputs:
                try:
                    ode_trajs = integrate_ode(output, [xi[0,1:] for xi in _x], [xi[:,0] for xi in _x])
                except Exception:
                    continue
                try:
                    r2s = calculate_R2(ode_trajs, _x)
                    r2 = np.mean(r2s)
                except Exception:
                    continue
                if r2 > best_r2:
                    best_r2 = r2
                    best_output = output
            return {
                'output': best_output,
                'beam_outputs': outputs,
                'info': info
            }

        else:
            raise ValueError(f'Invalid decode strategy: {decode_strategy}')
    
    @torch.inference_mode()
    def greedy_decode(self, x_embed, max_seq_len=200):
        assert x_embed.shape[0] == 1  # batch size must be 1
        # Initialize decoding sequences
        dec_input = [['<sos>']]

        # Greedy decoding
        for _ in range(max_seq_len):
            # Create causal mask
            tgt_len = len(dec_input[0])
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(self.device)
            
            # Get decoder input embeddings
            dec_embed = self.decoder.embedder.embed_symbols(dec_input, False)[0]
            dec_embed = self.decoder.PE(dec_embed)
            
            # Forward pass through decoder
            dec_embed = self.decoder.decoder(tgt=dec_embed, memory=x_embed, tgt_mask=tgt_mask, tgt_is_causal=True)
                
            # Predict next token
            if self.decoder.embedder_type == 'symformer':
                num_pred = self.decoder.num_head(dec_embed[:,-1:,:]).item() # [1,]
                sym_pred = self.decoder.sym_head(dec_embed[:,-1:,:])[0,0,:]  # [vocab_sz,]
                sym_pred = torch.argmax(sym_pred, dim=-1).item()    # [1,]

                sym_token = self.decoder.embedder.idx2vocab[sym_pred] # [1,]
                
                # Decode number tokens
                if sym_token in self.decoder.embedder.tokenizer.vocabs: # is a number
                    sym_token = str(round(self.decoder.embedder.tokenizer.decode([num_pred, sym_token]), 4))

                # Append to decoder input
                dec_input[0].append(sym_token)
            else:
                # TODO: fix this
                sym_pred = self.decoder.sym_head(dec_embed[:,-1:,:])[0,0,:]  # [vocab_sz,]
                sym_pred = torch.argmax(sym_pred, dim=-1).item()    # [1,]
                
                sym_token = self.decoder.embedder.idx2vocab[sym_pred] # [1]
                dec_input[0].append(sym_token)
                
            # Check if finished
            if '<eos>' in dec_input[0]:
                break

        # Truncate at <eos> token
        output = dec_input[0]
        try:
            eos_idx = output.index('<eos>')
            output = output[1:eos_idx] # exclude <sos> and <eos>
        except ValueError:
            output = output[1:] # exclude <sos>
        
        if self.decoder.embedder_type == 'sme':
            _output = []
            for idx,tok in enumerate(output):
                if tok in ['+', '-']:
                    try:
                        num = self.decoder.embedder.tokenizer.decode(output[idx:idx+3])
                        _output.append(str(round(float(num), 4)))
                    except:
                        raise Exception(f'SME decoding error at: {output[idx:idx+3]}')
                else:
                    _output.append(tok)
            output = _output

        return output
    
    @torch.inference_mode()
    def beam_decode(self, x_embed, num_beams=20, max_len=200, temperature=0.1, top_k=50):
        """
        Beam sampling.
        """
        assert x_embed.shape[0] == 1  # batch size must be 1
        
        # Initialize beams
        x_embed = x_embed.repeat(num_beams, 1, 1) # [num_beams, Tx, embed_dim]

        dec_input = [['<sos>'] for _ in range(num_beams)]  # list of sequences, one per beam
        beam_scores = [0 for _ in range(num_beams)]  # for each beam
        
        finished_beams = []
        finished_scores = []

        for _ in range(max_len):
            if len(dec_input) == 0:
                break

            # Forward pass through decoder
            dec_embed = self.decoder.embedder.embed_symbols(dec_input, False)[0] # [num_beams, L, embed_dim]
            dec_embed = self.decoder.PE(dec_embed)

            tgt_len = dec_embed.shape[1]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(self.device)
            dec_embed = self.decoder.decoder(tgt=dec_embed, memory=x_embed, tgt_mask=tgt_mask, tgt_is_causal=True) # [num_beams, L, embed_dim]

            # Get logits
            if self.decoder.embedder_type == 'symformer':
                num_pred = self.decoder.num_head(dec_embed[:,-1:,:])  # [num_beams, 1, 1]
                sym_pred = self.decoder.sym_head(dec_embed[:,-1:,:])  # [num_beams, 1, vocab_sz]
                logits = sym_pred / temperature
            else:
                sym_pred = self.decoder.sym_head(dec_embed[:,-1:,:])  # [num_beams, 1, vocab_sz]
                logits = sym_pred / temperature
                
            # Select top-k
            top_k_logits, top_k_idxs = torch.topk(logits[:,0,:], top_k, dim=-1)  # [num_beams, top_k]
            probs = torch.nn.functional.softmax(top_k_logits, dim=-1)  # [num_beams, top_k]

            # Sample from top-k
            sampled_idxs = torch.multinomial(probs, 1)  # [num_beams, 1]
            sampled_probs = torch.gather(probs, -1, sampled_idxs).squeeze(-1).cpu().numpy()  # [num_beams]
            sampled_token_idxs = torch.gather(top_k_idxs, -1, sampled_idxs).squeeze(-1).cpu().numpy()  # [num_beams]

            # Update decoder input
            if self.decoder.embedder_type == 'symformer':
                for beam,idx in enumerate(sampled_token_idxs):
                    tok = self.decoder.embedder.idx2vocab[idx]
                    if tok in self.decoder.embedder.tokenizer.vocabs:
                        num = self.decoder.embedder.tokenizer.decode([num_pred[beam,0,0].item(), tok])
                        dec_input[beam].append(str(round(float(num), 4)))
                    else:
                        dec_input[beam].append(tok)
            else:
                for beam, idx in enumerate(sampled_token_idxs):
                    tok = self.decoder.embedder.idx2vocab[idx]
                    dec_input[beam].append(tok)

            # Update beam scores
            beam_scores = [score + np.log(prob) for score, prob in zip(beam_scores, sampled_probs)]

            # Check if any sequences are finished
            # Note: after this, num_beams is changed in the outer loop
            for beam, sequence in enumerate(dec_input):
                if '<eos>' in sequence:
                    finished_beams.append(sequence)
                    finished_scores.append(beam_scores[beam])
                    # remove finished beam
                    dec_input.pop(beam)
                    beam_scores.pop(beam)
                    x_embed = x_embed[:-1,:,:]

        # Add any unfinished sequences to the finished list
        finished_beams.extend(dec_input)
        finished_scores.extend(beam_scores)
            
        # Sort beams by scores
        sorted_idxs = np.argsort(finished_scores)[::-1] # descending
        best_beams = [finished_beams[i] for i in sorted_idxs]
        # Process sequences to remove <sos> and truncate at <eos>
        outputs = []
        for beam in best_beams:
            try:
                eos_idx = beam.index('<eos>')
                outputs.append(beam[1:eos_idx])  # exclude <sos> and <eos>
            except ValueError:
                outputs.append(beam[1:])  # exclude <sos>
            
        # Check for SMEs
        if self.decoder.embedder_type == 'sme':
            _outputs = []
            for beam in outputs:
                _beam = []
                for idx,tok in enumerate(beam):
                    if tok in ['+', '-']:
                        try:
                            num = self.decoder.embedder.tokenizer.decode(beam[idx:idx+3])
                            _beam.append(str(round(float(num), 4)))
                        except:
                            raise Exception(f'SME decoding error at: {beam[idx:idx+3]}')
                    else:
                        _beam.append(tok)
                _outputs.append(_beam)
            outputs = _outputs

        return outputs