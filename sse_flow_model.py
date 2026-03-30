#%%

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is None: seq_len = x.shape[1]
        if self.cos_cached is None or self.cos_cached.shape[1] < seq_len:
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1) 
            self.register_buffer("cos_cached", emb.cos()[None, :, None, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, :, None, :], persistent=False)
        return self.cos_cached[:, :seq_len, ...], self.sin_cached[:, :seq_len, ...]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RoPEAttention(nn.Module):
    def __init__(self, d_model, n_heads, rope_emb=None):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.rope_emb = rope_emb 

    def forward(self, x, attn_mask=None):
        B, L, D = x.shape
        q = self.wq(x).view(B, L, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, L, self.n_heads, self.head_dim)
        v = self.wv(x).view(B, L, self.n_heads, self.head_dim)
        
        cos, sin = self.rope_emb(v, seq_len=L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        mask_bool = None
        if attn_mask is not None:
            mask_bool = attn_mask[:, None, None, :] > 0.5
            
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_bool)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.wo(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4.0, rope_emb=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RoPEAttention(d_model, n_heads, rope_emb=rope_emb)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model)
        )

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x

class SSEEncoderRoPE(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, pad_id: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.rope = RotaryEmbedding(dim=d_model // n_heads)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, rope_emb=self.rope) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(tokens)
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return self.norm(x)

class PairProjector(nn.Module):
    def __init__(self, d_model: int, pair_dim: int):
        super().__init__()
        self.proj = nn.Linear(d_model * 2, pair_dim)
        self.act = nn.GELU()

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        B, L, D = seq.shape
        row = seq.unsqueeze(2).expand(B, L, L, D)
        col = seq.unsqueeze(1).expand(B, L, L, D)
        pair = self.proj(torch.cat([row, col], dim=-1))
        return self.act(pair)

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000.0) / max(1, half - 1)))
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb

class FlowHead(nn.Module):

    def __init__(self, pair_dim: int, time_dim: int = 64, out_channels: int = 3):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        
        self.net = nn.Sequential(
            nn.Linear(pair_dim + out_channels + time_dim, pair_dim),
            nn.GELU(),
            nn.Linear(pair_dim, pair_dim),
            nn.GELU(),
            nn.Linear(pair_dim, out_channels)
        )

    def forward(self, pair: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, C, L, _ = x_t.shape
        x_in = x_t.permute(0, 2, 3, 1) # [B, L, L, 3]
        t_emb = self.time_emb(t).view(B, 1, 1, -1).expand(-1, L, L, -1)
        
        h = torch.cat([pair, x_in, t_emb], dim=-1)
        v = self.net(h)
        return v.permute(0, 3, 1, 2)

class FlowMatchingModel(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, d_model=128, n_layers=6, n_heads=4, pair_dim=64):
        super().__init__()
        self.encoder = SSEEncoderRoPE(vocab_size, d_model, n_layers, n_heads, pad_id=pad_id)
        self.pair_proj = PairProjector(d_model, pair_dim)
 
        self.flow_head = FlowHead(pair_dim, out_channels=3)

    def forward(self, tokens: torch.Tensor, attn_mask: torch.Tensor, 
                x_t: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None):
        
        seq = self.encoder(tokens, attn_mask)
        pair = self.pair_proj(seq)
        
        pair_mask = attn_mask[:, None, :] * attn_mask[:, :, None]
        pair = pair * pair_mask.unsqueeze(-1)
        
        out = {"pair": pair}
        
        if x_t is not None and t is not None:
            out["v_pred"] = self.flow_head(pair, x_t, t)
            
        return out

class SSETokenizer:
    def __init__(self, vocab: Optional[Sequence[str]] = None, 
                 pad_token: str = "[PAD]", unk_token: str = "[UNK]", null_token: str = "[NULL]"):
        if vocab is None:
            vocab = [pad_token, unk_token, null_token] + [f"S{i}" for i in range(52)]
        else:
            vocab += [pad_token, unk_token, null_token]
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.null_token = null_token
        self.itos = list(vocab)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    @property
    def pad_id(self): return self.stoi[self.pad_token]
    @property
    def null_id(self): return self.stoi[self.null_token]

    def encode(self, sse_list: List[str]) -> torch.Tensor:
        tokens = [self.stoi.get(t, self.stoi[self.unk_token]) for t in sse_list]
        return torch.tensor(tokens, dtype=torch.long)

@dataclass
class TopologyExample:
    prot_id: str
    sse_tokens: List[str]
    contacts: List[Tuple[int, int, float, float, int]]

def collate_batch(examples, tokenizer,train=True):
    token_tensors = [tokenizer.encode(ex.sse_tokens) for ex in examples]
    max_L = max(t.size(0) for t in token_tensors)
    
    tokens_padded, masks, target_maps, weight_maps = [], [], [], []
    k = 50
    n = 3
    for i, tok in enumerate(token_tensors):
        L = tok.size(0)
        pad_len = max_L - L
        tokens_padded.append(F.pad(tok, (0, pad_len), value=tokenizer.pad_id))
        
        masks.append(torch.cat([torch.ones(L), torch.zeros(pad_len)]))
        
        map_3c = torch.zeros((3, max_L, max_L), dtype=torch.float32)

        w_map = torch.ones((1, max_L, max_L), dtype=torch.float32)
        
        for u, v, fi, fj, count in examples[i].contacts:
            map_3c[0, u, v] = map_3c[0, v, u] = 1.0
            map_3c[1, u, v] = fi
            map_3c[2, u, v] = fj
            map_3c[1, v, u] = fj
            map_3c[2, v, u] = fi
            
            w_min = 0.2
            
            
            w_map[0, u, v] = w_map[0, v, u] = w_min+math.log(1+count,5)*(count**n/(count**n+k**n))
            
        target_maps.append(map_3c)
        weight_maps.append(w_map)

    return {
        "tokens": torch.stack(tokens_padded).long(),
        "attn_mask": torch.stack(masks).float(),
        "target_map": torch.stack(target_maps).float(),
        "weight_map": torch.stack(weight_maps).float() # [B, 1, L, L]
    }


def flow_matching_loss(v_pred, x1, x0, mask, weight_map=None):

    target_v = x1 - x0
    

    loss = F.mse_loss(v_pred, target_v, reduction='none')
    
    if weight_map is not None:

        loss = loss * weight_map

    loss_masked = loss * mask.expand_as(loss)
    
    return loss_masked.sum() / mask.sum().clamp_min(1.0)

def training_step(model, batch, optimizer, null_id, cfg_prob=0.15):
    model.train()
    tokens, attn_mask = batch["tokens"], batch["attn_mask"]
    target_map = batch["target_map"] # [B, 3, L, L]
    device = tokens.device
    
    # 1. CFG Dropout
    if cfg_prob > 0:
        mask = torch.rand(tokens.shape[0], device=device) < cfg_prob
        tokens_in = tokens.clone()
        if mask.any():
            tokens_in[mask] = null_id 
    else:
        tokens_in = tokens

    x0 = torch.randn_like(target_map) 
    x1 = target_map 
    
    t = torch.rand(tokens.shape[0], device=device)
    t_broad = t.view(-1, 1, 1, 1)
    
    x_t = (1 - t_broad) * x0 + t_broad * x1 
    
    out = model(tokens_in, attn_mask, x_t=x_t, t=t)
    
    pair_mask = attn_mask[:, None, :] * attn_mask[:, :, None] # [B, L, L]
    loss_mask = pair_mask.unsqueeze(1)
    
    flow_loss = flow_matching_loss(out["v_pred"], x1, x0, loss_mask)
    
    optimizer.zero_grad()
    flow_loss.backward()
    optimizer.step()
    
    return flow_loss.item()

@torch.no_grad()
def sample_ensemble(model, tokens, attn_mask, null_id, num_samples=5, steps=20, guidance_scale=2.0):

    model.eval()
    device = tokens.device
    B, L = tokens.shape
    
    cond_tokens = tokens.repeat(num_samples, 1)
    mask = attn_mask.repeat(num_samples, 1)
    uncond_tokens = torch.full_like(cond_tokens, null_id)
    
    cond_feat = model(cond_tokens, mask)["pair"]
    if guidance_scale != 1.0:
        uncond_feat = model(uncond_tokens, mask)["pair"]
    
    x = torch.randn((num_samples, 3, L, L), device=device)
    dt = 1.0 / steps
    
    for s in range(steps):
        t = torch.full((num_samples,), s/steps, device=device)
        
        v_cond = model.flow_head(cond_feat, x, t)
        
        if guidance_scale == 1.0:
            v = v_cond
        else:
            v_uncond = model.flow_head(uncond_feat, x, t)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
        
        x = x + v * dt
        

    x = torch.clamp(x, 0.0, 1.0)

    bin_map = x[:, 0, :, :]
    frac_map = x[:, 1:, :, :]
    
    return bin_map, frac_map
