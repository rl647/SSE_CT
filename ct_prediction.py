#%%
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
import string
import random
import time

from sse_flow_model import *


sse_path = sys.argv[1]   # File containing: Protein_ID \t SSE_Sequence
RESULTS_DIR = sys.argv[2] # Output folder

CHECKPOINT_PATH = 'data/parameters/para_50.pth'

NUM_SAMPLES_TO_GENERATE = 20
GUIDANCE_SCALE = 2.0
STEPS = 50
CONTACT_THRESHOLD = 0.5      

os.makedirs(RESULTS_DIR, exist_ok=True)

#%%

vocab = string.ascii_letters
tokenizer = SSETokenizer(vocab=list(vocab))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FlowMatchingModel(len(tokenizer.itos), tokenizer.pad_id).to(device)

if torch.cuda.is_available():
    checkpoint = torch.load(CHECKPOINT_PATH)
else:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))

if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()

#%%

unique_seqs = defaultdict(list)

with open(f'{sse_path}') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            prot_id = parts[0]
            seq = parts[1]
            unique_seqs[seq].append(prot_id)

all_sse_sequences = list(unique_seqs.keys())

#%%

def save_sparse_to_txt(fname, contact_map, frac_map_i, frac_map_j, threshold=0.5):

    L = contact_map.shape[0]
    lines = []
    for u in range(L):
        for v in range(u + 1, L):
            prob = contact_map[u, v]
            if prob > threshold:
                fi = frac_map_i[u, v]
                fj = frac_map_j[u, v]
                lines.append(f"{u}\t{v}\t{prob:.4f}\t{fi:.4f}\t{fj:.4f}\n")
    
    with open(fname, 'w') as f:
        f.writelines(lines)

#%%
t1 = time.time()
print(f"\n--- Starting Inference on {len(all_sse_sequences)} Sequences ---")

for idx, sse_seq in enumerate(all_sse_sequences):
    
    folder_name = sse_seq[:]
    sse_dir = os.path.join(RESULTS_DIR, folder_name)
    os.makedirs(sse_dir, exist_ok=True)
    
    associated_ids = unique_seqs[sse_seq]
    with open(os.path.join(sse_dir, "associated_ids.txt"), "w") as f_ids:
        f_ids.write("\n".join(associated_ids))

    tokens = tokenizer.encode(sse_seq).unsqueeze(0).to(device)
    mask = torch.ones_like(tokens).float().to(device)
    
    bins, fracs = sample_ensemble(
        model, tokens, mask, tokenizer.null_id, 
        num_samples=NUM_SAMPLES_TO_GENERATE, 
        guidance_scale=GUIDANCE_SCALE, 
        steps=STEPS
    )
    
    bins_np = bins.cpu().numpy()
    fracs_np = fracs.cpu().numpy()
    
    avg_prob_map = np.mean(bins_np, axis=0)
    
    weights = bins_np[:, np.newaxis, :, :] # Expand to [N, 1, L, L]
    weighted_fracs = fracs_np * weights
    
    sum_weighted_fracs = np.sum(weighted_fracs, axis=0)
    sum_weights = np.sum(weights, axis=0)
    
    eps = 1e-8
    avg_frac_map = sum_weighted_fracs / (sum_weights + eps)
    
    consensus_path = os.path.join(sse_dir, "prediction_consensus.txt")
    save_sparse_to_txt(consensus_path, avg_prob_map, avg_frac_map[0], avg_frac_map[1], threshold=CONTACT_THRESHOLD)
    

    if (idx + 1) % 50 == 0:
        print(f"Processed {idx + 1}/{len(all_sse_sequences)}...")


#%%