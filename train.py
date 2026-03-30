#%%
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys 
from sse_flow_model import *
from collections import defaultdict
import numpy as np
import string
from torch.utils.data import DataLoader


vocab = string.ascii_letters
print(vocab)

#%%

tokenizer = SSETokenizer(vocab=list(vocab))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FlowMatchingModel(len(tokenizer.itos), tokenizer.pad_id).to(device)
checkpoint_file = '' #path to save the checkpoint

if torch.cuda.is_available():
    checkpoint = torch.load(checkpoint_file)
else:
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))

if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


#%%

# load training set
train = set()
training_data = ''
with open(training_data) as f:
    for line in f:
        if line.startswith('>'):
            s = line.strip().split('\t')
            train.add(s[0][1:])
# load ct
ct_path = ''
sse = defaultdict(list)
with open(ct_path) as f:
    for line in f:
        s = line.strip().split('\t')
        if s[0] not in train: continue
        ct = np.array(s[1:], dtype=float)
        ct = tuple([int(ct[0]), int(ct[1]), ct[4], ct[5],int(ct[8])])
        sse[s[0]].append(ct)

# read sse
sse_path = ''
with open(sse_path) as f:
    for line in f:
        s = line.strip().split('\t')
        if s[0] not in sse:
            continue
        sse[s[0]].append(s[1])



ks = list(sse.keys())
for k in ks:
    if k not in train:
        del sse[k]

print(f"Training Samples: {len(sse)}")

dataset = []
for key, val in sse.items():
    dataset.append(TopologyExample(key, val[-1], val[:-1]))

train_loader = DataLoader(
    dataset, 
    batch_size=32, 
    shuffle=True, 
    collate_fn=lambda x: collate_batch(x, tokenizer)
)

#%%
epochs = 50
checkpoint_path = ''
for epoch in range(1, epochs + 1):
    total_loss = 0
    batches = 0
    
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        loss = training_step(model, batch, optimizer, tokenizer.null_id, cfg_prob=0.2)
        
        total_loss += loss
        batches += 1
    
    avg_loss = total_loss / batches
    
    if epoch % 1 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d}/{epochs}: Loss={avg_loss:.4f}")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }
        torch.save(checkpoint, f'{checkpoint_path}/para_{epoch}.pth')
        print(f"  >> Checkpoint saved: {checkpoint_path}")
# %%
