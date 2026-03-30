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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader

from sse_flow_model import *

RESULTS_DIR = ''
SUMMARY_FILE = ''
SUMMARY_protein = ''

STRENGTH_METRIC_FILE = ''
CHECKPOINT_PATH = '.../para_50.pth'


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


test_ids = set()

with open('.../test.fasta') as f:
    for line in f:
        if line.startswith('>'):
            s = line.strip().split('\t')
            test_ids.add(s[0][1:])


sse_groups = defaultdict(list)
id_to_seq = {}
contact_strength_lookup = {} 

with open('.../data/sse.txt') as f:
    for line in f:
        s = line.strip().split('\t')
        prot_id = s[0]
        if prot_id not in test_ids or len(s)==1:
            continue
        id_to_seq[prot_id] = s[1]
ks = list(test_ids)
for key in ks:
    if key not in id_to_seq:
        test_ids.remove(key)
# Load Contacts with Strength
with open('.../data/cir_topo.txt') as f:
    for line in f:
        s = line.strip().split('\t')
        prot_id = s[0]
        if prot_id not in test_ids or prot_id not in id_to_seq:
            continue
        
        u, v = int(s[1]), int(s[2])
        fi, fj = float(s[5]), float(s[6])
        
        strength = int(s[-1])

        # Sort u,v to ensure unique key
        min_uv, max_uv = min(u, v), max(u, v)
        contact_strength_lookup[(prot_id, min_uv, max_uv)] = strength
        
        ct_tuple = (u, v, fi, fj,strength) # Normal tuple for model
        
        sequence = id_to_seq[prot_id]
        
        found = False
        for entry in sse_groups[sequence]:
            if entry['id'] == prot_id:
                entry['contacts'].append(ct_tuple)
                found = True
                break
        
        if not found:
            sse_groups[sequence].append({'id': prot_id, 'contacts': [ct_tuple]})

all_sse_sequences = list(sse_groups.keys())



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

def calculate_metrics_pairwise(pred_bin, pred_flow, gt_bin, gt_flow):

    L = pred_bin.shape[0]
    triu_mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
    
    # Flatten based on Upper Triangle
    p_flat = pred_bin[triu_mask]
    g_flat = gt_bin[triu_mask]
    
    # Thresholding
    p_cont = (p_flat > CONTACT_THRESHOLD).float()
    g_cont = (g_flat > CONTACT_THRESHOLD).float()
    
    tp = (p_cont * g_cont).sum().item()
    fp = (p_cont * (1 - g_cont)).sum().item()
    fn = ((1 - p_cont) * g_cont).sum().item()
    
    epsilon = 1e-8
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    

    valid_contact_mask = (g_cont > 0.5)
    
    if valid_contact_mask.sum() > 0:

        p_flow_flat = pred_flow[:, triu_mask] 
        g_flow_flat = gt_flow[:, triu_mask]
        

        p_target = p_flow_flat[:, valid_contact_mask]
        g_target = g_flow_flat[:, valid_contact_mask]
        
        abs_diff = torch.abs(p_target - g_target)
        
        mae = abs_diff.mean().item()
    else:
        mae = 0.0
        
    return f1, mae, precision, recall

def analyze_contact_strength(pred_bin, gt_bin, prot_id,fracs,gt_ct):
  
    results = []
    L = pred_bin.shape[0]
    
    # Identify all Ground Truth Contacts
    # Iterate Upper Triangle
    for u in range(L):
        for v in range(u + 1, L):
            pred_prob = pred_bin[u, v].item()
            gt_probability = gt_bin[u, v]
            if gt_probability > CONTACT_THRESHOLD or pred_prob > CONTACT_THRESHOLD:
                # This is a true contact
                # Retrieve strength
                strength = contact_strength_lookup.get((prot_id, u, v), 0)
                f_u = gt_ct[0][u][v]
                f_v = gt_ct[1][u][v]
                p_f_u = fracs[0][u][v]
                p_f_v = fracs[1][u][v]


                # Check Prediction
                
                # is_recovered = 1 if pred_prob > CONTACT_THRESHOLD else 0
                if gt_probability > CONTACT_THRESHOLD and pred_prob > CONTACT_THRESHOLD:
                    results.append(f"{prot_id}\t{u}\t{v}\t{strength}\t{pred_prob:.4f}\tTP\t{f_u:.4f}\t{f_v:.4f}\t{p_f_u:.4f}\t{p_f_v:.4f}\n")
                elif gt_probability > CONTACT_THRESHOLD and pred_prob <= CONTACT_THRESHOLD:
                    results.append(f"{prot_id}\t{u}\t{v}\t{strength}\t{pred_prob:.4f}\tFN\t{f_u:.4f}\t{f_v:.4f}\t{p_f_u:.4f}\t{p_f_v:.4f}\n")
                elif gt_probability <= CONTACT_THRESHOLD and pred_prob > CONTACT_THRESHOLD:
                    results.append(f"{prot_id}\t{u}\t{v}\t{strength}\t{pred_prob:.4f}\tFP\t{f_u:.4f}\t{f_v:.4f}\t{p_f_u:.4f}\t{p_f_v:.4f}\n")

                
    return results


#%%
import time 
t1 = time.time()
summary_lines = []
summary_by_proteins = [] 
summary_lines.append("SSE_Sequence\tAvg_F1\tAvg_MAE\tAvg_Prece\tAvg_Recall\tNum_GTs\tNum_Preds\tProtein_IDs\n")
summary_by_proteins.append("P_IDs\tF1\tMAE\tPrece\tRecall\tSSE_Sequence\n")


for idx, sse_seq in enumerate(all_sse_sequences):
    
    folder_name = sse_seq[:]
    sse_dir = os.path.join(RESULTS_DIR, folder_name)
    os.makedirs(sse_dir, exist_ok=True)
    
    gt_list = sse_groups[sse_seq]
    gt_prot_ids = [g['id'] for g in gt_list]
    
    temp_examples = [TopologyExample(g['id'], sse_seq, g['contacts']) for g in gt_list]
    
    batch_gt = collate_batch(temp_examples, tokenizer)
    target_maps = batch_gt['target_map'].to(device) 
    
    for g_i in range(len(gt_list)):
        gt_path = os.path.join(sse_dir, f"ground_truth_{gt_list[g_i]['id']}.txt")
        gt_np = target_maps[g_i].cpu().numpy()
        save_sparse_to_txt(gt_path, gt_np[0], gt_np[1], gt_np[2], threshold=0.5)
    
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
    
    sum_weighted_fracs = np.sum(weighted_fracs, axis=0) # [2, L, L]
    sum_weights = np.sum(weights, axis=0) # [1, L, L]
    
    # Avoid division by zero
    eps = 1e-8
    avg_frac_map = sum_weighted_fracs / (sum_weights + eps)
    
    consensus_path = os.path.join(sse_dir, "prediction_consensus.txt")
    save_sparse_to_txt(consensus_path, avg_prob_map, avg_frac_map[0], avg_frac_map[1], threshold=CONTACT_THRESHOLD)
    
    for s_i in range(NUM_SAMPLES_TO_GENERATE):
        pred_path = os.path.join(sse_dir, f"prediction_sample_{s_i}.txt")
        save_sparse_to_txt(pred_path, bins_np[s_i], fracs_np[s_i, 0], fracs_np[s_i, 1], threshold=CONTACT_THRESHOLD)

    total_f1 = 0.0
    total_mae = 0.0
    total_prec = 0.0
    total_recall = 0.0
    pair_count = 0
    L = bins.shape[-1]
    
    pairwise_metrics = ['Sample_Idx\tGT_Protein_ID\tF1\tMAE\tPrecision\tSensitivity\n']
    
    for g_i in range(len(gt_list)):
        f1_max = [0,-1,0,0,0]
        strength_metrics_lines = []
        strength_metrics_lines.append("P_ID\tu\tv\tStrength\tP_Prob\tCM\tfu\tfv\tpfu\tpfv\n")

        for s_i in range(NUM_SAMPLES_TO_GENERATE):
            f1, mae, precision, recall = calculate_metrics_pairwise(
                bins[s_i], fracs[s_i], target_maps[g_i, 0], target_maps[g_i, 1:]
            )
            pairwise_metrics.append(f'Pred_{s_i}\t{gt_prot_ids[g_i]}\t{f1:.3f}\t{mae:.3f}\t{precision:.3f}\t{recall:.3f}\n')
            if f1>f1_max[0]:
                f1_max = [f1,s_i,mae, precision, recall]
        total_f1 += f1_max[0]
        total_mae += f1_max[2]
        total_prec += f1_max[3]
        total_recall += f1_max[4]
        pair_count += 1

        strength_rows = analyze_contact_strength(bins[f1_max[1]], target_maps[g_i, 0], gt_prot_ids[g_i],fracs[f1_max[1]],target_maps[g_i,1:])
        strength_metrics_lines.extend(strength_rows)
        summary_by_proteins.append(f'{gt_prot_ids[g_i]}\t{f1_max[0]:.4f}\t{f1_max[2]:.4f}\t{f1_max[3]:.4f}\t{f1_max[4]:.4f}\t{sse_seq}\n')
        with open(f'{sse_dir}/{gt_prot_ids[g_i]}_analysis.txt','w')as f1:
            f1.writelines(strength_metrics_lines)
    with open(f'{sse_dir}/pairwise_analysis.txt','w')as f1:
        f1.writelines(pairwise_metrics)
    avg_f1 = total_f1 / pair_count if pair_count > 0 else 0.0
    avg_mae = total_mae / pair_count if pair_count > 0 else 0.0
    avg_prece = total_prec / pair_count if pair_count > 0 else 0.0
    avg_recall = total_recall / pair_count if pair_count > 0 else 0.0
    ids_str = ",".join(gt_prot_ids)
    summary_lines.append(f"{sse_seq}\t{avg_f1:.4f}\t{avg_mae:.4f}\t{avg_prece:.4f}\t{avg_recall:.4f}\t{len(gt_list)}\t{NUM_SAMPLES_TO_GENERATE}\t{ids_str}\n")
    

    
    if (idx + 1) % 50 == 0:
        print(f"Processed {idx + 1}/{len(all_sse_sequences)} SSEs...")

with open(SUMMARY_FILE, 'w') as f:
    f.writelines(summary_lines)
with open(SUMMARY_protein, 'w') as f:
    f.writelines(summary_by_proteins)
    

#%%