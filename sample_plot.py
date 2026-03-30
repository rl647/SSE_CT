
#%%

import numpy as np
import matplotlib.pyplot as plt
import os
import sys 
from sse_extraction import ss_extraction
CONTACT_THRESHOLD=0.5
def plot_asymmetric_map(pred_bins, pred_fracs, gt_maps_list,results):

    L = pred_bins.shape[1]
    
    p_cont = pred_bins     
    p_fi = pred_fracs[0]   
    p_fj = pred_fracs[1]   

    p_cont_upper = np.triu(p_cont, k=1)
    pred_contact_viz = p_cont_upper + p_cont_upper.T
    
    asym_pred = np.zeros((L, L))
    pred_topo_mask = np.zeros((L, L)) 

    for r in range(L):
        for c in range(L):
            if r == c: continue
            prob = p_cont[min(r,c), max(r,c)]
            
            if prob > CONTACT_THRESHOLD:
                pred_topo_mask[r, c] = 1.0
                if r < c:
                    asym_pred[r, c] = p_fi[r, c]
                else:
                    asym_pred[r, c] = p_fj[c, r] # Lower triangle stores j
            else:
                pred_topo_mask[r, c] = 0.0

    pred_topo_viz = np.ma.masked_where(pred_topo_mask == 0, asym_pred)

    gt_raw = gt_maps_list
    gt_bin = gt_raw[0]
    gt_fi = gt_raw[1]
    gt_fj = gt_raw[2]
    
    gt_contact_viz = gt_bin
    
    asym_gt = np.zeros((L, L))
    gt_topo_mask = np.zeros((L, L))
    
    for r in range(L):
        for c in range(L):
            if r == c: continue
            # Check GT contact
            is_contact = gt_bin[min(r,c), max(r,c)] > 0.5
            
            if is_contact:
                gt_topo_mask[r, c] = 1.0
                if r < c:
                    asym_gt[r, c] = gt_fi[r, c]
                else:
                    asym_gt[r, c] = gt_fj[c, r]
            else:
                gt_topo_mask[r, c] = 0.0

    gt_topo_viz = np.ma.masked_where(gt_topo_mask == 0, asym_gt)

    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    cmap_contact = 'Greys'
    cmap_topo = plt.cm.jet 
    cmap_topo.set_bad(color='white') 
    
    TICK_FONTSIZE = 20
    im0 = axs[0, 0].imshow(pred_contact_viz, cmap=cmap_contact, vmin=0, vmax=1)
    axs[0, 0].set_title(f"Pred Contact F1 score: {results[0]}", fontsize=25)
    axs[0, 0].tick_params(axis='both', labelsize=TICK_FONTSIZE) # X and Y axis
    cb0 = plt.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)
    cb0.ax.tick_params(labelsize=TICK_FONTSIZE) # Colorbar axis
    
    im1 = axs[0, 1].imshow(pred_topo_viz, cmap=cmap_topo, vmin=0, vmax=1)
    axs[0, 1].set_title(f"Pred Topology (Asym, MAE: {results[1]})\nUpper: $f_i$ | Lower: $f_j$", fontsize=25)
    axs[0, 1].tick_params(axis='both', labelsize=TICK_FONTSIZE)
    cb1 = plt.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)
    cb1.ax.tick_params(labelsize=TICK_FONTSIZE)
    
    im2 = axs[1, 0].imshow(gt_contact_viz, cmap=cmap_contact, vmin=0, vmax=1)
    axs[1, 0].set_title(f"GT Contact", fontsize=25)
    axs[1, 0].tick_params(axis='both', labelsize=TICK_FONTSIZE)
    cb2 = plt.colorbar(im2, ax=axs[1, 0], fraction=0.046, pad=0.04)
    cb2.ax.tick_params(labelsize=TICK_FONTSIZE)
    
    im3 = axs[1, 1].imshow(gt_topo_viz, cmap=cmap_topo, vmin=0, vmax=1)
    axs[1, 1].set_title(f"GT Topology (Asym)\nUpper: $f_i$ | Lower: $f_j$", fontsize=25)
    axs[1, 1].tick_params(axis='both', labelsize=TICK_FONTSIZE)
    cb3 = plt.colorbar(im3, ax=axs[1, 1], fraction=0.046, pad=0.04)
    cb3.ax.tick_params(labelsize=TICK_FONTSIZE)
    
    plt.tight_layout()


#%%
results = {}
SUMMARY_protein = ''
with open(SUMMARY_protein)as f:
    for line in f:
        s=line.strip().split('\t')
        results[s[0]] = [s[1],s[2]]
# sse = 'DAGbFGCbEEbGADDJCAAIbEbE'
prot_id = '2cdu_B'
_,sse = ss_extraction(prot_id)
L = len(sse)
path = '' #RESULTS_DIR from testing
p_contact = np.zeros((L,L))
p_ct = np.zeros((2,L,L))
g_ct = np.zeros((3,L,L))

with open(f'{path}/{sse}/{prot_id}_analysis.txt')as f:
    for line in f:
        if line.startswith('P_ID'):
            continue 
        s=line.strip().split('\t')
        u,v = int(s[1]),int(s[2])
        p_contact[u][v]=float(s[4])
        g_ct[0][u][v]=1 if int(s[3])>0 else 0 
        g_ct[0][v][u]=1 if int(s[3])>0 else 0 

        p_ct[0][u][v]=float(s[8])
        p_ct[0][v][u]=float(s[8])
        p_ct[1][u][v]=float(s[9])
        p_ct[1][v][u]=float(s[9])


        g_ct[1][u][v]=float(s[6])
        g_ct[1][v][u]=float(s[6])
        g_ct[2][u][v]=float(s[7])
        g_ct[2][v][u]=float(s[7])
plot_asymmetric_map(p_contact,p_ct,g_ct,results[prot_id])
# %%
