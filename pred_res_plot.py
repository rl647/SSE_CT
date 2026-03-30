#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, label
from scipy.spatial.distance import cdist
import os
import sys 

def get_pdb_residue_contact(pdb_path, contact_threshold=7):
    residues = []
    if not os.path.exists(pdb_path): return None
    
    try:
        current_res_id = None
        current_atoms = []
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    res_id = line[22:27]
                    if res_id != current_res_id:
                        if current_atoms:
                            residues.append(np.array(current_atoms))
                        current_res_id = res_id
                        current_atoms = []
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    current_atoms.append([x, y, z])
            if current_atoms:
                residues.append(np.array(current_atoms))
    except Exception as e:
        print(f"Error parsing PDB: {e}")
        return None
        
    if not residues: return None

    L = len(residues)
    contact_map = np.zeros((L, L))
    for i in range(L):
        for j in range(i, L):
            dists = cdist(residues[i], residues[j], metric='euclidean')
            if np.min(dists) < contact_threshold:
                contact_map[i, j] = 1.0
                contact_map[j, i] = 1.0 
    return contact_map

def process_gt_map(gt_map, min_seq_sep=6, gt_dilation=2, min_cluster_size=1):
    L = gt_map.shape[0]
    ones_matrix = np.ones((L, L), dtype=bool) 
    mask_diag = np.triu(ones_matrix, k=min_seq_sep) | np.tril(ones_matrix, k=-min_seq_sep)
    gt_clean = (gt_map * mask_diag) > 0.5
    
    gt_dilated = binary_dilation(gt_clean, iterations=gt_dilation)
    gt_upper = np.triu(gt_dilated, 1)
    
    labeled_upper, num_gt_clusters = label(gt_upper, structure=np.ones((3,3)))
    labeled_gt = labeled_upper + labeled_upper.T 
    
    gt_cluster_sizes = np.bincount(labeled_gt.ravel())
    for i in range(1, num_gt_clusters + 1):
        if gt_cluster_sizes[i] < min_cluster_size:
            labeled_gt[labeled_gt == i] = 0
            
    return labeled_gt

def plot_npy_vs_gt(ax, labeled_gt, pred_heatmap, title):
    masked_labeled_gt = np.ma.masked_where(labeled_gt == 0, labeled_gt)
    cmap_gt = plt.get_cmap('rainbow') 
    cmap_gt.set_bad('white')
    ax.imshow(masked_labeled_gt, cmap=cmap_gt, origin='upper', interpolation='nearest', alpha=0.6)

    pred_blobs = pred_heatmap > 0.01 
    
    if pred_blobs is not None:
        ax.contour(pred_blobs.astype(int), levels=[0.5], colors='black', 
                   linewidths=3.0, alpha=1.0, linestyles='dashed')
        
        ax.plot([], [], color='black', linestyle='dashed', linewidth=3.0, label='Pred Halo')
    
    ax.tick_params(axis='both', which='major', labelsize=20) 
    ax.set_title(title, fontsize=25, fontweight='bold')
    ax.legend(loc='upper right', framealpha=1.0, facecolor='white', prop={'size': 20})
    ax.set_facecolor('white')


if __name__ == '__main__':
    
    prot_name =  sys.argv[1]#'1ae6_H'
    npy_file = f'data/res_map/{prot_name}.npy'
    # npy_file = 'SSE_CT/data/res_map'
    pdb_file = f'.../{prot_name}.pdb'

    if not os.path.exists(npy_file):
        print(f"Error: Could not find {npy_file}")
    else:
        pred_heatmap = np.load(npy_file)
        L = pred_heatmap.shape[0]

        gt_map = get_pdb_residue_contact(pdb_file)
        

        if gt_map.shape[0] != L:

            min_L = min(L, gt_map.shape[0])
            pred_heatmap = pred_heatmap[:min_L, :min_L]
            gt_map = gt_map[:min_L, :min_L]
        
        labeled_gt = process_gt_map(gt_map, min_seq_sep=6, gt_dilation=2)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        plot_npy_vs_gt(ax, labeled_gt, pred_heatmap, title=f'{prot_name}')
        plt.tight_layout()
        
     
# %%
