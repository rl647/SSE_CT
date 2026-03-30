#%%
import numpy as np
from scipy.ndimage import gaussian_filter
import os
import sys


output_dir = sys.argv[1]
os.makedirs(output_dir, exist_ok=True)

# P6 Data Paths
p6_pred_dir = sys.argv[2]#'/home/runfeng/Dropbox/flow_topo/data/P6/predictions'
sse_range_file = sys.argv[3]#'/home/runfeng/Dropbox/flow_topo/data/P6/sse_range.txt'
sse_seq_file = sys.argv[4]#'/home/runfeng/Dropbox/flow_topo/data/P6/sse.txt'
test_fasta_file = sys.argv[5]#'/home/runfeng/Dropbox/flow_topo/data/test_data/test.fasta'

# Parameters
sigma_val = 1.5      
threshold_val = 0.5  


def get_seeds_only(sse_ranges, protein_len, p_contact, p_ct, threshold=0.5):

    seed_points = []
    num_sse = len(sse_ranges)
    
    for i in range(num_sse):
        for j in range(i + 1, num_sse):
            if p_contact[i, j] > threshold: 
                f_i, f_j = p_ct[0, i, j], p_ct[1, i, j]
                
                s_i, e_i = sse_ranges[i]
                s_j, e_j = sse_ranges[j]
                
                # Map relative fraction (0-1) to integer residue index
                c_i = int(s_i + (e_i - s_i - 1) * f_i)
                c_j = int(s_j + (e_j - s_j - 1) * f_j)
                
                # Boundary check
                if 0 <= c_i < protein_len and 0 <= c_j < protein_len:
                    seed_points.append((c_i, c_j))
                    # Symmetric seed
                    seed_points.append((c_j, c_i))
                    
    return seed_points

def generate_gaussian_map(L, seeds, sigma=1.5):

    pred_map = np.zeros((L, L), dtype=float)
    
    for (r, c) in seeds:
        pred_map[r, c] = 1.0
        

    heatmap = gaussian_filter(pred_map, sigma=sigma, mode='constant')
    
    heatmap = (heatmap + heatmap.T) / 2.0
    

    return heatmap


test_sse_range = {}
test_sse_seq = {}

if os.path.exists(sse_range_file):
    with open(sse_range_file) as f:
        for line in f:
            s = line.strip().split('\t')
            prot_id = s[0]
            ranges = []
            for r_str in s[1:]:
                parts = r_str.strip().split('-')
                if len(parts) == 2:
                    ranges.append(np.array([int(parts[0]), int(parts[1])]))
            test_sse_range[prot_id] = ranges

if os.path.exists(sse_seq_file):
    with open(sse_seq_file) as f:
        for line in f:
            s = line.strip().split('\t')
            if len(s) > 1:
                test_sse_seq[s[0]] = s[1]


prot_L_dict = {}
if os.path.exists(test_fasta_file):
    from Bio import SeqIO
    for record in SeqIO.parse(test_fasta_file, "fasta"):

        pid = record.id[:6] if len(record.id) >= 6 else record.id
        prot_L_dict[pid] = len(record.seq)



count = 0
if os.path.exists(test_fasta_file):

    with open(test_fasta_file) as f:
        for line in f:
            if not line.startswith('>'):
                continue
            
            prot_name = line[1:7] 
            
            if prot_name not in test_sse_range or prot_name not in test_sse_seq:
                continue
            
            pred_file = os.path.join(p6_pred_dir,f'{test_sse_seq[prot_name]}', f'prediction_consensus.txt')
            if not os.path.exists(pred_file):
                continue


            if prot_name in prot_L_dict:
                L = prot_L_dict[prot_name]
            else:
                ranges = test_sse_range[prot_name]
                if not ranges: continue
                L = max([r[1] for r in ranges]) + 5 
            
            sse_len = len(test_sse_seq[prot_name])
            p_cont_P6 = np.zeros((sse_len, sse_len))
            p_ct_P6 = np.zeros((2, sse_len, sse_len))
            
            try:
                with open(pred_file) as f_ana:
                    for l_ana in f_ana:
                        if l_ana.startswith('P_ID'): continue
                        s_ana = l_ana.strip().split('\t')
                        if len(s_ana) < 5: continue
                        
                        u, v = int(s_ana[0]), int(s_ana[1])
                        
                        p_cont_P6[u][v] = p_cont_P6[v][u] = float(s_ana[2])
                        
                        p_ct_P6[0][u][v] = float(s_ana[3]) # Fraction i
                        p_ct_P6[1][u][v] = float(s_ana[4]) # Fraction j
                        
                        p_ct_P6[0][v][u] = float(s_ana[4])
                        p_ct_P6[1][v][u] = float(s_ana[3])
            except Exception as e:
                print(f"Error reading {prot_name}: {e}")
                continue


            seeds = get_seeds_only(test_sse_range[prot_name], L, p_cont_P6, p_ct_P6, threshold=threshold_val)
            
            if not seeds:
                final_map = np.zeros((L, L))
            else:
                final_map = generate_gaussian_map(L, seeds, sigma=sigma_val)
            
            save_path = os.path.join(output_dir, f'{prot_name}.npy')
            np.save(save_path, final_map)
            
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} proteins...")

#%%