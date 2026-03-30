
#%%
import os 
import numpy as np 
import math 
import shutil
import sys 
from scipy.spatial.distance import cdist 
import gc
from sse_extraction import ss_extraction 
#%%

def calculate_circuit_topology(pdb_dict, sse_ranges_dict, contact_threshold=7, min_contacts=10):

    topology_data = {}

    for prot_id in pdb_dict.keys():
        if prot_id not in sse_ranges_dict or not pdb_dict[prot_id]:
            continue
            
        ranges = sse_ranges_dict[prot_id]
        res_keys = list(pdb_dict[prot_id].keys())
        total_res = len(res_keys)
        
        prot_topology = []
        
        for i in range(len(ranges)):
            for j in range(i + 1, len(ranges)):
                # if not (i==0 and j==2):
                #     continue
                start_i, end_i = ranges[i] 

                if start_i >= total_res or end_i > total_res: continue
                
                atoms_i = []
                atom_res_idx_i = [] 
                
                sse_i_indices = range(start_i, end_i) 
                for local_idx, r_idx in enumerate(sse_i_indices):
                    res_id = res_keys[r_idx]
                    coords = pdb_dict[prot_id][res_id] # Shape: (N_atoms, 3)
                    atoms_i.append(coords)
                    atom_res_idx_i.extend([local_idx] * len(coords))
                
                if not atoms_i: continue
                atoms_i = np.vstack(atoms_i)
                atom_res_idx_i = np.array(atom_res_idx_i)

                start_j, end_j = ranges[j]
                if start_j >= total_res or end_j > total_res: continue

                atoms_j = []
                atom_res_idx_j = []
                
                sse_j_indices = range(start_j, end_j)
                for local_idx, r_idx in enumerate(sse_j_indices):
                    
                    res_id = res_keys[r_idx]
                    coords = pdb_dict[prot_id][res_id]
                    atoms_j.append(coords)
                    atom_res_idx_j.extend([local_idx] * len(coords))
                
                if not atoms_j: continue
                atoms_j = np.vstack(atoms_j)
                atom_res_idx_j = np.array(atom_res_idx_j)



                dists = cdist(atoms_i, atoms_j, metric='euclidean')
                # print(i,len(atoms_i),atoms_i)
                # print(j,len(atoms_j),atoms_j,dists,'\n','\n','\n')

                contact_mask = dists < contact_threshold
                num_contacts = np.sum(contact_mask)
                # print(num_contacts)
                if num_contacts >= 1:

                    rows, cols = np.where(contact_mask)
                    

                    contact_res_i = atom_res_idx_i[rows]
                    contact_res_j = atom_res_idx_j[cols]
                    
                    avg_pos_i = np.mean(contact_res_i)
                    avg_pos_j = np.mean(contact_res_j)
                    
                    len_i = end_i - start_i
                    len_j = end_j - start_j
                    
                    frac_i = avg_pos_i / max(1, len_i - 1)
                    frac_j = avg_pos_j / max(1, len_j - 1)
                    
                    prot_topology.append([i, j,round(avg_pos_i),round(avg_pos_j), round(frac_i, 3), round(frac_j, 3),len(atoms_i),len(atoms_j),num_contacts])
        
        topology_data[prot_id] = prot_topology
    return topology_data

#%%
import time
ct_file = '' # file that store contact of sses
sse_path = '' # file that store sse of proteins
path = '' # path to pdb file

sse = {}
sse_range = {}

pdb = {}
c=0
with open(sse_path) as f:
    for line in f:
        s=line.strip().split('\t')
        # print(s[0])
        if len(s)>1 and len(s[1])>=5:
            pdb[s[0]] = {}
            c+=1
        # if len(pdb)>=1:
        #     break
        if len(pdb)>=1000:  
            print(f'Processing 1000 samples out of {c}')
            sse = {}
            sse_range = {}
            for key in pdb.keys():
                if not os.path.exists(f'{path}/{key}.pdb'):
                    continue 
                sse_range[key],sse[key] = ss_extraction(key)
                with open(f'{path}/{key}.pdb') as f:
                    res = None 
                    for line in f:
                        # s=line.strip().split('\t')
                        if not line.startswith('ATOM'):
                            continue
                        if line[22:27].replace(' ','') not in pdb[key]:
                            if pdb[key] and res is not None: 
                                pdb[key][res] = np.array(pdb[key][res])
                            res = line[22:27].replace(' ','')
                            pdb[key][res] = []
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        pdb[key][res].append(np.array([x,y,z],dtype=float))
                    
                    if res is not None:
                        pdb[key][res] = np.array(pdb[key][res])


            circuit_topology_results = calculate_circuit_topology(pdb, sse_range)

            with open(ct_file,'a')as f:
                for key, val in circuit_topology_results.items():
                    
                    for i, e in enumerate(val):
                        f.write(key+'\t')
                        for i2,e2 in enumerate(e):
                            f.write(str(e2)+'\t')
                        f.write('\n')
            pdb = {}
            del circuit_topology_results
            gc.collect()
            # time.sleep(10)
# %%
