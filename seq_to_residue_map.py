#%%
import json
from Bio import SeqIO
import os
import sys
# sys.path.append('/home/runfeng/Dropbox/flow_topo/sse_topo/code')
from fasta_to_json import fasta_to_json
import time
#%%
fasta_file_path = 'data/validation.fasta'
json_out = 'data/validation.json'
fasta_to_json(fasta_file_path,json_out)
path_embedded_esm2 = "data/esm2"
P6_out = 'data/P6_out'
predicted_ss = 'data/predicted_ss.txt'

os.system(f'python esm_embedding.py {fasta_file_path} {path_embedded_esm2} {json_out} {P6_out} {predicted_ss}')

#%%
os.system(f'python sse_conversion.py {predicted_ss} data/P6_sse')

#%%
sse_path = 'data/P6_sse/sse.txt'
results_dir = 'data/sse_flow'
os.system(f'python ct_prediction.py {sse_path} {results_dir}')



# %%

output_dir = 'data/res_map'
# P6 Data Paths
p6_pred_dir = results_dir#'/home/runfeng/Dropbox/flow_topo/data/P6/predictions'
sse_range_file = 'data/P6_sse/sse_range.txt' #'/home/runfeng/Dropbox/flow_topo/data/P6/sse_range.txt'
sse_seq_file = 'data/P6_sse/sse.txt' #'/home/runfeng/Dropbox/flow_topo/data/P6/sse.txt'
test_fasta_file = fasta_file_path #'/home/runfeng/Dropbox/flow_topo/data/test_data/test.fasta'

# Parameters
os.system(f'python convert_residue_map.py {output_dir} {p6_pred_dir} {sse_range_file} {sse_seq_file} {test_fasta_file}')

# %%
