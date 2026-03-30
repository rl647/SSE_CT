# %%
import os
import torch
import subprocess
import numpy as np
import gc  # Garbage collector
from Bio import SeqIO

try:
    import esm
except ImportError:
    subprocess.check_call(["pip", "install", "fair-esm"])
    import esm

# --- LOW MEMORY CONFIGURATION ---
BATCH_SIZE = 16  # Keep this at 1 for 8GB VRAM
# --------------------------------
finish = set()
with open('/home/runfeng/Dropbox/flow_topo/data/P6/P6_predicted_ss.txt') as f:
    for line in f:
        s=line.strip().split('\t')
        finish.add(s[0])
# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# 2. Load Model
print("Loading model...")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

# OPTIMIZATION: Convert model to Half Precision (FP16)
# This drastically reduces memory usage from ~2.6GB to ~1.3GB
if torch.cuda.is_available():
    model = model.half().to(device)
    print("Model converted to FP16 for memory efficiency.")
else:
    model = model.to(device)

model.eval()
batch_converter = alphabet.get_batch_converter()

def read_fasta(file_path):
    sequences = []
    a=0
    for record in SeqIO.parse(file_path, "fasta"):
        # Truncate very long sequences if necessary (optional safeguard)
        # if len(record.seq) > 4000: continue 
        # if os.path.exists(f'/home/runfeng/Dropbox/flow_topo/porter6/predictor3/output/trained_model/esm2/{record.id}.npy'):
        #     print(record.id)
        #     continue

        if record.id in finish:
            # print(record.id)
            a+=1
            continue
        
        sequences.append((record.id, str(record.seq)))
    print(a)
    return sequences

fasta_file_path = '/home/runfeng/Dropbox/flow_topo/data/test_data/test.fasta'
path_embedded_esm2 = "/home/runfeng/Dropbox/flow_topo/data/P6/esm2"

if not os.path.exists(path_embedded_esm2):
    os.makedirs(path_embedded_esm2, exist_ok=True)

# 3. Read and Sort
# Sorting is still useful even with Batch Size 1 to predict time remaining more accurately
raw_data = read_fasta(fasta_file_path)
raw_data.sort(key=lambda x: len(x[1]), reverse=True)

print(f"Processing {len(raw_data)} sequences...")

# 4. Processing Loop
# inference_mode is slightly more efficient than no_grad
with torch.inference_mode():
    for i in range(0, len(raw_data), BATCH_SIZE):
        batch = raw_data[i : i + BATCH_SIZE]
        
        # Check if file already exists to skip (resume capability)
        # seq_id = batch[0][0]
        # if os.path.exists(os.path.join(path_embedded_esm2, f"{seq_id}.npy")):
        #     continue

        try:
            ids, strs, tokens = batch_converter(batch)
            tokens = tokens.to(device)

            # Run model
            # return_contacts=False is critical to save memory
            results = model(tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]

            # Save results
            for j, (seq_id, seq_str) in enumerate(batch):
                seq_len = len(seq_str)
                # Slice and move to CPU
                # We cast back to float32 (numpy default) for saving compatibility
                embedded_seq = token_representations[j, 1 : seq_len + 1, :].float().cpu().numpy()
                np.save(os.path.join(path_embedded_esm2, f"{seq_id}.npy"), embedded_seq)

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"| WARNING: OOM on sequence {batch[0][0]} (Length: {len(batch[0][1])}). Skipping.")
                torch.cuda.empty_cache()
                continue
            else:
                raise e

        # Aggressive cleanup
        del tokens
        del results
        del token_representations
        if i % 1000 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            os.system(f'python /home/runfeng/Dropbox/flow_topo/porter6/predictor3/new_test_ensemble.py')
            print(f"Processed {i}/{len(raw_data)}")

print('Done!!!!')
# %%
