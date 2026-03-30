# %%
import os
import torch
import subprocess
import numpy as np
import gc  # Garbage collector
from Bio import SeqIO
import sys

try:
    import esm
except ImportError:
    subprocess.check_call(["pip", "install", "fair-esm"])
    import esm

# --- LOW MEMORY CONFIGURATION ---
BATCH_SIZE = 16  
# --------------------------------

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# 2. Load Model
print("Loading model...")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()



if torch.cuda.is_available():
    model = model.half().to(device)
else:
    model = model.to(device)

model.eval()
batch_converter = alphabet.get_batch_converter()

def read_fasta(file_path):
    sequences = []

    for record in SeqIO.parse(file_path, "fasta"):

        sequences.append((record.id, str(record.seq)))
    return sequences

fasta_file_path = sys.argv[1]
path_embedded_esm2 = sys.argv[2]
json_out = sys.argv[3]
p6_out = sys.argv[4]
pss = sys.argv[5]
if not os.path.exists(path_embedded_esm2):
    os.makedirs(path_embedded_esm2, exist_ok=True)



raw_data = read_fasta(fasta_file_path)
raw_data.sort(key=lambda x: len(x[1]), reverse=True)

print(f"Processing {len(raw_data)} sequences...")


with torch.inference_mode():
    for i in range(0, len(raw_data), BATCH_SIZE):
        batch = raw_data[i : i + BATCH_SIZE]
        


        try:
            ids, strs, tokens = batch_converter(batch)
            tokens = tokens.to(device)

            results = model(tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]

            for j, (seq_id, seq_str) in enumerate(batch):
                seq_len = len(seq_str)

                embedded_seq = token_representations[j, 1 : seq_len + 1, :].float().cpu().numpy()
                np.save(os.path.join(path_embedded_esm2, f"{seq_id}.npy"), embedded_seq)

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"| WARNING: OOM on sequence {batch[0][0]} (Length: {len(batch[0][1])}). Skipping.")
                torch.cuda.empty_cache()
                continue
            else:
                raise e

        del tokens
        del results
        del token_representations
        if i % 1000 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            os.system(f'python new_test_ensemble.py {json_out} {p6_out} {path_embedded_esm2} {pss}')
            print(f"Processed {i}/{len(raw_data)}")
    print('starting p6')
    torch.cuda.empty_cache()
    gc.collect()
    os.system(f'python new_test_ensemble.py {json_out} {p6_out} {path_embedded_esm2} {pss}')

# %%
