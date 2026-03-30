#%%
import json
from Bio import SeqIO
import os
# Automatically use the dataset directory relative to the script location
# dataset_dir = os.path.join(os.path.dirname(__file__), 'data', 'dataset')
# fasta_file_path = os.path.join(dataset_dir, 'set3.fasta')  # File will be set3.fasta under 'dataset'
# json_output_path = os.path.join(dataset_dir, 'test_dataset.json')


def fasta_to_json(fasta_file_path, json_file_path):
    # Create a list to store the entries
    sequence_data = []
    
    # Read the FASTA file
    for record in SeqIO.parse(fasta_file_path, "fasta"):
        sequence_entry = {
            "id": record.id,             # Name/ID of the sequence
            "sequence": str(record.seq), # Sequence as a string
            "seq_len": len(record.seq)   # Length of the sequence
        }
        sequence_data.append(sequence_entry)
    
    # Save the data to a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(sequence_data, json_file, indent=4)
    
    print(f"FASTA data has been successfully converted to {json_file_path}.")



# %%
