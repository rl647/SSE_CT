#!/bin/bash


# Step 1: Convert FASTA to JSON
echo "Running fasta2json.py..."
/home/runfeng/miniconda3/envs/bin/python /home/runfeng/Dropbox/flow_topo/porter6/fasta2json.py
if [ $? -ne 0 ]; then
    echo "Error running fasta2json.py. Exiting."
    exit 1
fi

# Step 2: Generate embeddings with ESM
echo "Running emb_esm3_fasta.py..."
/home/runfeng/miniconda3/envs/bin/python /home/runfeng/Dropbox/flow_topo/porter6/emb_esm3_fasta.py
if [ $? -ne 0 ]; then
    echo "Error running emb_esm3_fasta.py. Exiting."
    exit 1
fi


echo "Running new_test_ensemble.py in predictor3..."
/home/runfeng/miniconda3/envs/bin/python /home/runfeng/Dropbox/flow_topo/porter6/predictor3/new_test_ensemble.py
if [ $? -ne 0 ]; then
    echo "Error running new_test_ensemble.py. Exiting."
    exit 1
fi

echo "Running json2csv3.py in predictor3..."
/home/runfeng/miniconda3/envs/bin/python /home/runfeng/Dropbox/flow_topo/porter6/predictor3/json2csv3.py
if [ $? -ne 0 ]; then
    echo "Error running json2csv3.py. Exiting."
    exit 1
fi

echo "All steps completed successfully!"
