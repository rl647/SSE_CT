# SSE-CT: Millisecond Prediction of Protein Contact Maps

This repository contains the files for predicting Secondary Structure Element (SSE) contact maps and circuit topology using a generative match model. 

This work highlights that coarse-grained secondary structure preserves strong constraints on protein folding. It demonstrates exceptionally strong performance on long-range and beta-sheet interactions, which have traditionally been the bottleneck for residue-based contact predictions. Furthermore, it is extremely fast, enabling rapid explorations for protein-level genotype-phenotype (GP) studies.

## Key Features
* **Ultra-Fast:** Completes single predictions in milliseconds on average.
* **Robust:** Achieves similar accuracy when using either experimental or predicted SSEs.
* **Targeted:** Highly effective for folding core predictions.
* **Dynamic Insight:** Able to distinguish rigid bodies from flexible regions based on entropy.

---

## Set Up the Environment

### 1. Clone the Repository
First, clone this repository to your local machine and enter the directory:
```bash
git clone [https://github.com/rl647/SSE_CT.git](https://github.com/rl647/SSE_CT.git)
cd SSE_CT
```

### 2. Installation
Create and activate a new Conda environment with Python 3.12, then install the required dependencies:

```bash

# Create and activate the environment
conda create -n sse_env python=3.12
conda activate sse_env

# Install Python dependencies
pip install -r requirements.txt

# Install DSSP
conda install -c bioconda dssp=4.0.4

```
---
## Usage Workflow
The pipeline consists of several stages, moving from data preparation to final analysis. Note: Please ensure you have updated the data paths inside the scripts before running them.

### Stage 1: Data Preparation
Sample SSE extraction can be done by running the extraction script. Requirement: You must uncompress data/ssnw.tar.xz before running this.

```bash

python3 data_preparation/sse_extraction.py
(For extraction directly from coordinate files, please refer to the script in our SSE_search repository.)
```

### Stage 2: Circuit Topology Extraction
Extract the circuit topology using the prepared data:

```bash
python3 data_preparation/topo_extraction.py
Stage 3: Coarse-Grained Residue Prediction
Generate the coarse-grained, residue-level predictions:
```

```bash

python3 seq_to_residue_map.py
```
### Stage 4: Customization & Fine-Tuning
For testing and further fine-tuning on your specific datasets, you can customize and run:

train.py

test.py
---

## Citation
If you use SSE-CT in your research, please cite our work:

```bibtex
@article{Lin2026,
  title = {Millisecond Prediction of Protein Contact Maps from Amino Acid Sequences},
  url = {[http://dx.doi.org/10.64898/2026.03.15.711852](http://dx.doi.org/10.64898/2026.03.15.711852)},
  DOI = {10.64898/2026.03.15.711852},
  publisher = {openRxiv},
  author = {Lin, Runfeng and Ahnert, Sebastian E},
  year = {2026},
  month = mar 
}
```
