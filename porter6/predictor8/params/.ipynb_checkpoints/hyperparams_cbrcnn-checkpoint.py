import os
import params.rnn_hyperparams as rnn_feature

# esm&msa_transformer: https://github.com/facebookresearch/esm
featureType_list = ['onehot', 'protTrans', 'esm2']
featureType = featureType_list[2]
net_type = 'cbrcnn' # cbrcnn

netName_stage1 = f'model.cbrcnn_stage1'
netName_stage2 = f'model.cbrcnn_stage2'
model_name_stage1 = f'{netName_stage1[6:]}'
model_name_stage2 = f'{netName_stage2[6:]}'
model_name = 'cbrcnn'

# mdoel paramethers
padding = False

lr = 0.00005
train_epochs = 2
batch_size = 1
dropout = 0.0

if featureType=='protTrans':
    n_features = 1024
elif featureType=='esm2':
    n_features = 1280
elif featureType=='onehot':
    n_features = 21

# filtering sequence length
MAX_seq_length = 400000

# load model
# if True, overwriting model_pth after training.
load_model = False
k_folds = 5