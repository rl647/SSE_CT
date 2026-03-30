import os
import params.rnn_hyperparams as rnn_feature

# esm&msa_transformer: https://github.com/facebookresearch/esm
featureType_list = ['onehot', 'protTrans', 'esm2']
featureType = featureType_list[2]
# netName = 'model.cnn_L11'
netName_list = ['cnn_L11', 'ffnn', 'rnn_bi', 'lstm_bi']
netName = f'model.{netName_list[0]}'
model_name = f'{netName[6:]}'

# cnn: netName=['cnn_L11', 'ffnn']
# rnn: netName=['rnn_bi', 'lstm_bi']
NNtype_list = ['cnn', 'rnn']
if netName in ['model.cnn_L11', 'model.ffnn']:
    net_type = NNtype_list[0] # cnn
else:
    net_type = NNtype_list[1] # rnn

# Rnn model, True; otherwise, False.
# transpose = False

# mdoel paramethers
padding = False

lr = 0.00005
train_epochs = 50
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
