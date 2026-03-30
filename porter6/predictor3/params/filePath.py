import os
from params.hyperparams import *
#from params.hyperparams_cbrcnn import *


# idr/
# ROOT = os.path.realpath('..')
ROOT = os.path.realpath('..')

###
# Data
###
path_data = os.path.join(ROOT, 'data')

# 1. dataset
path_dataset = os.path.join(path_data, 'dataset')
path_dataset_train = os.path.join(path_dataset, 'train_dataset.json')
path_dataset_test = os.path.join('/home/runfeng/Dropbox/flow_topo/data/test_data', 'test.json')

# 2. features
path_features = '/home/runfeng/Dropbox/flow_topo/data/P6'
path_embedded_eval = os.path.join(path_features, 'evaluation')
path_embedded_protTrans = os.path.join(path_features, 'protTrans')
path_embedded_onehot = os.path.join(path_features, 'onehot')
# path_embedded_hmm = os.path.join(path_features, 'hmm')
path_embedded_esm2 = '/home/runfeng/Dropbox/flow_topo/data/P6'
# 3. model
path_predictor = os.path.join(ROOT, 'predictor3')
path_output = '/home/runfeng/Dropbox/flow_topo/data/P6/output'

# plots_dir = os.path.join(path_output, f'plots/{featureType}/{model_name}')
# if not os.path.isdir(plots_dir):
#     os.mkdir(plots_dir)
    
model_pth = os.path.join(path_output, f'trained_model/{featureType}/{model_name}.pth')
auc_loss_pth = os.path.join(path_output, f'auc_loss/{featureType}/{model_name}.csv')


# log
path_log = os.path.join(path_output, f'log/{featureType}')

# 4. uniprot
path_uniprot = os.path.join(path_data, 'uniprot')

# 5. predictions
path_pred = os.path.join(path_output, 'pred')  
path_pred_plot = os.path.join(path_pred, 'plots')
path_pred_files = os.path.join(path_pred, 'files')