import os
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import label_binarize

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from torch.utils.data import DataLoader
from dataset.ss_dataset import Sequence
import params.filePath as paramF
import params.hyperparams as paramH

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def trim_padding_and_flat(sequences: List[Sequence], pred):
    all_target = np.array([])
    all_trimmed_pred = np.array([])
    for i, seq in enumerate(sequences):
        # tmp_pred = pred[i][:len(seq)].cpu().detach().numpy()
        all_target = np.concatenate([all_target, seq.clean_target])
        # all_trimmed_pred = np.concatenate([all_trimmed_pred, tmp_pred])
    all_trimmed_pred = pred.cpu().detach().numpy()
    return all_target, all_trimmed_pred

def concat_target_and_output(sequences: List[Sequence], pred):
    all_target = np.array([])
    all_pred = np.array([])
    for i, seq in enumerate(sequences):
        all_target = np.concatenate([all_target, seq.clean_target])
    pred = pred.squeeze(0)
    all_pred = pred.cpu().detach().numpy()
    return all_target, all_pred

def get_targetPred(sequences: List[Sequence], pred):    
    if paramH.padding:
        target, pred = trim_padding_and_flat(sequences, pred)
    else:
        target, pred = concat_target_and_output(sequences, pred)
    return target, pred


def batch_auc(target, pred):
    '''
    !!!! Cannot calculate AUC score since some sequence does not include all 9 classes.
    
    Given target&pred, calculate AUC score.
    params:
        target - np.array, ground truth
        pred - np.array, predicted valued by a predictor.

    return:
        auc - float, auc score.
    '''
    # Classes that are expected to be in the data
    expected_classes = [0, 1, 2, 3, 4, 5, 6, 7]

    # Determine which expected classes are not in y_true
    missing_classes = list(set(expected_classes) - set(target))
    print("Missing classes:", missing_classes)

    # Binarize the true labels for one-vs-rest calculation
    y_true_binarized = label_binarize(target, classes=expected_classes)

    # Adjust the predicted probabilities if any class is missing
    if missing_classes:
        # Remove columns corresponding to missing classes
        y_pred_proba_adjusted = np.delete(pred, missing_classes, axis=1)
        y_true_binarized = np.delete(y_true_binarized, missing_classes, axis=1)
    else:
        y_pred_proba_adjusted = pred

    auc = metrics.roc_auc_score(y_true_binarized, y_pred_proba_adjusted, multi_class='ovr')
    # auc = metrics.auc(fpr, tpr)
    return auc

def batch_acc(target, pred):
    '''
    Given target&pred, calculate accuracy score.
    params:
        target - np.array, ground truth
        pred - np.array, predicted valued by a predictor.

    return:
        acc - float, accuracy score.
    '''
    # Get the unique classes present in y_true
    unique_classes = np.unique(target)
    unique_classes = [int(u) for u in unique_classes]
    acc = metrics.top_k_accuracy_score(target, pred[:, unique_classes], k=1)
    #pred_labels = np.argmax(pred, axis=1)
    #acc = accuracy_score(target, pred_labels)
    return acc

def get_batch_PreTargetList(pred, target, lens):
    '''
    if batch_size>1, ignore the padding regions and get the actural pred and target lists.

    params:
        pred - list, list of padded prediction values.
        target - list, list of padded target values.
        lens - list, true lengths of the sequences.
    return:
        pre_list - list, all predictions for multiple sequences
        target_list - list, all true values for multiple sequences
    '''
    pre_list = []
    target_list = []
    
    for p, t, l in zip(pred, target, lens):
        pre_list += p[:l].tolist()
        target_list += t[:l].tolist()
        
    return pre_list, target_list

# To get the loss we cut the output and target to the length of the sequence, removing the padding.
# This helps the network to focus on the actual sequence and not the padding.
def get_loss(sequences, output, criterion) -> torch.Tensor:
    loss = 0.0
    # Cycle through the sequences and accumulate the loss, removing the padding
    for i, seq in enumerate(sequences):
        # seq_loss = criterion(output[i][:len(seq)], torch.tensor(seq.clean_target, device=device, dtype=torch.float))
        target = torch.tensor(seq.clean_target, device=device, dtype=torch.uint8)
        seq_loss = criterion(output[i], target)
        loss += seq_loss
    # Return the average loss over the sequences of the batch
    return loss / len(sequences)

# save and load model
def save_checkpoint(net, optimizer, Loss, EPOCH, PATH):
    torch.save({
                'epoch': EPOCH,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': Loss,
                }, PATH)
    
def load_checkpoint(net, optimizer, PATH):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(PATH):
        print("=> loading checkpoint '{}'".format(PATH))
        checkpoint = torch.load(PATH)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        losslogger = checkpoint['loss']
        
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(losslogger, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(PATH))

    return net, optimizer, start_epoch, losslogger

def load_model(net, optimizer, PATH):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(PATH):
        print("=> loading checkpoint '{}'".format(PATH))
        checkpoint = torch.load(PATH)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print("=> (epoch {})"
                  .format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(PATH))

    return net, optimizer, start_epoch

def count_modelParams(net):
    '''
    Given a mdoel, count the parameters inside this model.
    params:
        net - nn. Module

    return:
        int, number of params.
    '''
    return sum(p.numel() for p in net.parameters())
