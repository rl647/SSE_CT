from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from dataset.ss_dataset import Sequence
from model.utils import trim_padding_and_flat, concat_target_and_output, batch_auc, get_loss, get_targetPred, batch_acc
import params.filePath as paramF
import params.hyperparams as paramH

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    losses = np.array([])
    
    for batch_idx, (sequences, data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        
        loss = get_loss(sequences, output, criterion)
        loss.backward()
        optimizer.step()

        # if the model only has one optimizer, use optimizer.zero_grad
        # otherwose use model.zero_grad
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        
        running_loss += loss.item() * data.size(0)
        losses = np.append(losses, [loss.item()])
        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{:4d}/{} ({:2.0f}%)] Loss: {:.3f}'.format(
                    epoch + 1, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    epoch_loss = running_loss / len(train_loader.dataset)
    # detach, won't use them to update model. Only for recording the loss.
    return epoch_loss, losses

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    test_auc = 0
    all_pred = np.empty((0, 9))
    all_target = np.array([])
    
    with torch.no_grad():
        for sequences, data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += get_loss(sequences, output, criterion).item()
            pred = output
            target, pred = get_targetPred(sequences, pred)
            all_pred = np.append(all_pred, pred, axis=0)
            all_target = np.append(all_target, target)
        test_auc = batch_auc(all_target, all_pred)# * data.size(0)
        test_acc = batch_acc(all_target, all_pred)# * data.size(0)
        
    test_loss /= len(test_loader)
    # test_auc /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, AUC: {:.4f}, Accuracy: {:.4f}\n'.format(test_loss, test_auc, test_acc))
    return test_loss, test_auc, test_acc


def predict_one_sequence(model, sequence: Sequence, device):
    model.eval()
    data = sequence.data.reshape(1, paramH.n_features, -1).to(device)
    output = model(data)
    _, output = trim_padding_and_flat([sequence], output)
    return output
