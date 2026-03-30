import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import re

from utils.common import load_np

class PadRightTo(object):
    """Pad the tensor to a given size.

    Args:
        output_size (int): Desired output size.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, sample):
        padding = self.output_size - sample.size()[-1]
        return torch.nn.functional.pad(sample, (0, padding), 'constant', 0)


# Function to one-hot encode a list of labels
def one_hot_encode(labels_tensor, num_classes):
    # Convert to one-hot encoding
    one_hot_encoded = F.one_hot(labels_tensor, num_classes)
    return one_hot_encoded


def read_plm(plm_path):
    plm = load_np(plm_path)
    # plm = plm.dropna().astype(np.float32)
    plm = torch.tensor(plm, dtype=torch.float32)
    return plm

def parse_target(target, num_classes=9):
    '''
    params:
        target - str, e.g. '1000000110000'
    '''
    new_target = torch.tensor([int(t) for t in target])
    # new_target = one_hot_encode(new_target, num_classes)
    return new_target

def sequence_mapping(list_seq: list) -> list:
    '''
    Given a list of sequences, map rarely Amino Acids [U Z O B] to [X].
    
    params:
        list_seq - list of sequences, e.g. ['A E T C Z A O', 'S K T Z P']
        
    return:
        the list of sequences with rarely AAs mapped to X.
    '''
    return [re.sub(f'[UZOB]', 'X', sequence) for sequence in list_seq]
