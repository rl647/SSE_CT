import os
from typing import List

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset.utils import parse_target, read_plm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

class Sequence:
    def __init__(self, seq_id, target, feature_path=None, data_transform=None,
                 target_transform=None, feature_type='protTrans'):
        
        self.seq_id = seq_id
        self._target = parse_target(target)
            
        self.data_transform = data_transform
        self.target_transform = target_transform
        if feature_type=='protTrans':
            self.plm_encoding = read_plm(os.path.join(feature_path, 'protTrans/{}.npy'.format(self.seq_id)))
        elif feature_type=='onehot':
            self.plm_encoding = read_plm(os.path.join(feature_path, 'onehot/{}.npy'.format(self.seq_id)))
        elif feature_type=='esm2':
            self.plm_encoding = read_plm(os.path.join(feature_path, 'esm2/{}.npy'.format(self.seq_id)))
    @property
    def data(self):
        data = self.plm_encoding.T.squeeze(0)
        #print(data.size())
        return data.float()

    @property
    def target(self):
        target = self._target
        if self.target_transform is not None:
            target = self.target_transform(target)

        #print(target.size())
        return target.float()

    @property
    def clean_target(self):
        return self._target.numpy()

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, i):
        return self.data, self.target

    def as_dict(self):
        return {"seq_id": self.seq_id, "target": self.target, "data": self.data}


# Base class for the two datasets, with common functionality
class SSDataset(Dataset):
    def __init__(self, data, feature_root, transform=None, target_transform=None, feature_type='plm'):
        self.transform = transform
        self.target_transform = target_transform
        self.raw_data = data
        self.feature_root = feature_root
        self.feature_type = feature_type

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        seq_id, _, target = self.raw_data.iloc[idx]
        item = Sequence(seq_id, target, feature_path=self.feature_root, feature_type=self.feature_type,
                                      data_transform=self.transform, target_transform=self.target_transform)
        return item

def pad_packed_collate(batch: List[Sequence]):
    """Puts data, and lengths into a packed_padded_sequence then returns
       the packed_padded_sequence and the labels. Set use_lengths to True
       to use this collate function.
       Args:
         batch: (list of tuples) [(sequence, target)].
             sequence is a FloatTensor
             target has the same variable length with sequence
       Output:
         packed_batch: (PackedSequence), see torch.nn.utils.rnn.pack_padded_sequence
         labels: (Tensor), labels from the file names of the wav.
    """

    if len(batch) == 1:
        seqs, labels = [batch[0].data], [batch[0].target]
        lengths = [seqs[0].size(0)]
        
    if len(batch) > 1:
        # get data and sorted by the length of sequence
        seqs, labels, lengths = zip(*[(item.data.T, item.target, item.data.size(1)) for item in sorted(batch, key=lambda x: x.data.size(1), reverse=True)])
    seqs = pad_sequence(seqs, batch_first=True)
    labels = pad_sequence(labels, batch_first=True)
    packed_seqs = pack_padded_sequence(seqs, lengths, batch_first=True)
    packed_labels = pack_padded_sequence(labels, lengths, batch_first=True)
    
    return batch, packed_seqs, packed_labels
    
def collate_fn(batch: List[Sequence]):
    data = torch.stack([item.data for item in batch])
    target = torch.stack([item.target for item in batch])
    return batch, data, target

