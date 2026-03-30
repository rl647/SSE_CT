import numpy as np

def batch_idx(num_seq: int, batch_size: int) -> list:
    '''
    Given the number of the sequences & batch_size, return a list of the start_idx of each batch.
    
    params:
        num_seq - nnumber of sequences
        batch_size - the number of sequences in each batch.
        
    return:
        b_idx - list of idx numbers.
    '''
    
    b_idx = list(np.arange(0, num_seq, batch_size))
    if b_idx[-1] < (num_seq):
        b_idx.append(num_seq)
        
    return b_idx