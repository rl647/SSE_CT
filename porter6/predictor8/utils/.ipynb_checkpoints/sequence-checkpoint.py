import re

def sequence_mapping_str(seq: str) -> str:
    '''
    Given a aequence, map rarely Amino Acids [U Z O B] to [X].
    
    params:
        seq - str, e.g. 'AETCZAO'
        
    return:
        the sequence with rarely AAs mapped to X.
    '''
    return re.sub(f'[UZOB]', 'X', seq)