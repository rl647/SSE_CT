# %%
import os 
import numpy as np
import random
import string
import matplotlib.pyplot as plt
import sys
predicted_ss=sys.argv[1]
out_sse = sys.argv[2]
#%%
H = list(string.ascii_lowercase)
E = list(string.ascii_uppercase)
HE = H+E
score = {}
for i, e in enumerate(E):
    if i <= 8:
        score[i+2] = e
    elif i > 8 and i <= 18:
        score[(i*2)-6] = e
    elif i > 18:
        score[(i*3)-24] = e
#%%
ss = {}

f=open(f'{predicted_ss}')
for line in f:
    s=line.strip().split('\t')
    if len(s)>1:
        ss[s[0]] = s[1]
f.close()
#%%
from itertools import groupby
sse = {}
ss_range = {}
for key, val in ss.items():
    sse[key] = ''
    ss_range[key] = []
    vv = val[:]
    val = list(filter(None,(val.strip().split('C'))))
    
    for i, e in enumerate(val):
        ee = list(set(list(e)))
        if len(ee)==1:
            i  = len(e)
            
            if i>1:
                idx=vv.index(e)
                ss_range[key].append([idx,idx+i])
                vv = vv[:idx]+i*'-'+vv[idx+i:]
                if i in score:
                    cs = score[i] if e[0] == 'E' else score[i].lower()
                elif i+1 in score:
                    cs = score[i+1] if e[0] == 'E' else score[i+1].lower()
                elif i+2 in score:
                    cs = score[i+2] if e[0] == 'E' else score[i+2].lower()
                elif i>51:
                    cs = score[51] if e[0] == 'E' else score[51].lower()
                sse[key]+=cs
        else:
            c=1
            e = ["".join(group) for _, group in groupby(e)]
            for i2, e2 in enumerate(e):
    
                i  = len(e2)
               
                if i>1:
                    idx=vv.index(e2)
                    ss_range[key].append([idx,idx+i])
                    vv = vv[:idx]+i*'-'+vv[idx+i:]
                    if i in score:
                        cs = score[i] if e[0] == 'E' else score[i].lower()
                    elif i+1 in score:
                        cs = score[i+1] if e[0] == 'E' else score[i+1].lower()
                    elif i+2 in score:
                        cs = score[i+2] if e[0] == 'E' else score[i+2].lower()
                    elif i>51:
                        cs = score[51] if e[0] == 'E' else score[51].lower()
                    sse[key]+=cs
    if len(sse[key])!=len(ss_range[key]):
        print(sse[key],ss_range[key],len(sse[key]),len(ss_range[key]))
#%%
f=open(f'{out_sse}/sse.txt','w')
for key, val in sse.items():
    if len(val)<5:
        continue
    f.write(key+'\t'+val+'\n')
f.close()

f=open(f'{out_sse}/sse_range.txt','w')
for key, val in ss_range.items():
    if len(val)<5:
        continue
    f.write(key+'\t')
    for i, e in enumerate(val):
        f.write(f'{e[0]}-{e[1]}\t')
    f.write('\n')
f.close()