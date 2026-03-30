#%%
import os
contact_map_path = ''# home to contact map path you can find it at data/ssnw.tar.xz 
import string
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
element_L = {}
for key, val in score.items():
    element_L[val]=key
    element_L[val.lower()]=key
element = {}
osc = {}
e_score = {}
h_score = {}
for i,e in zip(H,E):
    element[i]='H'
    element[e]='E'
for i,e in enumerate(E):
    if i<=8:
        h_score[e.lower()] = (i*1)+2
        e_score[e] = i+2
    elif i>8 and i<=18:
        h_score[e.lower()] = (i*2)-6
        e_score[e] = i*2-6
    elif i>18:
        h_score[e.lower()] = (i*3)-24
        e_score[e] = i*3-24
osc.update(e_score)
osc.update(h_score)
def ss_extraction(protein,contact_map_path=contact_map_path):
    f = open(f'{contact_map_path}/{protein}.ssnw')
    length = f.readline().strip().split()[1:]
    element = f.readline().strip().split()[1:]
    start = 0
    ss = []
    ss_range = []
    for i in range(len(element)):
        e=element[i]
        length[i]=int(length[i])
        if 'T' not in element[i]:
            ss_range.append([start,int(int(length[i])+start)])
            if length[i] in score:
                cs = score[length[i]] if e[0] == 'E' else score[length[i]].lower()
            elif length[i]+1 in score:
                cs = score[length[i]+1] if e[0] == 'E' else score[length[i]+1].lower()
            elif length[i]+2 in score:
                cs = score[length[i]+2] if e[0] == 'E' else score[length[i]+2].lower()
            elif length[i]>51:
                cs = score[51] if e[0] == 'E' else score[51].lower()
            ss.append(cs)
        start += int(length[i])
    return ss_range, ''.join(ss)

