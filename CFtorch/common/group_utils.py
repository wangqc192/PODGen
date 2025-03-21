import pandas as pd
import os
import numpy as np
import re
import sys

def from_xyz_str(xyz_str: str):
    """
    Args:
        xyz_str: string of the form 'x, y, z', '-x, -y, z', '-2y+1/2, 3x+1/2, z-y+1/2', etc.
    Returns:
        affine operator as a 3x4 array
    """
    rot_matrix = np.zeros((3, 3))
    trans = np.zeros(3)
    tokens = xyz_str.strip().replace(" ", "").lower().split(",")
    re_rot = re.compile(r"([+-]?)([\d\.]*)/?([\d\.]*)([x-z])")
    re_trans = re.compile(r"([+-]?)([\d\.]+)/?([\d\.]*)(?![x-z])")
    for i, tok in enumerate(tokens):
        # build the rotation matrix
        for m in re_rot.finditer(tok):
            factor = -1.0 if m.group(1) == "-" else 1.0
            if m.group(2) != "":
                factor *= float(m.group(2)) / float(m.group(3)) if m.group(3) != "" else float(m.group(2))
            j = ord(m.group(4)) - 120
            rot_matrix[i, j] = factor
        # build the translation vector
        for m in re_trans.finditer(tok):
            factor = -1 if m.group(1) == "-" else 1
            num = float(m.group(2)) / float(m.group(3)) if m.group(3) != "" else float(m.group(2))
            trans[i] = num * factor
    return np.concatenate( [rot_matrix, trans[:, None]], axis=1) # (3, 4)


df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../../data/wyckoff_list.csv'))
df['Wyckoff Positions'] = df['Wyckoff Positions'].apply(eval)  # convert string to list
wyckoff_positions = df['Wyckoff Positions'].tolist()

symops = np.zeros((230, 28, 576, 3, 4)) # 576 is the least common multiple for all possible mult
mult_table = np.zeros((230, 28), dtype=int) # mult_table[g-1, w] = multiplicity , 28 because we had pad 0
wmax_table = np.zeros((230,), dtype=int)    # wmax_table[g-1] = number of possible wyckoff letters for g
dof0_table = np.ones((230, 28), dtype=bool)  # dof0_table[g-1, w] = True for those wyckoff points with dof = 0 (no continuous dof)
fc_mask_table = np.zeros((230, 28, 3), dtype=bool) # fc_mask_table[g-1, w] = True for continuous fc

for g in range(230):
    wyckoffs = []
    for x in wyckoff_positions[g]:
        wyckoffs.append([])
        for y in x:
            wyckoffs[-1].append(from_xyz_str(y))
    wyckoffs = wyckoffs[::-1] # a-z,A

    mult = [len(w) for w in wyckoffs]
    mult_table[g, 1:len(mult)+1] = mult
    wmax_table[g] = len(mult)

    # print (g+1, [len(w) for w in wyckoffs])
    for w, wyckoff in enumerate(wyckoffs):
        wyckoff = np.array(wyckoff)
        repeats = symops.shape[2] // wyckoff.shape[0]
        symops[g, w+1, :, :, :] = np.tile(wyckoff, (repeats, 1, 1))
        dof0_table[g, w+1] = np.linalg.matrix_rank(wyckoff[0, :3, :3]) == 0
        fc_mask_table[g, w+1] = np.abs(wyckoff[0, :3, :3]).sum(axis=1)!=0

# 1-2
# 3-15
# 16-74
# 75-142
# 143-194
# 195-230
mask = [1, 1, 1, 1, 1, 1] * 2 +\
       [1, 1, 1, 0, 1, 0] * 13+\
       [1, 1, 1, 0, 0, 0] * 59+\
       [1, 0, 1, 0, 0, 0] * 68+\
       [1, 0, 1, 0, 0, 0] * 52+\
       [1, 0, 0, 0, 0, 0] * 36

lattice_mask_table = np.array(mask).reshape(230, 6)