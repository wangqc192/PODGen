from pathlib import Path
import argparse
import ast
from collections import Counter
import multiprocessing
import time
    
import pandas as pd
import numpy as np
import torch

from scripts.eval_utils import load_model
from podgen.mcmc_utils import generate
from pymatgen.core import Composition
from CFtorch.pl_modules.model import CrystalFormer

element_list = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']

element_dict = {value: index for index, value in enumerate(element_list)}

def letter_to_number(letter):
    """
    'a' to 1 , 'b' to 2 , 'z' to 26, and 'A' to 27 
    """
    return ord(letter) - ord('a') + 1 if 'a' <= letter <= 'z' else 27 if letter == 'A' else None

def gen(batch_size, model, spacegroup, top_p=1.0, temperature=1.0, w_mask=None, atom_mask=None):
    data = generate(batch_size, model, spacegroup, top_p=1.0, temperature=1.0, w_mask=w_mask, atom_mask=atom_mask)
    data['lattice'].reshape(batch_size,-1)
    data['lattice']=data['lattice'].reshape(batch_size,-1)
    for i in data.keys():
        data[i] = data[i].tolist()
    new_data = {}
    for i in range(len(data['lattice'])):
        data['lattice'][i][-3:] = [j * 180/np.pi for j in data['lattice'][i][-3:]]
    new_data["L"] = data["lattice"]
    new_data["X"] = data["frac_coor"]
    new_data["A"] = data["atom_type"]
    new_data["W"] = data["wyckoff"]
    new_data["M"] = data["M"]
    df = pd.DataFrame(new_data)
    return df
    
def filt_formula(df, target_formula):
    a = df["A"]
    m = df["M"]
    meeting_idx = []
    for i in range(len(a)):
        a_i = [x for x in a[i] if x !=0]
        a_i = [element_list[x] for x in a_i]
        m_i = [x for x in m[i] if x !=0]
        composition_counter = Counter()
        for ele, multi in zip(a_i, m_i):
            composition_counter[ele] += multi
        gen_formula = Composition(dict(composition_counter))
        if gen_formula.reduced_formula == target_formula.reduced_formula:
            print(gen_formula)
            print(target_formula)
            meeting_idx.append(i)
    return df.iloc[meeting_idx]

            
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', type=str, default="/home/wangqc/PODGen/output/hydra/singlerun/2025-06-27/mp20", help='')
    parser.add_argument('--file_for_gen', type=str, default="input.csv", help='')
    parser.add_argument('--out_dir', type=str, default="gen", help='')
    parser.add_argument('--batch_size', type=int, default="128",help='')
    parser.add_argument('--spacegroup', type=int, nargs='+', help='The space group id to be sampled (1-230)')
    parser.add_argument('--top_p', type=float, default=1.0, help='1.0 means un-modified logits, smaller value of p give give less diverse samples')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature used for sampling')
    parser.add_argument('--wyckoff', type=str, default=None, nargs='+', help='The Wyckoff positions to be sampled, e.g. a, b')
    parser.add_argument('--elements', type=str, default=None, nargs='+', help='name of the chemical elemenets, e.g. Bi, Ti, O')
    parser.add_argument('--formula', type=str, default=None, nargs='+', help='formula e.g. NaCl')

    parser.add_argument('--atom_types', type=int, default=119, help='Atom types including the padded atoms')
    parser.add_argument('--n_max', type=int, default=21, help='The maximum number of atoms in the cell')
    
    args = parser.parse_args()

    model, _, _ = load_model(Path(args.model_path), load_data=True)
    if torch.cuda.is_available():
        model = model.to('cuda')

    if args.file_for_gen is not None:
        df = pd.read_csv(args.file_for_gen)
        formula = df['formula'].tolist()
        spacegroup = df['spacegroup'].tolist()
        target_formulas = []
        atom_masks = []

        for i in range(len(formula)):
            formu = Composition(formula[i])
            idx = [element_dict[e] for e in formu.as_dict().keys()]
            atom_mask = [1] + [1 if a in idx else 0 for a in range(1, args.atom_types)]
            atom_mask = torch.tensor(atom_mask, dtype=bool)
            atom_mask = torch.stack([atom_mask] * args.n_max, axis=0)
            atom_masks.append(atom_mask)
            target_formulas.append(formu)
        print ('sampling structure formed by these elements:', formula)
    else:
        formula = None



        
    if args.wyckoff is not None:
        idx = [letter_to_number(w) for w in args.wyckoff]
        # padding 0 until the length is args.n_max
        w_mask = idx + [0]*(args.n_max -len(idx))
        # w_mask = [1 if w in idx else 0 for w in range(1, args.wyck_types+1)]
        w_mask = torch.tensor(w_mask, dtype=int)
        print ('sampling structure formed by these Wyckoff positions:', args.wyckoff)
        print (w_mask)
    else:
        w_mask = None
  
    
    all_filt_df = pd.DataFrame()
    
    out_dir = Path(args.model_path).joinpath(args.out_dir)
    if out_dir:
        csv_files = list(out_dir.glob('*.csv'))
        csv_files = [int(i.stem) for i in csv_files]
        finish = max(csv_files)
        out_dir.joinpath(f'err_after_{finish}').touch()
        start = finish+1
    else:   
        start = 0
        
    for i in range(start, len(formula)):
        print(f"generate {i} start")
        start_time = time.time()
        while True:
            current_time = time.time()
            if current_time - start_time > 60:
                break  
            gen_df = gen(args.batch_size, model, spacegroup[i], top_p=1.0, temperature=1.0, w_mask=w_mask, atom_mask=atom_masks[i])
            filt_df = filt_formula(gen_df, target_formulas[i])
            if len(filt_df) > 0:
                all_filt_df = pd.concat([all_filt_df, filt_df],ignore_index=True)
            if len(all_filt_df) >= 2:
                break
        out_path = Path(args.model_path).joinpath(args.out_dir).joinpath(str(i) + '.csv')

        all_filt_df.to_csv(out_path)
        all_filt_df = pd.DataFrame()
        print(f"generate {i} finish")
    
    #if formula is not None:
        #for i in range(len(formula)):
            #print(target_formulas[i])
            #filt_df = filt_formula(gen_dfs[i], target_formulas[i])
            #out_path = Path(args.model_path).joinpath(formula[i] + str(spacegroup[i]) + '.csv')
            #filt_df.to_csv(out_path)
    
    