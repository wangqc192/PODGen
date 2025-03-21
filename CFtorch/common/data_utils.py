import pandas as pd
import numpy as np
from pyxtal import pyxtal
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from p_tqdm import p_umap

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


def process_one(row, atom_types, wyck_types, n_max, tol):
    #print('0')
    crystal_str = row['cif']
    #print('1')
    crystal = Structure.from_str(crystal_str, fmt='cif')
    #print('2')
    spga = SpacegroupAnalyzer(crystal, symprec=tol)
    #print('3')
    crystal = spga.get_refined_structure()
    #print('4')
    c = pyxtal()
    try:
        c.from_seed(crystal, tol=tol)
    except:
        c.from_seed(crystal, tol=0.001)
    #print('5')

    g = c.group.number
    num_sites = len(c.atom_sites)
    assert (n_max > num_sites)  # we will need at least one empty site for output of L params
    #print('6')  
    natoms = 0
    ww = []
    aa = []
    fc = []
    ws = []
    for site in c.atom_sites:
        a = element_list.index(site.specie)
        x = site.position
        m = site.wp.multiplicity
        w = letter_to_number(site.wp.letter)
        symbol = str(m) + site.wp.letter
        natoms += site.wp.multiplicity
        assert (a < atom_types)
        assert (w < wyck_types)
        assert (np.allclose(x, site.wp[0].operate(x)))
        aa.append(a)
        ww.append(w)
        fc.append(x)  # the generator of the orbit
        ws.append(symbol)
        # print ('g, a, w, m, symbol, x:', g, a, w, m, symbol, x)
    #print('7')
    idx = np.argsort(ww)
    ww = np.array(ww)[idx]
    aa = np.array(aa)[idx]
    fc = np.array(fc)[idx].reshape(num_sites, 3)
    ws = np.array(ws)[idx]
    # print (ws, aa, ww, natoms)
    #print('8')
    #padding
    aa = np.concatenate([aa,
                         np.full((n_max - num_sites,), 0)],
                        axis=0)

    ww = np.concatenate([ww,
                         np.full((n_max - num_sites,), 0)],
                        axis=0)
    fc = np.concatenate([fc,
                         np.full((n_max - num_sites, 3), 1e10)],
                        axis=0)
    #print('9')
    abc = np.array([c.lattice.a, c.lattice.b, c.lattice.c]) / natoms ** (1. / 3.) #The reduced lattice length
    angles = np.array([c.lattice.alpha, c.lattice.beta, c.lattice.gamma])
    l = np.concatenate([abc, angles])
    #print('10')
    result_dict={
        'mp_id': row['material_id'],
        'cif': crystal_str,
        'G': g,
        'lattice': l,
        'frac_coor': fc,
        'atom_type': aa,
        'wyckoff': ww,
        'num_sites': num_sites
    }
    #print('11')
    return result_dict


def preprocess(input_file, num_workers, n_atom_types, n_wyck_types, n_max, tol=0.01):
    df = pd.read_csv(input_file)
    #print('load df')
    unordered_results = p_umap(
        process_one,
        [df.iloc[idx] for idx in range(len(df))],
        [n_atom_types] * len(df),
        [n_wyck_types] * len(df),
        [n_max] * len(df),
        [tol] * len(df),
        num_cpus=num_workers)

    mpid_to_results = {result['mp_id']: result for result in unordered_results}
    ordered_results = [mpid_to_results[df.iloc[idx]['material_id']]
                       for idx in range(len(df))]

    return ordered_results