from ase.optimize.bfgs import BFGS
from ase.io import write
from p_tqdm import p_umap
from deepmd.calculator import DP
from ase.io.cif import write_cif, read_cif
from ase.io.trajectory import Trajectory
import os
from io import StringIO
from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd
from multiprocessing import Pool
from ase.filters import FrechetCellFilter


ENERGY_REF = {
    "Ne": -0.0259,
    "He": -0.0091,
    "Ar": -0.0688,
    "F": -1.9115,
    "O": -4.9467,
    "Cl": -1.8485,
    "N": -8.3365,
    "Kr": -0.0567,
    "Br": -1.553,
    "I": -1.4734,
    "Xe": -0.0362,
    "S": -4.1364,
    "Se": -3.4959,
    "C": -9.2287,
    "Au": -3.2739,
    "W": -12.9581,
    "Pb": -3.7126,
    "Rh": -7.3643,
    "Pt": -6.0711,
    "Ru": -9.2744,
    "Pd": -5.1799,
    "Os": -11.2274,
    "Ir": -8.8384,
    "H": -3.3927,
    "P": -5.4133,
    "As": -4.6591,
    "Mo": -10.8457,
    "Te": -3.1433,
    "Sb": -4.129,
    "B": -6.6794,
    "Bi": -3.8405,
    "Ge": -4.623,
    "Hg": -0.3037,
    "Sn": -4.0096,
    "Ag": -2.8326,
    "Ni": -5.7801,
    "Tc": -10.3606,
    "Si": -5.4253,
    "Re": -12.4445,
    "Cu": -4.0992,
    "Co": -7.1083,
    "Fe": -8.47,
    "Ga": -3.0281,
    "In": -2.7517,
    "Cd": -0.9229,
    "Cr": -9.653,
    "Zn": -1.2597,
    "V": -9.0839,
    "Tl": -2.3626,
    "Al": -3.7456,
    "Nb": -10.1013,
    "Be": -3.7394,
    "Mn": -9.162,
    "Ti": -7.8955,
    "Ta": -11.8578,
    "Pa": -9.5147,
    "U": -11.2914,
    "Sc": -6.3325,
    "Np": -12.9478,
    "Zr": -8.5477,
    "Mg": -1.6003,
    "Th": -7.4139,
    "Hf": -9.9572,
    "Pu": -14.2678,
    "Lu": -4.521,
    "Tm": -4.4758,
    "Er": -4.5677,
    "Ho": -4.5824,
    "Y": -6.4665,
    "Dy": -4.6068,
    "Gd": -14.0761,
    "Eu": -10.257,
    "Sm": -4.7186,
    "Nd": -4.7681,
    "Pr": -4.7809,
    "Pm": -4.7505,
    "Ce": -5.9331,
    "Yb": -1.5396,
    "Tb": -4.6344,
    "La": -4.936,
    "Ac": -4.1212,
    "Ca": -2.0056,
    "Li": -1.9089,
    "Sr": -1.6895,
    "Na": -1.3225,
    "Ba": -1.919,
    "Rb": -0.9805,
    "K": -1.1104,
    "Cs": -0.8954,
}


def get_e_form_per_atom(
    structure, energy: float, ref = ENERGY_REF
):
    comp = structure.get_chemical_symbols()
    natoms = len(comp)
    e_form = energy - sum(ref[ele] for ele in comp)
    return e_form / natoms


def chack_ele(structure, ref = ENERGY_REF):
    comps = structure.get_chemical_symbols()
    use_ele = list(ref.keys())
    for comp in comps:
        if comp not in use_ele:
            return False
    return True


def process_cif(material_id ,cif_str, threshold=0.02, max_step=200):
    cif_file = StringIO(cif_str)  
    atoms = read_cif(cif_file) 
    if not chack_ele(atoms):
        return (material_id, 'fail for elemet out of range', False ,None)
    atoms.calc = calc  
    ecf = FrechetCellFilter(atoms)
    

    rlx = BFGS(ecf, trajectory=None, logfile=None)
    rlx.run(fmax=threshold, steps=max_step)

    fmax = np.max(np.linalg.norm(atoms.get_forces(), axis=1))
    energy = atoms.get_potential_energy()
    formation_energy = get_e_form_per_atom(atoms, energy)
    if fmax > threshold:
        return (material_id, f'fail for force do not convergence fmax={fmax}', False , formation_energy)
    energy = atoms.get_potential_energy()
    formation_energy = get_e_form_per_atom(atoms, energy)

    cif_output = BytesIO()
    write(cif_output, atoms, format='cif')
    
    return (material_id, cif_output.getvalue().decode('utf-8'), True , formation_energy)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_csv', default='/data/work/cyye/0-project/11-PODGen/PODGen/testfile/test.csv', type=str)
    parser.add_argument('--out_csv', default='/data/work/cyye/0-project/11-PODGen/PODGen/testfile/relaxtest.csv', type=str)
    parser.add_argument('--dpmodel', default='/data/work/cyye/2-other/dp0929.pth', type=str)
    parser.add_argument('--numworker', default=5, type=int)
    parser.add_argument('--threshold', default=0.02, type=float)
    parser.add_argument('--max_step', default=100, type=int)
    args = parser.parse_args()

    df = pd.read_csv(args.start_csv)
    # df = df.head(10)
    generate_cif = list(df['cif'])
    material_id = list(range(len(generate_cif)))
    calc = DP(Path(args.dpmodel))
    threshold = args.threshold

    cif_results = p_umap(process_cif, 
                         material_id, 
                         generate_cif, 
                         [args.threshold]*len(generate_cif), 
                         [args.max_step]*len(generate_cif), 
                         num_cpus=args.numworker)
    sorted_cif_results = sorted(cif_results, key=lambda x: material_id.index(x[0]))

    relaxed_cif = [result[1] for result in sorted_cif_results]
    relaxed_result = [result[2] for result in sorted_cif_results]
    formation_energy_pre_atom = [result[3] for result in sorted_cif_results]

    df['relaxed_cif'] = relaxed_cif
    df['relaxed_result'] = relaxed_result
    df['formation_energy_pre_atom'] = formation_energy_pre_atom
    
    df.to_csv(args.out_csv, index=False)

