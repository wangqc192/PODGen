from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.composition import Composition

import time
import argparse
import torch
import hydra
import fnmatch
import copy
from pathlib import Path
import random

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(parent_dir)

from omegaconf import DictConfig, OmegaConf
from hydra.experimental import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from predictor.data_utils import StandardScalerTorch

def load_pre_model(model_path, model_file, cfg=None, num_target=None):
    if cfg==None:
        GlobalHydra.instance().clear()
        initialize_config_dir(config_dir=model_path)
        cfg: DictConfig = compose(config_name="hparams")
        cfg.model._target_ = 'predictor.'+cfg.model._target_
        model = hydra.utils.instantiate(cfg.model)

        model_root = os.path.join(model_path, model_file)
        checkpoint = torch.load(model_root, map_location=torch.device('cpu'))
        model_state_dict = checkpoint['model']
        model.load_state_dict(model_state_dict)
        scaler = Path(model_path) / 'prop_scaler.json'
        if os.path.exists(scaler):
            scaler = StandardScalerTorch.load_from_file(scaler)
            model.scaler = scaler
        return model, cfg

    else:
        copied_cfg = copy.deepcopy(cfg)
        if num_target != None:
            copied_cfg.model.num_targets = num_target
        model = hydra.utils.instantiate(copied_cfg.model)
        model_root = os.path.join(model_path, model_file)
        checkpoint = torch.load(model_root, map_location=torch.device('cpu'))
        model_state_dict = checkpoint['model']
        model.load_state_dict(model_state_dict)
        scaler = Path(model_path) / 'prop_scaler.json'
        if os.path.exists(scaler):
            scaler = StandardScalerTorch.load_from_file(scaler)
            model.scaler = scaler
        return model, None
    

def pt2structures(path):
    data = torch.load(path,map_location=torch.device('cpu'))

    lattices = data['lattices']
    num_atoms = data['num_atoms']
    frac_coors = data['frac_coords']
    atom_types = data['atom_types']

    lattices_list = lattices.numpy()
    num_atoms_list = num_atoms.tolist()
    frac_coors_list = frac_coors.numpy().tolist()
    atom_types_list = atom_types.tolist()

    num_materal = 0
    now_atom = 0
    struct_list = []
    for i in range(len(num_atoms_list)):

        lattice = Lattice(lattices_list[i,:,:])
        atom_num = num_atoms_list[i]
        frac_coord = frac_coors_list[now_atom: now_atom + atom_num][:]
        atom_type = atom_types_list[now_atom: now_atom + atom_num]

        crystal = Structure(lattice, atom_type, frac_coord, to_unit_cell=True)

        crystal = crystal.get_reduced_structure()

        structure = Structure(
            lattice=Lattice.from_parameters(*crystal.lattice.parameters),
            species=crystal.species,
            coords=crystal.frac_coords,
            coords_are_cartesian=False,
        )
        struct_list.append(structure)
        now_atom += atom_num
        num_materal += 1
    return struct_list


def pt2structures2(path):
    data = torch.load(path,map_location=torch.device('cpu'))

    lengths = data['lengths']
    angles = data['angles']
    num_atoms = data['num_atoms']
    frac_coors = data['frac_coords']
    atom_types = data['atom_types']
    
    lengths_list = lengths.numpy().tolist()
    angles_list = angles.numpy().tolist()
    num_atoms_list = num_atoms.tolist()
    frac_coors_list = frac_coors.numpy().tolist()
    atom_types_list = atom_types.tolist()
    
    num_materal = 0
    struct_list = []
    for i in range(len(num_atoms_list)): #第i个batch？
        now_atom = 0
        for a in range(len(num_atoms_list[i])): #第a个材料
            length = lengths_list[i][a]
            angle = angles_list[i][a]
            atom_num = num_atoms_list[i][a]
    
            atom_type = atom_types_list[i][now_atom: now_atom + atom_num]
            frac_coord = frac_coors_list[i][now_atom: now_atom + atom_num][:]
            lattice = Lattice.from_parameters(a=length[0], b=length[1], c=length[2], alpha=angle[0],
                                                beta=angle[1], gamma=angle[2])
    
            structure = Structure(lattice, atom_type, frac_coord, to_unit_cell=True)
            struct_list.append(structure)
            
            now_atom += atom_num
            num_materal += 1
            
    return struct_list

def cif2stuctures(directory_path):
    structures = []
    filename_list = []
    
    # 遍历指定路径及其子目录
    for root, dirs, files in os.walk(directory_path):
        for filename in fnmatch.filter(files, '*.cif'):
            cif_file_path = os.path.join(root, filename)
            # 解析 CIF 文件
            structure = Structure.from_file(cif_file_path)
            structures.append(structure)
            filename_list.append(filename[:-4])
    
    return structures,filename_list