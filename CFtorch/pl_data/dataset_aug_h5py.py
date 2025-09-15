import hydra
import omegaconf
import torch
import os
import pandas as pd
import numpy as np
from omegaconf import ValueNode
from torch.utils.data import Dataset

from torch_geometric.data import Data
from CFtorch.common.data_utils_aug import preprocess
#from CFtorch.common.data_utils import preprocess
from CFtorch.common.group_utils import dof0_table, wmax_table, fc_mask_table, lattice_mask_table, mult_table
import h5py


class CrystDataset(Dataset):
    def __init__(self, path, use_exit, save_path, preprocess_workers,
                 n_atom_types, n_wyck_types, n_max, Nf=10, tol=0.01):
        super().__init__()
        self.path = path
        self.Nf = Nf
        self.n_wyck_types = n_wyck_types
        self.n_atom_types = n_atom_types
        self.save_path = save_path

        if os.path.exists(save_path) and use_exit:
            pass
        else:
            self.cached_data = preprocess(path,preprocess_workers,n_atom_types,n_wyck_types,n_max,tol)
            #torch.save(self.cached_data, save_path)
            with h5py.File(self.save_path, 'w') as f:
                for idx, data in enumerate(self.cached_data):
                    grp = f.create_group(f'data_{idx}')
                    grp.create_dataset('G', data=data['G'])
                    grp.create_dataset('num_sites', data=data['num_sites'])
                    grp.create_dataset('lattice', data=data['lattice'])
                    grp.create_dataset('frac_coor', data=data['frac_coor'])
                    grp.create_dataset('wyckoff', data=data['wyckoff'])
                    grp.create_dataset('atom_type', data=data['atom_type'])


    def __len__(self) -> int:
        with h5py.File(self.save_path, 'r') as f:
            num_groups = len(list(f.keys()))
        return num_groups 

    def __getitem__(self, index):
        with h5py.File(self.save_path, 'r') as f:
            grp = f[f"data_{index}"]
            G = grp['G'][()]
            num_sites = grp['num_sites'][()]
            lattice = grp['lattice'][()]
            frac_coor = grp['frac_coor'][()]
            wyckoff = grp['wyckoff'][()]
            atom_type = grp['atom_type'][()]
            data_dict = {'G': G, 'num_sites': num_sites, 'lattice': lattice, 'frac_coor': frac_coor, 'wyckoff':wyckoff, 'atom_type': atom_type}

        data = data_dict.copy()

        FTfrac_coor = [fn(2 * np.pi * data_dict['frac_coor'][:, None] * f) for f in range(1, self.Nf + 1) for fn in
                       (np.sin, np.cos)]
        FTfrac_coor = np.squeeze(np.stack(FTfrac_coor, axis=-1), axis=1)

        M = mult_table[data['G'] - 1, data['wyckoff']]

        data = Data(
                    G=torch.LongTensor([data_dict['G']]).unsqueeze(dim=0),
                    num_sites=torch.LongTensor([data_dict['num_sites']]).unsqueeze(dim=0),
                    lattice=torch.Tensor(data_dict['lattice']).unsqueeze(dim=0),
                    frac_coor=torch.Tensor(data_dict['frac_coor']).unsqueeze(dim=0),
                    FTfrac_coor=torch.Tensor(FTfrac_coor).unsqueeze(dim=0),
                    wyckoff=torch.LongTensor(data_dict['wyckoff']).unsqueeze(dim=0),
                    atom_type=torch.LongTensor(data_dict['atom_type']).unsqueeze(dim=0),
                    M=torch.LongTensor(M).unsqueeze(dim=0),
                )
        return data
