import hydra
import omegaconf
import torch
import os
import pandas as pd
import numpy as np
from omegaconf import ValueNode
from torch.utils.data import Dataset

from torch_geometric.data import Data
from CFtorch.common.data_utils import preprocess
from CFtorch.common.group_utils import dof0_table, wmax_table, fc_mask_table, lattice_mask_table, mult_table


class CrystDataset(Dataset):
    def __init__(self, path, use_exit, save_path, preprocess_workers,
                 n_atom_types, n_wyck_types, n_max, Nf=10, tol=0.01):
        super().__init__()
        self.path = path
        self.Nf = Nf
        self.n_wyck_types = n_wyck_types
        self.n_atom_types = n_atom_types
        print(path)

        if os.path.exists(save_path) and use_exit:
            self.cached_data = torch.load(save_path)
        else:
            self.cached_data = preprocess(path,preprocess_workers,n_atom_types,n_wyck_types,n_max,tol)
            torch.save(self.cached_data, save_path)

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        data = data_dict.copy()

        FTfrac_coor = [fn(2 * np.pi * data_dict['frac_coor'][:, None] * f) for f in range(1, self.Nf + 1) for fn in
                       (np.sin, np.cos)]
        FTfrac_coor = np.squeeze(np.stack(FTfrac_coor, axis=-1), axis=1)

        M = mult_table[data['G'] - 1, data['wyckoff']]


        data.update({
            'G': torch.LongTensor([data_dict['G']]),
            'num_sites': torch.LongTensor([data_dict['num_sites']]),
            'lattice': torch.Tensor(data_dict['lattice']).view(1, -1),
            'frac_coor': torch.Tensor(data_dict['frac_coor']),
            'FTfrac_coor': torch.Tensor(FTfrac_coor),
            'wyckoff': torch.LongTensor(data_dict['wyckoff']),
            'atom_type': torch.LongTensor(data_dict['atom_type']),
            'M': torch.LongTensor(M),
        })

        return data
