import hydra
import omegaconf
import torch
import os
import pandas as pd
import numpy as np
from omegaconf import ValueNode

from torch_geometric.data import Data, Dataset
from CFtorch.common.data_utils_aug import preprocess
from CFtorch.common.group_utils import dof0_table, wmax_table, fc_mask_table, lattice_mask_table, mult_table


class CrystDataset(Dataset):
    def __init__(self, root, filename, use_exit, preprocess_workers,
                 n_atom_types, n_wyck_types, n_max, Nf=10, tol=0.01,
                 transform=None, pre_transform=None):

        self.filename = filename
        self.Nf = Nf
        self.n_wyck_types = n_wyck_types
        self.n_atom_types = n_atom_types
        self.use_exit = use_exit
        self.preprocess_workers = preprocess_workers
        self.n_max = n_max
        self.tol = tol

        super().__init__(root, transform, pre_transform)

        if not use_exit:
            self.process()

        self._indices = list(range(len(os.listdir(self.processed_dir))-2))

    @property
    def raw_file_names(self):
        return [self.filename]

    @property
    def processed_file_names(self):
        files = sorted(os.listdir(self.processed_dir)) if os.path.exists(self.processed_dir) else []
        return files


    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            data_list = preprocess(
                raw_path, self.preprocess_workers,
                self.n_atom_types, self.n_wyck_types, self.n_max, self.tol
            )

            os.makedirs(self.processed_dir, exist_ok=True)
            for data_dict in data_list:
                torch.save(data_dict, os.path.join(self.processed_dir, f"data_{idx}.pt"))
                idx += 1

    def len(self):
        return len(self._indices)

    def get(self, idx):
        data_dict = torch.load(os.path.join(self.processed_dir, f"data_{idx}.pt"))
        
        FTfrac_coor = [fn(2 * np.pi * data_dict['frac_coor'][:, None] * f)
                       for f in range(1, self.Nf + 1)
                       for fn in (np.sin, np.cos)]
        FTfrac_coor = np.squeeze(np.stack(FTfrac_coor, axis=-1), axis=1)

        M = mult_table[data_dict['G'] - 1, data_dict['wyckoff']]

        data = Data(
            G=torch.LongTensor([data_dict['G']]),
            num_sites=torch.LongTensor([data_dict['num_sites']]),
            lattice=torch.Tensor(data_dict['lattice']).view(1, -1),
            frac_coor=torch.Tensor(data_dict['frac_coor']),
            FTfrac_coor=torch.Tensor(FTfrac_coor),
            wyckoff=torch.LongTensor(data_dict['wyckoff']),
            atom_type=torch.LongTensor(data_dict['atom_type']),
            M=torch.LongTensor(M),
        )
        return data
