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
        self.n_max = n_max

        if os.path.exists(save_path) and use_exit:
            pass
        else:
            self.cached_data = preprocess(path,preprocess_workers,n_atom_types,n_wyck_types,n_max,tol)
            #torch.save(self.cached_data, save_path)
            self._write_hdf5(self.cached_data)
        with h5py.File(self.save_path, 'r') as f:
                    self.length = f['G'].shape[0]

        self._file = None

    def _write_hdf5(self, data_list):
        N = len(data_list)
        G_arr = np.zeros((N,), dtype=np.int64)
        num_sites = np.zeros((N,), dtype=np.int64)
        lattice_arr = np.zeros((N,6), dtype=np.float32)
        frac_coor_arr = np.zeros((N,self.n_max,3), dtype=np.float32)
        wyckoff_arr = np.zeros((N,self.n_max), dtype=np.int64)
        atom_type_arr = np.zeros((N,self.n_max), dtype=np.int64)

        for i, d in enumerate(data_list):
            G_arr[i] = d['G']
            num_sites[i] = d['num_sites']
            lattice_arr[i] = d['lattice']
            frac_coor_arr[i] = d['frac_coor']       # 长度 n_max
            wyckoff_arr[i] = d['wyckoff']           # 长度 n_max
            atom_type_arr[i] = d['atom_type']       # 长度 n_max

        with h5py.File(self.save_path, 'w') as f:
            f.create_dataset('G', data=G_arr, chunks=True)
            f.create_dataset('num_sites', data=num_sites, chunks=True)
            f.create_dataset('lattice', data=lattice_arr, chunks=True)
            f.create_dataset('frac_coor', data=frac_coor_arr, chunks=True)
            f.create_dataset('wyckoff', data=wyckoff_arr, chunks=True)
            f.create_dataset('atom_type', data=atom_type_arr, chunks=True)
         
    def __len__(self) -> int:
        return self.length 

    def __getitem__(self, index):
        if self._file is None:
            self._file = h5py.File(self.save_path, 'r')

        G = self._file['G'][index]
        num_sites = self._file['num_sites'][index]
        lattice = self._file['lattice'][index]
        frac_coor = self._file['frac_coor'][index]       # 长度 n_max
        wyckoff = self._file['wyckoff'][index]           # 长度 n_max
        atom_type = self._file['atom_type'][index]       # 长度 n_max

        FTfrac_coor = [fn(2 * np.pi * frac_coor[:, None] * f) for f in range(1, self.Nf + 1) for fn in
                       (np.sin, np.cos)]
        FTfrac_coor = np.squeeze(np.stack(FTfrac_coor, axis=-1), axis=1)

        M = mult_table[G - 1, wyckoff]

        data = Data(
                    G=torch.LongTensor([G]).unsqueeze(dim=0),
                    num_sites=torch.LongTensor([num_sites]).unsqueeze(dim=0),
                    lattice=torch.Tensor(lattice).unsqueeze(dim=0),
                    frac_coor=torch.Tensor(frac_coor).unsqueeze(dim=0),
                    FTfrac_coor=torch.Tensor(FTfrac_coor).unsqueeze(dim=0),
                    wyckoff=torch.LongTensor(wyckoff).unsqueeze(dim=0),
                    atom_type=torch.LongTensor(atom_type).unsqueeze(dim=0),
                    M=torch.LongTensor(M).unsqueeze(dim=0),
                )
        return data
