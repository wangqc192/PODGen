from typing import Any, Dict

import hydra
import math
import numpy as np
import omegaconf
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter
from tqdm import tqdm
from torch.autograd import Function
import torch.special

from CFtorch.common.utils import PROJECT_ROOT
from CFtorch.common.group_utils import dof0_table, wmax_table, fc_mask_table
from CFtorch.pl_modules.attention import CrystalFormer_lays


mask = [1, 1, 1, 1, 1, 1] * 2 +\
           [1, 1, 1, 0, 1, 0] * 13+\
           [1, 1, 1, 0, 0, 0] * 59+\
           [1, 0, 1, 0, 0, 0] * 68+\
           [1, 0, 1, 0, 0, 0] * 52+\
           [1, 0, 0, 0, 0, 0] * 36
lattice_mask_table_tensor = torch.tensor(mask, dtype=torch.bool).reshape(230, 6)
wmax_table_tensor = torch.Tensor(wmax_table)
dof0_table_tensor = torch.tensor(dof0_table, dtype=torch.bool)
fc_mask_table_tensor = torch.tensor(fc_mask_table, dtype=torch.bool)


def truncated_normal_(tensor, mean=0, std=1.0, trunc_std=2):
    """
    Initializes a tensor with a truncated normal distribution.
    Only values within trunc_std standard deviations from the mean are kept.
    """
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < trunc_std) & (tmp > -trunc_std)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.mul_(std).add_(mean)
    return tensor


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}


class CrystalFormer(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.coord_types = 3 * self.hparams.Kx
        self.lattice_types = self.hparams.Kl + 2 * 6 * self.hparams.Kl
        self.output_size = np.max(
            np.array([self.hparams.n_atom_types + self.lattice_types, self.coord_types, self.hparams.n_wyck_types]))
        # self.dof0_table = nn.Parameter(torch.Tensor(dof0_table), requires_grad=False)

        self.g_embeddings = nn.Embedding(self.hparams.n_grou_types, self.hparams.embed_size)
        self.w_embeddings = nn.Embedding(self.hparams.n_wyck_types, self.hparams.embed_size)
        self.a_embeddings = nn.Embedding(self.hparams.n_atom_types, self.hparams.embed_size)

        self.hW_linear = nn.Linear(2 * self.hparams.embed_size + 1, self.hparams.model_size)
        self.hA_linear = nn.Linear(2 * self.hparams.embed_size, self.hparams.model_size)
        self.hX_linear = nn.Linear(2 * self.hparams.Nf + self.hparams.embed_size, self.hparams.model_size)
        self.hY_linear = nn.Linear(2 * self.hparams.Nf + self.hparams.embed_size, self.hparams.model_size)
        self.hZ_linear = nn.Linear(2 * self.hparams.Nf + self.hparams.embed_size, self.hparams.model_size)

        mods = [nn.Linear(self.hparams.embed_size, self.hparams.model_size),
                nn.ReLU(),
                nn.Linear(self.hparams.model_size, self.hparams.n_wyck_types)]
        self.fc_firstW = nn.Sequential(*mods)

        self.positional_embeddings = nn.Parameter(torch.randn(5 * self.hparams.n_max, self.hparams.model_size))

        self.transformer = CrystalFormer_lays(n_lay=self.hparams.transformer_layers,
                                              input_dim=self.hparams.model_size,
                                              num_heads=self.hparams.num_heads,
                                              key_size=self.hparams.key_size,
                                              value_size=self.hparams.value_size,
                                              widering_facter=self.hparams.widering_facter,
                                              dropout_rate=self.hparams.dropout_rate, )

        self.layer_norm = nn.LayerNorm([self.hparams.model_size])
        self.out_linear = nn.Linear(self.hparams.model_size, self.output_size)

        self.apply(self._initialize_weights)
        # print('he')

    def _initialize_weights(self, module):
        if isinstance(module, nn.Embedding):
            truncated_normal_(module.weight.data, mean=0, std=0.01, trunc_std=2)
        elif isinstance(module, nn.Linear):
            truncated_normal_(module.weight.data, mean=0, std=0.01, trunc_std=2)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Parameter):
            truncated_normal_(module.data, mean=0, std=0.01, trunc_std=2)
        elif isinstance(module, nn.Sequential):
            for mod in module:
                if isinstance(mod, nn.Linear):
                    truncated_normal_(mod.weight.data, mean=0, std=0.01, trunc_std=2)
                    if mod.bias is not None:
                        nn.init.constant_(mod.bias, 0)
        # print('he')


    def forward(self, data):
        wmax_table_tensor = torch.Tensor(wmax_table)
        dof0_table_tensor = torch.tensor(dof0_table, dtype=torch.bool)
        wmax_table_tensor = wmax_table_tensor.to(self.device)
        dof0_table_tensor = dof0_table_tensor.to(self.device)
        n_max = data['wyckoff'].shape[1]
        batch_size = data['G'].shape[0]
        w_max = wmax_table_tensor[data['G'] - 1].unsqueeze(-1).expand(batch_size, 1, self.hparams.n_wyck_types)                   # [batch_size, 1, n_wyck_types]
        g_embeddings = self.g_embeddings(data['G'] - 1)                                         # [batch_size, 1, embed_size]
        w_embeddings = self.w_embeddings(data['wyckoff'])                                   # [batch_size, n_max, embed_size]
        a_embeddings = self.a_embeddings(data['atom_type'])                                 # [batch_size, n_max, embed_size]

        w_logit = self.fc_firstW(g_embeddings)                                              # [batch_size, 1, n_wyck_types]

        # (1) the first atom should not be the pad atom
        # (2) mask out unavaiable position for the given spacegroup
        range_tensor = torch.arange(self.hparams.n_wyck_types, device=self.device).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, self.hparams.n_wyck_types)
        w_mask = torch.logical_and(range_tensor > 0, range_tensor <= w_max)
        w_logit = torch.where(w_mask, w_logit, w_logit - 1e10)
        w_logit = w_logit - torch.special.logsumexp(w_logit, dim=2, keepdim=True)

        h0 = torch.cat((
                        w_logit,
                        torch.zeros((batch_size, 1, self.output_size - self.hparams.n_wyck_types), device=self.device),
                        ), dim=-1)                                                           # [batch_size, 1, output_size]

        if n_max == 0:
            return h0

        hW = torch.cat((
            g_embeddings.expand(batch_size, w_embeddings.size()[1], g_embeddings.size()[2]),
            w_embeddings,
            data['M'].unsqueeze(-1)), dim=2)                                                # [batch_size, n_max, 2 * embed_size + 1]
        hW = self.hW_linear(hW)                                                             # [batch_size, n_max, model_size]

        hA = torch.cat((
            g_embeddings.expand(batch_size, a_embeddings.size()[1], g_embeddings.size()[2]),
            a_embeddings), dim=2)                                                           # [batch_size, n_max, 2 * embed_size]
        hA = self.hA_linear(hA)                                                             # [batch_size, n_max, model_size]

        hX = torch.cat((g_embeddings.expand(batch_size, data['frac_coor'].size()[1], g_embeddings.size()[2]),
                        torch.squeeze(data['FTfrac_coor'][:, :, 0, :], dim=2)), dim=2)
        hY = torch.cat((g_embeddings.expand(batch_size, data['frac_coor'].size()[1], g_embeddings.size()[2]),
                        torch.squeeze(data['FTfrac_coor'][:, :, 1, :], dim=2)), dim=2)
        hZ = torch.cat((g_embeddings.expand(batch_size, data['frac_coor'].size()[1], g_embeddings.size()[2]),
                        torch.squeeze(data['FTfrac_coor'][:, :, 2, :], dim=2)), dim=2)
        hX = self.hX_linear(hX)                                                             # [batch_size, n_max, model_size]
        hY = self.hY_linear(hY)
        hZ = self.hZ_linear(hZ)

        h = torch.stack((hW, hA, hX, hY, hZ), dim=2)                                # [batch_size, n_max, 5, model_size]
        h = torch.reshape(h, (hW.size()[0], -1, hW.size()[-1]))                      # [batch_size, 5*n_max, model_size]

        rp_position = torch.unsqueeze(self.positional_embeddings, dim=0)
        rp_position = rp_position.repeat(batch_size, 1, 1)
        h = h + rp_position[:, :h.size()[1], :]

        mask = torch.tril(torch.ones(h.size()[1], h.size()[1], device=self.device))         # [5*n_max, 5*n_max]
        h = self.transformer(h, mask)                                                       # [batch_size, 5*n_max, model_size]
        h = self.layer_norm(h)
        h = self.out_linear(h)                                                              # [batch_size, 5*n_max, output_size]

        h = h.reshape(batch_size, hW.size()[1], 5, -1)
        h_al, h_x, h_y, h_z, w_logit = h[:, :, 0, :], h[:, :, 1, :], h[:, :, 2, :], h[:, :, 3, :], h[:, :, 4, :]    # [batch_size, n_max, output_size] * 5

        h_x = self.renormalize(h_x)                                                         # [batch_size, n_max, output_size]
        h_y = self.renormalize(h_y)
        h_z = self.renormalize(h_z)

        a_logit = h_al[:, :, :self.hparams.n_atom_types]
        w_logit = w_logit[:, :, :self.hparams.n_wyck_types]


        '''Start Mask'''
        # (1) impose the constrain that W_0 <= W_1 <= W_2
        # while for Wyckoff points with zero dof it is even stronger W_0 < W_1
        range_tensor = torch.arange(1, self.hparams.n_wyck_types,                           # [batch_size, n_max, n_wyck_types - 1]
                                    device=self.device).view(1, self.hparams.n_wyck_types - 1).unsqueeze(1).expand(batch_size, w_logit.size()[1], self.hparams.n_wyck_types-1)
        W = data['wyckoff'].unsqueeze(2).expand(batch_size, w_logit.size()[1], self.hparams.n_wyck_types-1)
        w_mask_less_equal = range_tensor < W                                                # [batch_size, n_max, n_wyck_types - 1]
        w_mask_less = range_tensor <= W
        dof = dof0_table_tensor[data['G'].expand(batch_size, w_logit.size()[1]) - 1, data['wyckoff']].unsqueeze(2).expand(batch_size, w_logit.size()[1], self.hparams.n_wyck_types-1)
        w_mask = torch.where(dof, w_mask_less, w_mask_less_equal)                           # [batch_size, n_max, n_wyck_types - 1]

        w_mask = torch.cat([torch.zeros((batch_size, w_mask.size()[1], 1),device=self.device, dtype=torch.bool),
                            w_mask], dim=2)                                                 # [batch_size, n_max, n_wyck_types]
        w_logit_adjustment = torch.where(w_mask, torch.tensor(1e10, device=self.device),
                                         torch.tensor(0.0, device=self.device))
        w_logit = w_logit - w_logit_adjustment
        logsumexp_result = torch.logsumexp(w_logit, dim=2, keepdim=True)
        logsumexp_result_broadcasted = logsumexp_result.expand_as(w_logit)
        w_logit = w_logit - logsumexp_result_broadcasted

        # (2) # enhance the probability of pad atoms if there is already a type 0 atom
        W = data['wyckoff']
        w_mask = torch.where(W == 0, torch.ones_like(W, device=self.device), torch.zeros_like(W, device=self.device))
        w_mask = w_mask.unsqueeze(2)
        w_mask = torch.cat([w_mask, torch.zeros((w_mask.size()[0], w_mask.size()[1], self.hparams.n_wyck_types-1), device=self.device)], dim=2)
        w_mask = w_mask.bool()                                                              # [batch_size, n_max, n_wyck_types]

        w_logit_adjustment = torch.where(w_mask, torch.tensor(1e10, device=self.device),
                                         torch.tensor(0.0, device=self.device))
        w_logit = w_logit + w_logit_adjustment
        logsumexp_result = torch.logsumexp(w_logit, dim=2, keepdim=True)
        logsumexp_result_broadcasted = logsumexp_result.expand_as(w_logit)
        w_logit = w_logit - logsumexp_result_broadcasted

        # (3) mask out unavaiable position after w_max for the given spacegroup
        range_tensor = torch.arange(self.hparams.n_wyck_types,                              # [batch_size, n_max, n_wyck_types]
                                    device=self.device).view(1, self.hparams.n_wyck_types).unsqueeze(1).expand(
                                    batch_size, w_logit.size()[1], self.hparams.n_wyck_types)
        w_mask = range_tensor <= w_max.expand(batch_size, w_logit.size()[1], self.hparams.n_wyck_types)

        w_logit_adjustment = torch.where(w_mask, torch.tensor(0.0, device=self.device),
                                         torch.tensor(1e10, device=self.device))
        w_logit = w_logit - w_logit_adjustment
        logsumexp_result = torch.logsumexp(w_logit, dim=2, keepdim=True)
        logsumexp_result_broadcasted = logsumexp_result.expand_as(w_logit)
        w_logit = w_logit - logsumexp_result_broadcasted

        # (4) if w !=0 the mask out the pad atom, otherwise mask out true atoms
        a_mask1 = W > 0
        a_mask2 = W == 0
        a_mask1 = a_mask1.unsqueeze(2)
        a_mask2 = a_mask2.unsqueeze(2).expand(batch_size, a_logit.size()[1], a_logit.size()[2]-1)
        a_mask = torch.cat([a_mask1, a_mask2], dim=2)

        a_logit_adjustment = torch.where(a_mask, torch.tensor(1e10, device=self.device),
                                         torch.tensor(0.0, device=self.device))
        a_logit = a_logit - a_logit_adjustment
        logsumexp_result = torch.logsumexp(a_logit, dim=2, keepdim=True)
        logsumexp_result_broadcasted = logsumexp_result.expand_as(a_logit)
        a_logit = a_logit - logsumexp_result_broadcasted

        w_logit = F.pad(w_logit, (0, self.output_size - w_logit.size()[-1]), "constant", 0)     # [batch_size, n_max, output_size]


        # now move on to lattice part
        l_logit, mu, sigma = torch.split(
            h_al[:, :, self.hparams.n_atom_types:self.hparams.n_atom_types + self.lattice_types],
            [self.hparams.Kl, 6 * self.hparams.Kl, 6 * self.hparams.Kl], dim=-1)
        # normalization
        logsumexp_result = torch.logsumexp(l_logit, dim=2, keepdim=True)
        logsumexp_result_broadcasted = logsumexp_result.expand_as(l_logit)
        l_logit = l_logit - logsumexp_result_broadcasted
        # ensure positivity
        sigma = F.softplus(sigma) + self.hparams.sigmamin
        h_l = torch.cat([l_logit, mu, sigma], dim=-1)

        h_al = torch.cat([a_logit, h_l], dim=-1)
        h_al = F.pad(h_al, (0, self.output_size - h_al.size()[-1]), "constant", 0)              # [batch_size, n_max, output_size]

        h = torch.stack([h_al, h_x, h_y, h_z, w_logit], dim=2)                          # [batch_size, n_max, 5, output_size]
        h = h.reshape((batch_size, -1, self.output_size))
        h = torch.cat([h0, h], dim=1)                                                   # [batch_size, 5*n_max+1, output_size]

        return h


    def compute_stats(self, data, outputs, prefix):
        lattice_mask_table_tensor = torch.tensor(mask, dtype=torch.bool).reshape(230, 6)
        fc_mask_table_tensor = torch.tensor(fc_mask_table, dtype=torch.bool)
        fc_mask_table_tensor = fc_mask_table_tensor.to(self.device)
        lattice_mask_table_tensor = lattice_mask_table_tensor.to(self.device)
        batch_size = data['G'].shape[0]
        num_sites = torch.sum(data['atom_type']!=0, dim=1, keepdim=True)

        w_logit = outputs[:, 0::5, :self.hparams.n_wyck_types]  # (batch_size, n_max+1, wyck_types)
        w_logit = w_logit[:, :-1, :]  # (batch_size, n_max, wyck_types)
        a_logit = outputs[:, 1::5, :self.hparams.n_atom_types]
        h_x = outputs[:, 2::5, :self.coord_types]
        h_y = outputs[:, 3::5, :self.coord_types]
        h_z = outputs[:, 4::5, :self.coord_types]

        logp_w = torch.gather(w_logit, 2, data['wyckoff'].unsqueeze(-1)).squeeze(-1)
        logp_a = torch.gather(a_logit, 2, data['atom_type'].unsqueeze(-1)).squeeze(-1)

        fc_mask1 = fc_mask_table_tensor[data['G'].expand(batch_size, w_logit.size()[1]) - 1, data['wyckoff']]
        fc_mask2 = data['wyckoff'] > 0
        fc_mask2 = fc_mask2.unsqueeze(-1).expand(batch_size, fc_mask2.size()[1], 3)
        fc_mask = torch.logical_and(fc_mask1, fc_mask2)

        logp_x = self.compute_logp_x(h_x, data['frac_coor'][:, :, 0], fc_mask[:, :, 0])
        logp_y = self.compute_logp_x(h_y, data['frac_coor'][:, :, 1], fc_mask[:, :, 1])
        logp_z = self.compute_logp_x(h_z, data['frac_coor'][:, :, 2], fc_mask[:, :, 2])
        logp_xyz = logp_x + logp_y + logp_z

        h_al = outputs[:, 1::5, :]
        indices = num_sites.unsqueeze(-1).expand(-1, -1, h_al.shape[2])
        h_al = torch.gather(h_al, 1,indices)
        lattice_mask = torch.gather(lattice_mask_table_tensor, 0, data['G'].repeat(1, 6)-1)
        l_logit, mu, sigma = torch.split(
            h_al[:, :, self.hparams.n_atom_types:self.hparams.n_atom_types + self.lattice_types],
            [self.hparams.Kl, 6 * self.hparams.Kl, 6 * self.hparams.Kl], dim=-1)

        lattice = data['lattice'].unsqueeze(1).repeat(1, 1, self.hparams.Kl, 1)
        lattice_mask = lattice_mask.unsqueeze(1)
        mu = mu.view(mu.size()[0], mu.size()[1], self.hparams.Kl, 6)                           # (batch_size, 1, kl, 6)
        sigma = sigma.view(sigma.size()[0], sigma.size()[1], self.hparams.Kl, 6)

        logp_l = -(lattice - mu) ** 2 / (2 * sigma ** 2) - torch.log(sigma) - 0.5 * math.log(2 * math.pi)
        l_logit = l_logit.unsqueeze(-1).repeat(1, 1, 1, 6)
        logp_l = torch.logsumexp(l_logit + logp_l, dim=-2) # (batch_size, 1, 6)
        logp_l = torch.sum(torch.where(lattice_mask, logp_l, torch.zeros_like(logp_l)), dim=-1)

        logp_a = torch.sum(logp_a, dim=-1)
        logp_w = torch.sum(logp_w, dim=-1)
        logp_a = torch.mean(logp_a)
        logp_w = torch.mean(logp_w)
        logp_xyz = torch.mean(logp_xyz)
        logp_l = torch.mean(logp_l)

        loss_a = -logp_a
        loss_xyz = -logp_xyz
        loss_w = -logp_w
        loss_l = -logp_l

        loss = self.hparams.lamb_a * loss_a + self.hparams.lamb_w * loss_w + self.hparams.lamb_l * loss_l + loss_xyz

        all_n_sites = torch.sum(num_sites)
        wyckoff = data['wyckoff']
        atom_type = data['atom_type']
        pre_w = w_logit.argmax(dim=-1)
        pre_a = a_logit.argmax(dim=-1)
        deta_a = torch.where(torch.abs(pre_a - atom_type) < 0.1, torch.zeros_like(pre_a, device=self.device),
                             torch.ones_like(pre_a, device=self.device))
        deta_w = torch.where(torch.abs(pre_w - wyckoff) < 0.1, torch.zeros_like(pre_a, device=self.device),
                             torch.ones_like(pre_a, device=self.device))
        deta_a = torch.sum(deta_a)
        deta_w = torch.sum(deta_w)
        acc_a = 1 - deta_a / all_n_sites
        acc_w = 1 - deta_w / all_n_sites

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_loss_a': -logp_a,
            f'{prefix}_loss_w': -logp_w,
            f'{prefix}_loss_l': -logp_l,
            f'{prefix}_loss_xyz': -logp_xyz,
            f'{prefix}_acc_a': acc_a,
            f'{prefix}_acc_w': acc_w,
        }

        return log_dict, loss


    def compute_logp(self, data, outputs, temperature=1.0, without_xyzl=False):
        lattice_mask_table_tensor = torch.tensor(mask, dtype=torch.bool).reshape(230, 6)
        fc_mask_table_tensor = torch.tensor(fc_mask_table, dtype=torch.bool)
        fc_mask_table_tensor = fc_mask_table_tensor.to(self.device)
        lattice_mask_table_tensor = lattice_mask_table_tensor.to(self.device)
        batch_size = data['G'].shape[0]
        num_sites = torch.sum(data['atom_type'] != 0, dim=1, keepdim=True)

        w_logit = outputs[:, 0::5, :self.hparams.n_wyck_types]  # (batch_size, n_max+1, wyck_types)
        w_logit = w_logit[:, :-1, :]  # (batch_size, n_max, wyck_types)
        a_logit = outputs[:, 1::5, :self.hparams.n_atom_types]
        h_x = outputs[:, 2::5, :self.coord_types]
        h_y = outputs[:, 3::5, :self.coord_types]
        h_z = outputs[:, 4::5, :self.coord_types]


        logits = w_logit / temperature
        w_logit = F.log_softmax(logits, dim=-1)
        logits = a_logit / temperature
        a_logit = F.log_softmax(logits, dim=-1)


        logp_w = torch.gather(w_logit, 2, data['wyckoff'].unsqueeze(-1)).squeeze(-1)
        logp_a = torch.gather(a_logit, 2, data['atom_type'].unsqueeze(-1)).squeeze(-1)

        fc_mask1 = fc_mask_table_tensor[data['G'].expand(batch_size, w_logit.size()[1]) - 1, data['wyckoff']]
        fc_mask2 = data['wyckoff'] > 0
        fc_mask2 = fc_mask2.unsqueeze(-1).expand(batch_size, fc_mask2.size()[1], 3)
        fc_mask = torch.logical_and(fc_mask1, fc_mask2)

        logp_x = self.compute_logp_x(h_x, data['frac_coor'][:, :, 0], fc_mask[:, :, 0], temperature)
        logp_y = self.compute_logp_x(h_y, data['frac_coor'][:, :, 1], fc_mask[:, :, 1], temperature)
        logp_z = self.compute_logp_x(h_z, data['frac_coor'][:, :, 2], fc_mask[:, :, 2], temperature)
        logp_xyz = logp_x + logp_y + logp_z

        h_al = outputs[:, 1::5, :]
        indices = num_sites.unsqueeze(-1).expand(-1, -1, h_al.shape[2])
        h_al = torch.gather(h_al, 1, indices)
        lattice_mask = torch.gather(lattice_mask_table_tensor, 0, data['G'].long().repeat(1, 6) - 1)
        l_logit, mu, sigma = torch.split(
            h_al[:, :, self.hparams.n_atom_types:self.hparams.n_atom_types + self.lattice_types],
            [self.hparams.Kl, 6 * self.hparams.Kl, 6 * self.hparams.Kl], dim=-1)

        logits = l_logit / temperature
        l_logit = F.log_softmax(logits, dim=-1)

        lattice = data['lattice'].unsqueeze(1).repeat(1, 1, self.hparams.Kl, 1)
        lattice_mask = lattice_mask.unsqueeze(1)
        mu = mu.view(mu.size()[0], mu.size()[1], self.hparams.Kl, 6)                           # (batch_size, 1, kl, 6)
        sigma = sigma.view(sigma.size()[0], sigma.size()[1], self.hparams.Kl, 6)

        sigma = sigma * temperature

        # print(f'shape of lattice {lattice.shape}, shape of mu {mu.shape}, shape of sigme {sigma.shape}')
        logp_l = -(lattice - mu) ** 2 / (2 * sigma ** 2) - torch.log(sigma) - 0.5 * math.log(2 * math.pi)
        l_logit = l_logit.unsqueeze(-1).repeat(1, 1, 1, 6)
        logp_l = torch.logsumexp(l_logit + logp_l, dim=-2) # (batch_size, 1, 6)
        logp_l = torch.sum(torch.where(lattice_mask, logp_l, torch.zeros_like(logp_l)), dim=-1)

        logp_a = torch.sum(logp_a, dim=-1)
        logp_w = torch.sum(logp_w, dim=-1)
        logp_l = torch.sum(logp_l, dim=-1)

        if without_xyzl:
            logp = logp_w + logp_a
        else:
            logp = logp_w + logp_l + logp_a + logp_xyz

        return logp

    def renormalize(self, h_x):
        x_logit, x_loc, x_kappa = torch.split(h_x[:, :, :self.coord_types], [self.hparams.Kx, self.hparams.Kx, self.hparams.Kx], dim=-1)
        logsumexp_result = torch.logsumexp(x_logit, dim=2, keepdim=True)
        logsumexp_result_broadcasted = logsumexp_result.expand_as(x_logit)
        x_logit = x_logit - logsumexp_result_broadcasted
        x_kappa = F.softplus(x_kappa)

        h_x = torch.cat([x_logit,x_loc,x_kappa],dim=-1)
        h_x = F.pad(h_x, (0, self.output_size - h_x.size()[-1]), "constant", 0)
        return h_x

    def compute_logp_x(self, h_x, X, fc_mask_x, temperature=1.0):
        batch_size = h_x.size()[0]
        x_logit, loc, kappa = torch.split(h_x, [self.hparams.Kx, self.hparams.Kx, self.hparams.Kx], dim=-1)

        loc = loc.view(batch_size, self.hparams.n_max, self.hparams.Kx)
        kappa = kappa.view(batch_size, self.hparams.n_max, self.hparams.Kx)

        logits = x_logit / temperature
        x_logit = F.log_softmax(logits, dim=-1)
        kappa = kappa / temperature

        logp_x = self.von_mises_logpdf((X - 0.5) * 2 * torch.tensor(math.pi, device=self.device), loc, kappa)
        logp_x = torch.logsumexp(x_logit + logp_x, dim=-1)  # (n_max,)

        logp_x = torch.where(fc_mask_x, logp_x, torch.zeros_like(logp_x, device=self.device))
        logp_x = torch.sum(logp_x,dim=-1)
        return logp_x

    def von_mises_logpdf(self, x, loc, concentration):
        '''
        Compute log-pdf of von Mises distribution:
        x, loc: shape [batch, n, K]
        concentration: shape [batch, n, K] or broadcastable

        Returns:
            log_pdf: shape [batch, n, K]
        '''
        i0e = torch.special.i0e(concentration)
        log_i0 = torch.log(i0e) + concentration  # i0e = I0(kappa) / exp(kappa)

        x = x.unsqueeze(-1).repeat(1, 1, loc.size(-1))  # make shape match loc
        log_pdf = -math.log(2 * math.pi) - log_i0 + concentration * torch.cos(x - loc)

        return log_pdf


    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='train')
        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='val')
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='test')
        self.log_dict(
            log_dict,
        )
        return loss




@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
    datamodule.setup('fit')
    train_loader = datamodule.train_dataloader()
    for i, batch in enumerate(train_loader):
        print(i)
        outputs = model(batch)
        log_dict, loss = model.compute_stats(batch, outputs, prefix='train')
    return model


if __name__ == "__main__":
    lattice_mask_table_tensor = lattice_mask_table_tensor.to('cpu')
    wmax_table_tensor = wmax_table_tensor.to('cpu')
    dof0_table_tensor = dof0_table_tensor.to('cpu')
    fc_mask_table_tensor = fc_mask_table_tensor.to('cpu')
    main()




