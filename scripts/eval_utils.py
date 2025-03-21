import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(parent_dir)
import itertools
import numpy as np
import torch
import hydra

from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
from hydra.experimental import compose
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_dir
from pathlib import Path
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import torch.distributions as dist

from CFtorch.common.utils import PROJECT_ROOT
from CFtorch.pl_modules.model import CrystalFormer
from CFtorch.common.group_utils import symops

def load_model(model_path, model_file=None, load_data=False, testing=True):
    GlobalHydra.instance().clear()  # 清除之前的初始化
    with initialize_config_dir(str(model_path)):
        cfg = compose(config_name='hparams')
        ckpts = list(model_path.glob('*.ckpt'))
        if len(ckpts) > 0:
            ckpt_epochs = np.array(
                [int(ckpt.parts[-1].split('-')[0].split('=')[1]) for ckpt in ckpts])
            ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
        if model_file != None:
            ckpt = os.path.join(model_path, model_file)
        model = CrystalFormer.load_from_checkpoint(ckpt, map_location="cpu")
        print('Loaded checkpoint from {}'.format(ckpt))

        if load_data:
            print('data information', cfg.data.root_path)
            datamodule = hydra.utils.instantiate(
                cfg.data.datamodule, _recursive_=False, scaler_path=model_path
            )
            if testing:
                datamodule.setup('test')
                test_loader = datamodule.test_dataloader()[0]
            else:
                datamodule.setup()
                test_loader = datamodule.val_dataloader()[0]
        else:
            test_loader = None

    return model, test_loader, cfg


def top_p_sampling(w_logit, p, temperature=1.0):
    # 先将 logits 除以 temperature
    logits = w_logit / temperature

    # 转换 logits 为概率分布
    probs = F.softmax(logits, dim=-1)

    # 对概率进行排序并获取排序后的索引
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

    # 计算累积概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 创建一个 mask，标记累积概率大于 p 的部分
    mask = cumulative_probs > p
    mask[..., 1:] = mask[..., :-1].clone()  # 右移一位，保证至少有一个元素保留
    mask[..., 0] = 0  # 第一个元素保留

    # 将被 mask 掉的概率设为 0
    sorted_probs[mask] = 0.0

    # 重新归一化概率
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    # 从排序的概率中进行采样
    sampled_indices = torch.multinomial(sorted_probs, num_samples=1)

    # 获取对应的原始索引
    final_indices = sorted_indices.gather(-1, sampled_indices)

    return final_indices


def sample_x(h_x, Kx, top_p, temperature, batchsize):
    coord_types = 3 * Kx
    x_logit, loc, kappa = torch.split(h_x, [Kx, Kx, Kx], dim=-1)
    kappa = F.softplus(kappa)
    k = top_p_sampling(x_logit, top_p, temperature)
    loc = loc.view(batchsize, Kx)[torch.arange(batchsize), k.squeeze(1)]
    kappa = kappa.view(batchsize, Kx)[torch.arange(batchsize), k.squeeze(1)]
    x = sample_von_mises(loc, kappa / temperature, (1,)).unsqueeze(1)
    x = (x + torch.tensor(np.pi)) / (2.0 * torch.tensor(np.pi))  # wrap into [0, 1]
    return x


def sample_von_mises(mean, kappa, shape):
    # 确保 mean 和 kappa 具有相同的形状
    assert mean.shape == kappa.shape # "Mean and kappa must have the same shape"

    # 获取张量的形状
    shape = mean.shape
    # 将 mean 和 kappa 展平成 1D 张量
    mean_flat = mean.to(torch.float32).view(-1)
    kappa_flat = kappa.to(torch.float32).view(-1)

    # 创建 VonMises 分布对象
    vm = dist.von_mises.VonMises(mean_flat, kappa_flat)

    # 从 VonMises 分布中采样
    samples_flat = vm.sample()

    # 将采样结果重新整形为原始形状
    samples = samples_flat.view(shape)

    return samples


def project_xyz(g, w, x, idx):
    '''
    apply the randomly sampled Wyckoff symmetry op to sampled fc, which
    should be (or close to) the first WP
    '''
    op = symops[g.cpu()-1, w.cpu(), idx].reshape(g.size()[0],3, 4)
    affine_point = torch.cat([x, torch.ones(x.size()[0],1).to(device=x.device)], dim=-1)  # (4, )
    # a = torch.tensor(op, dtype=torch.double)
    # b = affine_point.unsqueeze(2).to(torch.double)
    a = torch.tensor(op).to(torch.float).to(device=x.device)
    b = affine_point.unsqueeze(2)
    x = torch.matmul(a, b).squeeze(2)#[:-1]  # (3, )
    x -= torch.floor(x)
    return x

def project_xyz2(g, w, x, idx):
    '''
    apply the randomly sampled Wyckoff symmetry op to sampled fc, which
    should be (or close to) the first WP
    '''

    g_indices = g.expand(-1, x.size()[1])-1  # 扩展 g 形状以匹配 (10, 21)
    w_indices = w
    op = symops[g_indices.to('cpu').numpy(), w_indices.to('cpu').numpy(), idx]

    # op = symops[g.cpu()-1, w.cpu(), idx].reshape(g.size()[0],3, 4)
    affine_point = torch.cat([x, torch.ones(x.size()[0],x.size()[1],1).to(device=x.device)], dim=-1)  # (4, )
    # a = torch.tensor(op, dtype=torch.double)
    # b = affine_point.unsqueeze(2).to(torch.double)
    a = torch.tensor(op).to(torch.float).to(device=x.device)
    b = affine_point.unsqueeze(-1)
    x = torch.matmul(a, b).squeeze(-1)#[:-1]  # (3, )
    x -= torch.floor(x)
    return x

def sample_normal(mu, sigma):
    """
    从正态分布中采样

    参数:
    mu (torch.Tensor): 均值
    sigma (torch.Tensor): 标准差
    sample_shape (torch.Size): 采样形状，默认为标量

    返回:
    torch.Tensor: 采样的值
    """
    normal_dist = torch.distributions.Normal(mu, sigma)
    samples = normal_dist.rsample()
    return samples


def symmetrize_lattice(spacegroup, lattice):
    a, b, c, alpha, beta, gamma = torch.split(lattice, [1]*6, dim=-1)

    b = torch.where(spacegroup<=74, b, a)
    c = torch.where(spacegroup<=194, c, a)
    alpha = torch.where(spacegroup<=2, alpha, torch.tensor([90.]*a.size()[0], device=a.device).unsqueeze(1))
    beta = torch.where(spacegroup<=15, beta, torch.tensor([90.]*a.size()[0], device=a.device).unsqueeze(1))
    gamma = torch.where(spacegroup <= 2, gamma, torch.tensor([90.] * a.size()[0], device=a.device).unsqueeze(1))
    gamma = torch.where(spacegroup <= 142, gamma, torch.tensor([120.] * a.size()[0], device=a.device).unsqueeze(1))
    gamma = torch.where(spacegroup <= 194, gamma, torch.tensor([90.] * a.size()[0], device=a.device).unsqueeze(1))

    L = torch.cat([a, b, c, alpha, beta, gamma], dim=1)

    return L


def data2logp(data, model):
    data_device = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    model.eval()
    h = model(data_device)
    logp = model.compute_logp(data, h)

    return logp