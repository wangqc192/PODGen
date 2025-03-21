
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(parent_dir)
from pathlib import Path
import torch
import hydra
import omegaconf
import numpy as np
import random
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from utils import PROJECT_ROOT, param_statistics
from train import train


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="pre_default")
def main(cfg: omegaconf.DictConfig):
    if cfg.accelerator == 'DDP' or cfg.accelerator == 'ddp':
        torch.distributed.init_process_group(backend='nccl')
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)

    if cfg.train.deterministic:
        np.random.seed(cfg.train.random_seed)
        random.seed(cfg.train.random_seed)
        torch.manual_seed(cfg.train.random_seed)
        # torch.backends.cudnn.deterministic = True
        if(cfg.accelerator != 'cpu'):
            torch.cuda.manual_seed(cfg.train.random_seed)
            torch.cuda.manual_seed_all(cfg.train.random_seed)

    # Hydra run directory
    hydra_dir = Path(HydraConfig.get().run.dir)

    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
    datamodule.setup()



    # Instantiate model
    hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)
    param_statistics(model)
    model.scaler = datamodule.scaler.copy()
    model.scaler.save_to_file("prop_scaler.json")
    torch.save(datamodule.scaler, hydra_dir / 'prop_scaler.pt')


    if cfg.accelerator == 'DDP' or cfg.accelerator == 'ddp':
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif cfg.accelerator == 'gpu':
        model.cuda()

    # Store the YaML config separately into the wandb dir
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (hydra_dir / "hparams.yaml").write_text(yaml_conf)

    optimizer = hydra.utils.instantiate(
        cfg.optim.optimizer, params=model.parameters(), _convert_="partial"
    )
    scheduler = hydra.utils.instantiate(
        cfg.optim.lr_scheduler, optimizer=optimizer
    )

    hydra.utils.log.info('Start Train')
    train(cfg, model, datamodule, optimizer, scheduler, hydra_dir, best_loss_old=None)
    hydra.utils.log.info('END')

    return 0



if __name__ == "__main__":
    main()