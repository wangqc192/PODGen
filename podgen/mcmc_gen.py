# add check_convergence
# add ele control
import argparse
import yaml
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(parent_dir)
import pandas as pd
import numpy as np
import torch
import random
import time
from pathlib import Path
from pymatgen.analysis.structure_matcher import StructureMatcher

from scripts.eval_utils import load_model
from predictor.pre_utils import load_pre_model
from mcmc_utils import get_temperature_list, get_seed_crystal, data2struc, logp_of_pre, MCMC_setp, logging, sample_2_cif
from mcmc_utils import sample_list_from_file, saveinannea, loadinannea, add_ele_score
from CFtorch.common.utils import PROJECT_ROOT
from omegaconf import OmegaConf

def main(config):
    ################################ Prepare for MCMC ################################
    # set the randon seed
    if config['generate_setting']['deterministic']:
        np.random.seed(config['generate_setting']['seed'])
        random.seed(config['generate_setting']['seed'])
        torch.manual_seed(config['generate_setting']['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config['generate_setting']['seed'])

    # load the result already get
    output_file = config['generate_setting']['output_file']
    if not os.path.exists(output_file):
        os.makedirs(output_file)
    else:
        print(f'BE CAREFULL THE OUTPUT_FILE {output_file} ALREADY EXISTS !!!!!!', flush=True)
    start_i = 0
    sample_list = sample_list_from_file(output_file, config['generate_setting']['num_worker'])
    cif_name = len(sample_list)
    if os.path.exists(config['load_middle']):
        print('load middle from', config['load_middle'])
        if config['use_old_config']:
            data, start_i, config = loadinannea(config['load_middle'], config) # use the old config except random number seed, save path
        else:
            data, start_i, _ = loadinannea(config['load_middle'], config)

    # load_model
    model, _, cfg = load_model(Path(config['former_path']), config['former_file'])
    pre_models = []
    for i in range(len(config['pre_models'])):
        pre_models.append(load_pre_model(config['pre_models'][i]['model_path'], config['pre_models'][i]['model_file'])[0])
    if torch.cuda.is_available():
        model.to('cuda')
        pre_models = [p_m.to('cuda') for p_m in pre_models]
    model.eval()
    pre_models = [p_m.eval() for p_m in pre_models]
    matcher = StructureMatcher()

    # set the tempreture
    anneal_T = get_temperature_list(config['generate_setting']['max_T'],
                                    config['generate_setting']['min_T'],
                                    config['generate_setting']['annealing_num_T'],
                                    config['generate_setting']['T_type'])

    # propose strategy
    propose_strategy = [p / sum(config['generate_setting']['propose_strategy']) for p in config['generate_setting']['propose_strategy']]
    print(f'propose_strategy: {propose_strategy}')
    propose_strategy = torch.tensor(propose_strategy).float()

    # get the seed crystal
    if not os.path.exists(config['load_middle']):
        data = get_seed_crystal(data_file=config['generate_setting']['seed_dataset'],
                                num=config['generate_setting']['batch_size'],
                                atom_types=cfg.data.n_atom_types,
                                wyck_types=cfg.data.n_wyck_types,
                                n_max=cfg.data.n_max,
                                tol=cfg.data.tol,
                                Nf=cfg.data.Nf,
                                device=model.device,)
    else:
        data = {key: tensor.to(model.device) for key, tensor in data.items()}

    
    with torch.no_grad():
    ################################ Annealing ################################
        for i in range(start_i, config['generate_setting']['annealing_num_T']):
            avg_logp = []
            accept_list = []
            balance = 0
            step = 0

            use_T = anneal_T[i]
            data_h = model(data)
            struc = data2struc(data, num_io_process=config['generate_setting']['num_worker'])
            data_logp = model.compute_logp(data, data_h, temperature=use_T)
            predict_logp = logp_of_pre(struc, pre_models, config['pre_models'], graph_method=config['generate_setting']['graph_method'])
            max_avg_logp = data_logp.clone()
            for j in range(len(predict_logp)):
                max_avg_logp += config['pre_models'][j]['alpha'] * predict_logp[j]
            max_avg_logp = max_avg_logp.mean().cpu().item()
            min_avg_logp = max_avg_logp
            convergence = False

            # while balance < config['generate_setting']['annealing_tole_step']:
            while (balance < config['generate_setting']['annealing_tole_step']) or (not convergence):
            # while not convergence:
                try:
                    data, struc, data_h, data_logp, predict_logp, accept_rate = MCMC_setp(data, data_logp, data_h,
                                                                                            predict_logp, struc, model,
                                                                                            pre_models, propose_strategy,
                                                                                            use_T, config, cfg)
                except Exception as e:
                    print(f'ERR in MCMC_setp: {e}')
                    continue

                step += 1
                data_logp_ele = data_logp.clone()
                if 'ele_score' in config:
                    data_logp_ele, _ = add_ele_score(data_logp_ele, data_logp_ele, data, data, config['ele_score'])
                now_avg_logp, max_avg_logp, min_avg_logp, balance = logging(step, data_logp_ele, predict_logp, accept_rate,
                                                                       use_T, balance, max_avg_logp, min_avg_logp, config)
                avg_logp.append(now_avg_logp)
                accept_list.append(accept_rate)
                convergence = check_convergence(avg_logp, window_size=config['generate_setting']['conv_window_size'])

            # output the log
            filename = 'i=' + str(i) + '_log.csv'
            filename = os.path.join(output_file, filename)
            out_df = {'avg_logp': avg_logp, 'accept_rate': accept_list}
            out_df = pd.DataFrame(out_df)
            out_df.to_csv(filename, index=False)
            savepath = saveinannea(output_file, config, i, data, avg_logp, accept_list)
            print('middle save in', savepath, flush=True)

    ################################ Sampling ################################
        use_T = anneal_T[-1]
        sample_step = 0
        step = 17
        data_h = model(data)
        struc = data2struc(data, num_io_process=config['generate_setting']['num_worker'])
        data_logp = model.compute_logp(data, data_h, temperature=use_T)
        predict_logp = logp_of_pre(struc, pre_models, config['pre_models'], graph_method=config['generate_setting']['graph_method'])
        while sample_step < config['generate_setting']['isothermal_step']:
            try:
                data, struc, data_h, data_logp, predict_logp, accept_rate = MCMC_setp(data, data_logp, data_h,
                                                                                      predict_logp, struc, model,
                                                                                      pre_models, propose_strategy,
                                                                                      use_T, config, cfg)
            except Exception as e:
                print(f'ERR in MCMC_setp: {e}')
                continue
            data_logp_ele = data_logp.clone()
            if 'ele_score' in config:
                data_logp_ele, _ = add_ele_score(data_logp_ele, data_logp_ele, data, data, config['ele_score'])
            logging(step, data_logp_ele, predict_logp, accept_rate, use_T, None, None, None, config, 'Sampling')
            if (sample_step+1) % config['generate_setting']['sample_every_n_step'] == 0:
                current_time = time.localtime()
                formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
                print(f'sample_step: {sample_step+1}, already had {cif_name} samples. Time: {formatted_time}')
                sample_list, cif_name = sample_2_cif(config, sample_list, cif_name, struc, predict_logp, matcher)
            sample_step += 1




    print('END', flush=True)


def check_convergence(log_P, window_size=100, tolerance=1e-3):
    # 计算滚动均值
    rolling_mean = np.convolve(log_P, np.ones(window_size) / window_size, mode='valid')
    # 计算滚动标准差
    rolling_std = np.array([np.std(log_P[i:i + window_size]) for i in range(len(log_P) - window_size + 1)])

    # 检查最近窗口的均值和标准差变化是否小于阈值
    mean_diff = np.abs(rolling_mean[-1] - rolling_mean[-2]) if len(rolling_mean) > 1 else np.inf
    std_diff = np.abs(rolling_std[-1] - rolling_std[-2]) if len(rolling_std) > 1 else np.inf

    # 判断是否收敛
    if mean_diff < tolerance and std_diff < tolerance:
        return True
    return False


if __name__ == '__main__':
    ################################ Read the setting ################################
    parser = argparse.ArgumentParser(description='graph data generation')
    parser.add_argument('--config', default='/Users/yecaiyuan/0-project/0-materialgen/0-TF-torch/former1/scripts/MCMC_config_test.yaml', type=str, metavar='N')
    args = parser.parse_args()

    # with open(args.config, encoding='utf-8') as rstream:
    #     config = yaml.load(rstream, yaml.SafeLoader)

    config = OmegaConf.load(args.config)

    main(config)
