from .logger import setup_logger
from .seed import setup_seed
import os
import regex
import yaml
from collections import defaultdict
from copy import deepcopy
import random
import numpy as np
import re

from torch.utils.tensorboard import SummaryWriter
import torch


def setup(args, fileout=True, determenistic=True, benchmark=False):
    # copy the initial arguments
    old_args_dict = deepcopy(args.__dict__)
    load_cfg = ''
    if old_args_dict['config']:
        load_cfg = deepcopy(old_args_dict['config'])
        with open(old_args_dict['config'], 'r') as f:
            yaml_args = yaml.safe_load(f)
        old_args_dict['config'] = '' # avoid saving config

        yaml_args.pop('exp')
        yaml_args.pop('gpus')
        args.__dict__.update(yaml_args)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(i for i in args.gpus)
    os.environ['DATA_ROOT'] = args.dataroot
    torch.cuda.device_count.cache_clear()
    
    logger, savedir = setup_logger(args.exp, load_cfg, fileout=fileout)

    # saving the initial arguments
    with open(os.path.join(savedir, f"{args.model}-{'+'.join(args.dataset)}.yaml"), 'w') as f:
        yaml.dump(args.__dict__, f, sort_keys=False)
        
    tb_writer = SummaryWriter(os.path.join(savedir, 'tb'))
    
    logger.info(f"Random seed: {args.seed}.")
    setup_seed(args.seed, determenistic, benchmark)
    
    if args.sp_seeds:
        seeds = np.random.randint(0, 1000000, args.epochs).tolist()
    else:
        seeds = [-1] * args.epochs
    
        
    return tb_writer, logger, savedir, seeds


def str2list(arg, f=str):
    if ',' in arg:
        if f == bool:
            return map(bool, map(int, arg.split(',')))
        elif f == str2dict:
            return list(map(f, re.split(r',(?![^\[]*\])', arg)))
        return list(map(f, arg.split(',')))
    else:
        if f == bool:
            return [bool(int(arg))]
        return [f(arg)]

def split_string(s, c='|'):
    # 找到所有被 {} 包围的部分
    parts = regex.findall(r'\{(?:[^{}]|(?R))*\}', s)
    
    # 将这些部分替换为占位符
    for i, part in enumerate(parts):
        s = s.replace(part, f'{{{i}}}')
    
    # 根据 c 进行分割
    result = s.split(c)
    
    # 将占位符替换回原来的部分
    for i, item in enumerate(result):
        if '{' in item and '}' in item:
            index = int(item[item.index('{')+1 : item.index('}')])
            result[i] = item.replace(f'{{{index}}}', parts[index])
    
    return result

    
def str2dict(arg):
    #"cls_bias(bool)=1|lrs2_kwargs(dict)={milestones(int)=[30,60]|gamma(float)=0.1|tmp(dict)={a(int)=1|b(float)=2}|c(str)=123}"
    # "cls_bias(bool)=1|cls_weight(float)=1.0|tst_list(int)=[1,2,3,4]|lrs2_kwargs(dict)={milestones(int)=[30,60]|gamma(float)=0.1}"
    if arg == '':
        return {}
    else:        
        out = {}
        for sp in split_string(arg, '|'):
            k, v = split_string(sp, '=') # "cls_bias(bool)", "1"
            op = k.split('(')[1][:-1] # "bool"
            if v[0] == '[' and v[-1] == ']':
                v = str2list(v[1:-1], eval(op))
            elif v[0] == '{' and v[-1] == '}':
                v = str2dict(v[1:-1])
            else:
                if op == 'bool':
                    v = bool(int(v))
                else:
                    v = eval(op)(v)
            out[k.split('(')[0]] = v
    return out


class AverageMeter:
    def __init__(self):
        self.avgs = defaultdict(float)
        self.i = 0
        
    def __call__(self, losses):
        for k,v in losses.items():
            self.avgs[k] = (self.avgs[k] * self.i + v) / (self.i + 1)
        self.i += 1
            
    def __repr__(self) -> str:
        ss = ''
        for k,v in self.avgs.items():
            ss += f'{k}={v:.4e} '
        return ss
    