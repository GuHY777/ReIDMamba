from torch import optim
from .lr_schedulers import *
import logging
from timm.scheduler import create_scheduler_v2

logger = logging.getLogger(__name__)



def build_optimizer(args, model):
    logger.info('\n# --- Optimizer --- #')
    if args.optim =='sgd':
        optimizer = optim.SGD(model.get_params(args), **args.optim_kwargs)
    elif args.optim == 'adam':
        optimizer = optim.Adam(model.get_params(args), **args.optim_kwargs)
    elif args.optim == 'adamw':
        optimizer = optim.AdamW(model.get_params(args), **args.optim_kwargs)
    else:
        raise ValueError('Unsupported optimizer: {}'.format(args.optim))
    logger.info(f'Optimizer={args.optim}')
    logger.info(f'\tKwargs={args.optim_kwargs}\n')

    logger.info('\n# --- Learning rate scheduler --- #')
    if args.lr_scheduler == 'LinearWarmupLrScheduler':
        lr_scheduler = LinearWarmupLrScheduler(optimizer, **args.lr_scheduler_kwargs)
    elif args.lr_scheduler == 'TimmScheduler':
        lr_scheduler, _ = create_scheduler_v2(optimizer, **args.lr_scheduler_kwargs)
    else:
        raise ValueError('Unsupported lr_scheduler: {}'.format(args.lr_scheduler))
    logger.info(f'Scheduler={args.lr_scheduler}\n\tKwargs={args.lr_scheduler_kwargs}')
    
    return optimizer, lr_scheduler
    