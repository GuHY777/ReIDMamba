from .build import LOSS_REGISTRY

from .baselosses import *
from torch import nn

import logging
logger = logging.getLogger(__name__)


class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.losses = nn.ModuleList()
        
        self.loss_names = args.loss
        self.loss_weights = [args.loss_weights[0] for _ in range(len(args.loss))] if len(args.loss_weights) == 1 else args.loss_weights
        assert len(self.loss_names) == len(self.loss_weights)
        self.loss_nums = [args.loss_nums[0] for _ in range(len(args.loss))] if len(args.loss_nums) == 1 else args.loss_nums
        assert len(self.loss_names) == len(self.loss_nums)
        
        logger.info('\n# --- Loss --- #')
        
        for loss_name, loss_num, loss_kwarg, weight in zip(self.loss_names, self.loss_nums, args.loss_kwargs, self.loss_weights):
            self.losses.append(nn.ModuleList([LOSS_REGISTRY[loss_name](**loss_kwarg) for _ in range(loss_num)]))
            logger.info('{}({}): {} x {}'.format(loss_name, weight, loss_num, loss_kwarg))
        
        if len(set(self.loss_names)) < len(self.loss_names):
            self.idx_name = True
        else:
            self.idx_name = False
        
    def forward(self, outputs, targets):
        # outputs: tuple or tuple of list
        losses = {}
        loss_val = 0
        for i, (output, loss, weight, name, num) in enumerate(zip(outputs, self.losses, self.loss_weights, self.loss_names, self.loss_nums)):
            ki = name+'-'+str(i+1) if self.idx_name else name
            
            losses_i = [loss[j](output[j], targets) for j in range(num)]
            if len(losses_i) > 1:
                kij = ki+',' if self.idx_name else ki+'-'
                for j in range(num):
                    losses[kij+str(j)] = losses_i[j].item()
                    
            losses[ki] = sum(losses_i) / num
            loss_val = loss_val + weight * losses[ki]
            losses[ki] = losses[ki].item()

        losses['total'] = loss_val.item()
        return loss_val, losses