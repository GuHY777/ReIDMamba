from torch.optim.lr_scheduler import *

__all__ = ['LinearWarmupLrScheduler']

def LinearWarmupLrScheduler(optimizer, warmup_epochs=1, lrs2=None, lrs2_kwargs={}, lr_multiplier=1.0, delay_epochs=0, warmup_iters=0):
    
    if isinstance(lrs2, str):
        if lrs2 == 'MultiStepLR':
            lrs2 = MultiStepLR(optimizer, **lrs2_kwargs)
        elif lrs2 == 'CosineAnnealingLR':
            lrs2 = CosineAnnealingLR(optimizer, **lrs2_kwargs)
        elif lrs2 == 'StepLR':
            lrs2 = StepLR(optimizer, **lrs2_kwargs)
        elif lrs2 == 'ConstantLR':
            lrs2 = ConstantLR(optimizer, **lrs2_kwargs)
        else:
            raise ValueError('Unsupported lr scheduler: {}'.format(lrs2))
    
    if warmup_epochs == 0:
        return lrs2
        
    if warmup_iters:
        warmup_epochs = warmup_iters
    
    lrs1 = LinearLR(optimizer, lr_multiplier, total_iters=warmup_epochs)
    lrs = SequentialLR(optimizer, [lrs1, lrs2], milestones=[warmup_epochs+delay_epochs, ])
    
    return lrs

