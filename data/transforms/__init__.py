import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import logging
from .autoaugment import AutoAugment, AugMixAugment

logger = logging.getLogger(__name__)


def load_transforms(is_train, mean_=(0.485, 0.456, 0.406), std_=(0.229, 0.224, 0.225), size_train=(256, 128), size_test=(256, 128),
                    do_aa=False, aa_prob=0.1,
                    do_crop=False, crop_size=(256, 128), crop_scale=(0.08, 1.0), crop_ratio=(3./4., 4./3.),
                    do_pad=True, padding_size=(10, 10), padding_mode='constant', padding_fill=(0, 0, 0),
                    do_flip=True, flip_prob=0.5,
                    do_rea=True, rea_prob=0.5, rea_value=0.0, rea_scale=(0.02, 0.4), rea_ratio=(0.3, 3.33)
                    ):
    res = []

    if is_train:
        logger.info("Training transform:")
        
        if do_aa:
            res.append(T.RandomApply([AutoAugment()], p=aa_prob))
            logger.info("\tAutoAugment: prob={}".format(aa_prob))

        res.append(T.Resize(size=size_train, 
                            interpolation=InterpolationMode.BICUBIC))
        logger.info("\tResize: size={}".format(size_train))

        if do_crop:
            res.append(T.RandomResizedCrop(size=crop_size,
                                           interpolation=InterpolationMode.BICUBIC,
                                           scale=crop_scale, 
                                           ratio=crop_ratio))
            logger.info("\tRandomResizedCrop: size={}, scale={}, ratio={}".format(crop_size, crop_scale, crop_ratio))
            
        if do_pad:
            res.append(T.RandomCrop(size=size_train, 
                                    padding=padding_size, 
                                    pad_if_needed=True, 
                                    fill=padding_fill, 
                                    padding_mode=padding_mode))
            logger.info("\tPadding: size={}, mode={}, fill={}".format(padding_size, padding_mode, padding_fill))
            
        if do_flip:
            res.append(T.RandomHorizontalFlip(p=flip_prob))
            logger.info("\tRandomHorizontalFlip: prob={}".format(flip_prob))

        res.extend([T.ToTensor(), T.Normalize(mean=mean_, std=std_)])
        logger.info("\tToTensor, Normalize: mean={}, std={}".format(mean_, std_))
        
        if do_rea:
            res.append(T.RandomErasing(p=rea_prob, value=rea_value, scale=rea_scale, ratio=rea_ratio))
            logger.info("\tRandomErasing: prob={}, value={}, scale={}, ratio={}".format(rea_prob, rea_value, rea_scale, rea_ratio))
    
    else:
        logger.info("Test transform:")

        res.append(T.Resize(size=size_test, interpolation=InterpolationMode.BICUBIC))
        logger.info("\tResize: size={}".format(size_test))
        
        res.extend([T.ToTensor(), T.Normalize(mean=mean_, std=std_)])
        logger.info("\tToTensor, Normalize: mean={}, std={}".format(mean_, std_))
        
    return T.Compose(res)