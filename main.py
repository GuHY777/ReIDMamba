import yaml
import argparse
from utils import setup, str2list, str2dict
from functools import partial

from data import build_dataloaders
from model import build_model
from losses import Loss
from optims import build_optimizer
from evaluation import Evaluator
from engine import Engine
from timm.utils import ModelEma

from torch.utils import tensorboard
from torch.nn import DataParallel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # system
    parser.add_argument('--gpus', type=str2list, default='1', help='gpu ids (default: 0)')
    parser.add_argument('--exp', type=str, default='market1501_384', help='experiment name')
    parser.add_argument('--seed', type=int, default=777, help='random seed (default: 777)')
    parser.add_argument('--config', type=str, default='', help='path to config file')
    
    # data
    parser.add_argument('--dataroot', type=str, default='/root/data/DataSets/PersonReID_DataSets', help='path to dataset')
    parser.add_argument('--dataset', type=str2list, default='Market1501', help='dataset name (default: market1501)')
    parser.add_argument('--p_trn', type=float, default=1.0, help='percentage of training set (default: 0.8)')
    parser.add_argument('--split_mode_trn', type=str, default='by_person', help='split mode for training set (default: by_person)')
    parser.add_argument('--dataset_trn', type=str2list, default='', help='dataset name for training')
    parser.add_argument('--dataset_qry', type=str2list, default='', help='dataset name for query')
    parser.add_argument('--dataset_gal', type=str2list, default='', help='dataset name for gallery')
    parser.add_argument('--pixel_mean', type=partial(str2list, f=float), default='0.485,0.456,0.406', help='mean of dataset (default: 0.485,0.456,0.406)')
    parser.add_argument('--pixel_std', type=partial(str2list, f=float), default='0.229,0.224,0.225', help='std of dataset (default: 0.229,0.224,0.225)')
    parser.add_argument('--img_size', type=partial(str2list, f=int), default='384,128', help='size of input image (default: 256,128)')
    parser.add_argument('--num_workers', type=int, default=10, help='number of data loading workers (default: 4)')
    parser.add_argument('--bs_trn', type=partial(str2list, f=int), default='16,4', help='batch size for training (default: 16,4)')
    parser.add_argument('--bs_tst', type=partial(str2list, f=int), default='128', help='batch size for query (default: 16,4)')
    parser.add_argument('--sp_forever', type=bool, default=True, help='sampling forever (default: False)')
    parser.add_argument('--sp_seeds', type=bool, default=False, help='sampling seeds (default: False)')
    parser.add_argument('--aa_tf', type=str2dict, default='do_aa(bool)=0|aa_prob(float)=0.1')
    parser.add_argument('--crop_tf', type=str2dict, default='do_crop(bool)=0|crop_size(int)=[384,128]|crop_scale(float)=[0.08,1.0]|crop_ratio(float)=[0.75, 1.33]')
    parser.add_argument('--pad_tf', type=str2dict, default='do_pad(bool)=1|padding_size(int)=[10,10]|padding_mode(str)=constant|padding_fill(float)=[0,0,0]')
    parser.add_argument('--flip_tf', type=str2dict, default='do_flip(bool)=1|flip_prob(float)=0.5')
    parser.add_argument('--rea_tf', type=str2dict, default='do_rea(bool)=1|rea_prob(float)=0.5|rea_value(str)=random|rea_scale(float)=[0.02,0.4]|rea_ratio(float)=[0.3,3.33]')
    
    # testing
    parser.add_argument('--eval_freq', type=partial(str2list, f=int), default='20', help='test epochs (default: 20)')
    parser.add_argument('--show_nums', type=int, default=50)
    parser.add_argument('--dist_metric', type=str, default='cosine', help='distance metric for re-ranking (default: euclidean)')
    parser.add_argument('--use_cython', type=bool, default=True, help='using cython for evaluation (default: True)')
    parser.add_argument('--test_flip', type=bool, default=True, help='testing with flipped images (default: True)')
    parser.add_argument('--search_options', type=partial(str2list, f=int), default='3,2', help='')
    parser.add_argument('--rerank', type=bool, default=False, help='using re-ranking (default: False)')
    parser.add_argument('--rerank_k1', type=int, default=20, help='k1 for re-ranking (default: 20)')
    parser.add_argument('--rerank_k2', type=int, default=6, help='k2 for re-ranking (default: 6)')
    parser.add_argument('--rerank_lambda', type=float, default=0.3, help='lambda for re-ranking (default: 0.3)')
    
    # model
    parser.add_argument('--model', type=str, default='ReIDMamba', help='model name (default: BoT)')
    parser.add_argument('--model_path', type=str, default='', help='path to pre-trained model (default: None)') #keep_rates(float)=[0.75,0.75,0.75]|keep_list(str)=[6,12,18]|            |keep_rates(int)=[2,2,2,2,2,2,2]|keep_list(str)=[3,6,9,12,15,18,21]|ordered(bool)=1|sample_wise(bool)=1
    parser.add_argument('--model_kwargs', type=str2dict, default='backbone_name(str)=mambar_small_patch16_224|drop_path_rate(float)=0.3|num_cls_tokens(int)=12|cls_reduce(int)=4|use_cid(bool)=1|stride_size(int)=16|num_branches(int)=3|token_fusion_type(str)=max', help='kwargs for model ("cls_bias(bool)=1|cls_weight(float)=1.0|tst_list(int)=[1,2,3,4]")')
    parser.add_argument('--ema', type=str2dict, default='ema_model(bool)=0|ema_decay(float)=0.9992', help='ema for model (default: ema_model(bool)=1|ema_decay(float)=0.9999)')

    # loss
    parser.add_argument('--loss', type=str2list, default='triplet_loss,cross_entropy_loss,ratr_intra_loss,ratr_inter_loss', help='loss function name (default: CrossEntropyLoss)')
    parser.add_argument('--loss_weights', type=partial(str2list, f=float), default='1.0,1.0,1.0,1.0', help='loss weights (default: 1.0 for all losses, can only write one value)')
    parser.add_argument('--loss_nums', type=partial(str2list, f=int), default='3,3,1,1', help='loss numbers (default: 1 for all losses, can only write one value)')
    parser.add_argument('--loss_kwargs', type=partial(str2list, f=str2dict), default='margin(float)=1.2,label_smoothing(float)=0.1,N(int)=3|PK(int)=[16,4]|tau(float)=0.1,N(int)=3|PK(int)=[16,4]|tau(float)=0.1', help='kwargs for loss function ("****")')

    # optimizer
    parser.add_argument('--optim', type=str, default='sgd', help='optimizer name')
    parser.add_argument('--optim_kwargs', type=str2dict, default='lr(float)=0.008|weight_decay(float)=0.0|momentum(float)=0.9|nesterov(bool)=0', help='kwargs for optimizer ("****")')
    parser.add_argument('--lr_scheduler', type=str, default='LinearWarmupLrScheduler')
    parser.add_argument('--lr_scheduler_kwargs', type=str2dict, default='warmup_epochs(int)=4|warmup_iters(int)=1200|lr_multiplier(float)=1e-2|lrs2(str)=CosineAnnealingLR|lrs2_kwargs(str2dict)={T_max(int)=155|eta_min(float)=8e-6}')
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--freeze_bb', type=int, default=0)
    parser.add_argument('--eval_bb', type=int, default=0)
    parser.add_argument('--amp', type=bool, default=True, help='using amp for training (default: False)')
    parser.add_argument('--grad_clip', type=float, default=10.0, help='gradnorm clip (default: 0)') # msmt17: 30.0; Market1501: 25.0; dukemtmc: 30.0; CUHK03: 25.0
    
    args = parser.parse_args()
    
    tb_writer, logger, savedir, seeds = setup(args, determenistic=True, benchmark=True)
    
    dl_trn, dl_tst = build_dataloaders(args)
    model = build_model(args, dl_trn.num_cls, dl_trn.num_cid)
    model.to('cuda')
    if args.ema['ema_model']:
        model_ema = ModelEma(model, decay=args.ema['ema_decay'])
        logger.info(f'Using EMA with decay {args.ema["ema_decay"]}')
    else:
        model_ema = None
    
    optim, lrs = build_optimizer(args, model)
    loss = Loss(args)
    loss.to('cuda')
    evaluator = Evaluator(args, model, dl_tst, tb_writer)
 
    eng = Engine(seeds, args, model, model_ema, optim, lrs, loss, evaluator, dl_trn, dl_tst, tb_writer, savedir)
    logger.info(f'\nStarting training...')
    
    eng.train_test()