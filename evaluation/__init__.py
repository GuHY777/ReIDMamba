from .rank_cylib import evaluate_rank
import logging
from tabulate import tabulate
import torch
import torch.nn.functional as F
import numpy as np
from .rerank import *
from tqdm import tqdm


logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, args, model, dl, tb_writer=None):
        self.use_cython = args.use_cython
        self.test_flip = args.test_flip
        self.use_cid = model.use_cid
        self.search_options = args.search_options
        
        self.rerank = args.rerank
        self.k1 = args.rerank_k1
        self.k2 = args.rerank_k2
        self.lambda_value = args.rerank_lambda
        
        self.dist_metric = args.dist_metric
        
        self.dl = dl
        self.model = model
        
        self.tb_writer = tb_writer
        self.header = ['Epoch', 'Rank-1', 'Rank-3', 'Rank-5', 'Rank-10', 'mAP', 'mINP']
        self.results_ori = []
        self.results_flip = []

        logger.info('\n# --- Evaluator initialized --- #')
        logger.info(f'Cython={self.use_cython}\nFlip={self.test_flip}\nSearch options={self.search_options}\nRerank={self.rerank}\nK1={self.k1}\nK2={self.k2}\nLambda={self.lambda_value}\nMetric={self.dist_metric}\nCID={self.use_cid}')
        
        
    @torch.no_grad()
    def __call__(self, epoch, model=None):
        fss, ps, cs = self.extract_features(model)
        assert hasattr(self.dl, 'num_qry')
        
        results = []
        for i, fs in enumerate(fss):
            if i==1 and not self.test_flip:
                return results[0], None
            
            qf = fs[:self.dl.num_qry, :]
            q_pids = ps[:self.dl.num_qry]
            q_camids = cs[:self.dl.num_qry]
            
            gf = fs[self.dl.num_qry:, :]
            g_pids = ps[self.dl.num_qry:]
            g_camids = cs[self.dl.num_qry:]
            
            if self.dist_metric == 'cosine':
                qf = F.normalize(qf, dim=1)
                gf = F.normalize(gf, dim=1)
            
            if self.rerank:
                logger.info('Using reranking...')
                ori_D = torch.cdist(qf, gf).numpy()**2 # squared euclidean distance!!
                jac_D = jaccard_distance(fs, self.dl.num_qry, self.k1, self.k2, self.search_options[1])
                D = self.lambda_value * ori_D + (1 - self.lambda_value) * jac_D
                I = np.argsort(D, axis=1)
            else:
                I = faiss_search(qf, gf, -1, self.search_options[0])
            
            all_cmc, all_AP, all_INP = evaluate_rank(
                I, q_pids.numpy(), g_pids.numpy(), q_camids.numpy(), g_camids.numpy(),
            )
            
            if i == 0:
                self.results_ori.append([
                    epoch, 
                    all_cmc[0]*100, 
                    all_cmc[2]*100, 
                    all_cmc[4]*100, 
                    all_cmc[9]*100, 
                    np.mean(all_AP)*100, 
                    np.mean(all_INP)*100
                    ])
            else:
                self.results_flip.append([
                    epoch, 
                    all_cmc[0]*100, 
                    all_cmc[2]*100, 
                    all_cmc[4]*100, 
                    all_cmc[9]*100, 
                    np.mean(all_AP)*100, 
                    np.mean(all_INP)*100
                    ])
                
            if self.tb_writer is not None:
                self.tb_writer.add_scalars(f"Metrics_{'flip' if i==1 else 'ori'}" if model is None else f"Metrics_ema_{'flip' if i==1 else 'ori'}", {
                    'Rank-1': all_cmc[0]*100, 
                    'Rank-3': all_cmc[2]*100, 
                    'Rank-5': all_cmc[4]*100, 
                    'Rank-10': all_cmc[9]*100,
                    'mAP': np.mean(all_AP)*100,
                    'mINP': np.mean(all_INP)*100
                    }, epoch)
            if i == 0:
                logger.info(f"Results (original):\n{tabulate(self.results_ori, headers=self.header, tablefmt='orgtbl')}")
            else:
                logger.info(f"Results (flipped):\n{tabulate(self.results_flip, headers=self.header, tablefmt='orgtbl')}")
            
            results.append({
            'Rank-1': all_cmc[0]*100, 
            'Rank-3': all_cmc[2]*100, 
            'Rank-5': all_cmc[4]*100, 
            'Rank-10': all_cmc[9]*100,
            'mAP': np.mean(all_AP)*100,
            'mINP': np.mean(all_INP)*100
            })
            
        return results
    
    def extract_features(self, model=None, dl=None):
        if model is None:
            self.model.eval()
        else:
            model.eval()
            
        fs = []
        fs_flip = []
        ps = []
        cs = []
        
        if dl is None:
            dl =self.dl
        if dl.dataset.__len__() % dl.batch_size:
            N = dl.dataset.__len__() // dl.batch_size + 1
        else:
            N = dl.dataset.__len__() // dl.batch_size
        
        for (imgs, pids, cids) in tqdm(dl, desc='Extracting features', total=N):
            ps.append(pids)
            cs.append(cids)
            
            imgs = imgs.to('cuda')
            if self.use_cid:
                cids_kwargs = {'cids': cids.to('cuda')}
            else:
                cids_kwargs = {}
                
            f = self.model(imgs, **cids_kwargs) if model is None else model(imgs, **cids_kwargs)
            fs.append(f)
            if self.test_flip:
                imgs_flip = torch.flip(imgs, [3])
                f_flip = self.model(imgs_flip, **cids_kwargs) if model is None else model(imgs_flip, **cids_kwargs)
                f = (f + f_flip) / 2.0
                fs_flip.append(f)
        
        fs = torch.cat(fs, dim=0).detach().to('cpu')
        ps = torch.cat(ps, dim=0).detach().to('cpu')
        cs = torch.cat(cs, dim=0).detach().to('cpu')
        if self.test_flip:
            fs_flip = torch.cat(fs_flip, dim=0).detach().to('cpu')
        
        return (fs, fs_flip), ps, cs
    
    def reset(self):
        self.results_ori = []