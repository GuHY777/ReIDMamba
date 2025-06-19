import os.path as osp
import os
import glob
import re
from tabulate import tabulate


_name_base = {
    'Market1501': 'Market1501',
    'DukeMTMC': 'DukeMTMC',
    'MSMT17': 'MSMT17',
    'CUHK03-D': 'CUHK03/detected',
    'CUHK03-L': 'CUHK03/labeled',
    'OccludedDuke': 'Occluded_Duke',
}


class DataSource:
    def __init__(self, name):
        self._root = os.environ.get('DATA_ROOT', '/root/data/DataSets')
        self._base = osp.join(self._root, _name_base[name])
        self._name = name
        
        self._dir_trn = osp.join(self._base, 'bounding_box_train')
        self._dir_qry = osp.join(self._base, 'query')
        self._dir_gal = osp.join(self._base, 'bounding_box_test')
        assert osp.exists(self._dir_trn) and osp.exists(self._dir_qry) and osp.exists(self._dir_gal)
        
        self.infos_trn, self.pids_trn, self.cids_trn = self.process_dir(self._dir_trn)
        self.infos_qry, self.pids_qry, self.cids_qry = self.process_dir(self._dir_qry)
        self.infos_gal, self.pids_gal, self.cids_gal = self.process_dir(self._dir_gal)
    
    def process_dir(self, dir):
        image_pattern = osp.join(dir, '*.[jpg][png][jpeg]')
        image_files = glob.glob(image_pattern)
        
        infos = []
        pids = set()
        cids = set()
        for file in image_files:
            match = re.search(r'(-?\d+)_c(\d+)', osp.basename(file))
            assert match is not None
            pid, cid = match.group(1), match.group(2)
            if int(pid) < 0:
                continue # junk images are just ignored
            pid, cid = '_'.join([self._name, pid]), '_'.join([self._name, cid])
            infos.append([file, pid, cid])
            # '/root/data/DataSets/PersonReID_DataSets/Market1501/bounding_box_train/0001_c1.jpg'
            # 'Market1501_1'
            # 'Market1501_1'
            
            pids.add(pid)
            cids.add(cid)
        return infos, pids, cids

    def __repr__(self):
        data = [
            ['Train',   len(self.infos_trn), len(self.pids_trn), len(self.cids_trn)],
            ['Query',   len(self.infos_qry), len(self.pids_qry), len(self.cids_qry)],
            ['Gallery', len(self.infos_gal), len(self.pids_gal), len(self.cids_gal)],
        ]
        table = tabulate(data, headers=['Split', '#Images', '#PIDs', '#CIDs'], tablefmt='grid')
        return f'# --- {self._name} Dataset --- #\n{table}'