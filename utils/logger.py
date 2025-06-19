import logging
import os
import sys
import shutil


def create_file(exp, basedir='./logs'):
    if not os.path.exists(basedir):
        os.mkdir(basedir)
    if not os.path.exists(os.path.join(basedir, exp)):
        os.mkdir(os.path.join(basedir, exp))
    
    existing_folders = []
    for folder in os.listdir(os.path.join(basedir, exp)):
        if os.path.isdir(os.path.join(basedir, exp, folder)):
            try:
                existing_folders.append(int(folder))
            except:
                pass
    # existing_folders = [int(folder) for folder in os.listdir(os.path.join(basedir, exp))
    #                     if os.path.isdir(os.path.join(basedir, exp, folder))]

    for max_number in range(1000):
        if max_number not in existing_folders:
            path = os.path.join(basedir, exp, str(max_number))
            os.mkdir(path)
            return path
    
    # max_number = max(existing_folders) if len(existing_folders) else -1
    # path = os.path.join(basedir, exp, str(max_number + 1))
    # os.mkdir(path)
    # return path


def setup_logger(exp, load_cfg='', basedir='./runs', fileout=True, savecodes=True):
    savedir =create_file(exp, basedir)
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # stdout logging: master only
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s", datefmt="%m/%d %H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # file logging: all workers
    if fileout:
        plain_formatter = logging.Formatter(
                "[%(asctime)s] %(name)s <<%(levelname)s>>: %(message)s", datefmt="%m/%d %H:%M:%S"
                )
        filename = os.path.join(savedir, "log.txt")
        fh = logging.FileHandler(filename, 'w')
        fh.setLevel(logging.INFO)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)
        
    if savecodes:
        shutil.copytree(os.getcwd(), os.path.join(savedir, 'code'), ignore=shutil.ignore_patterns('runs', '__pycache__', '*.pth','.history','.vscode','.model_cache'))
    
    logger.info(f'Experiment \'{exp}\' started, saving to \'{savedir}\'.') 
    if load_cfg:    
        logger.info(f'Loading config from \'{load_cfg}\'.')
    return logger, savedir



