from utils.registry import Registry
import logging

logger = logging.getLogger(__name__)

MODEL_REGISTRY = Registry("MODEL")

from .bot import BoT
from .transreid import TransReID
from .reidmamba import ReIDMambaR


def build_model(args, num_classes, num_cids):
    logger.info("\n# --- Model --- #")
    return MODEL_REGISTRY[args.model](num_classes=num_classes, num_cids=num_cids,img_size=args.img_size, **args.model_kwargs)