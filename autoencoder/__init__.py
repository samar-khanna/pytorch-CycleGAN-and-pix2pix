
import importlib

from omegaconf import OmegaConf

def instantiate_from_config(config):
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def load_vae(config_path):
    config = OmegaConf.load(config_path)
    from .autoencoder import AutoencoderKL
    return AutoencoderKL(**config['model']["params"])