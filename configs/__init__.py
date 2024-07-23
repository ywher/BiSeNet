

import importlib
import json

class cfg_dict(object):

    def __init__(self, d):
        self.__dict__ = d
        self.get = d.get


def set_cfg_from_file(cfg_path):
    spec = importlib.util.spec_from_file_location('cfg_file', cfg_path)
    cfg_file = importlib.util.module_from_spec(spec)
    spec_loader = spec.loader.exec_module(cfg_file)
    cfg = cfg_file.cfg
    return cfg_dict(cfg)


def cvt_cfg_dict_to_json(cfg):
    cfg_dict = dict(cfg.__dict__)
    cfg_dict.pop('get')
    return cfg_dict



