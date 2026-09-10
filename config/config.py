from config.config_types import params
import yaml
from pathlib import Path

Root_path=Path(__file__).parent
param_path=Root_path/'params'


def load_config():
    with open(param_path,'r') as file:
        yaml_config= yaml.safe_load(file)
        
        param=params.model_validate(obj=yaml_config,strict=True)

        return param

params_config= load_config()

