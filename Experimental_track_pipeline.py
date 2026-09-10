from langfuse.api import datasets
from mlflow.entities import dataset
from config.config import params_config
from langfuse import get_client
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
Root_dir = Path(__file__).parent
eval_data_path= Root_dir/"data"/"evaluation"/"eval_dataset"
langfuse=get_client



def params_setup(parameter:dict)->list:
    params = []
    for key, value in params_config.items():
        if isinstance(value, dict):
            params.append(value)
        else:
            params.append(value)
    return params

def cur_eva_path(path:str)->str:
    eval_datasets=Path.glob(path)
    curr_eval=max(eval_datasets)
    return curr_eval

sys_prompt= langfuse.getprompt(params_setup()[prompt])

if __name__=='__main__':
   
