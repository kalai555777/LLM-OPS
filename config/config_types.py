from pydantic import BaseModel

class rag__app(BaseModel):
  prompt_label: str
  llm_model: str
  embedding_model: str
  dimensions: int
  chunk_size: int
  chunk_overlap: int
  encoding_name: str
  search_type: str
  k:int
  collection_name: str
class Filterationconfig(BaseModel):
  synthetic_input_quality_threshold: float
  max_quality_retries: int
  critic_model: str
  model_config={'extra':'forbid'}

class  evolutionconfig(BaseModel):
    num_evolutions: int
    evolution:dict[str,float]

    model_config={'extra':'forbid'}

class ContextConstructionConfig(BaseModel):
    critic_model:str
    max_contexts_per_document: int
    context_quality_threshold: float
    max_retries: int
    Context_chunk_size: int
    Context_chunk_overlap: int
    model_config={'extra':'forbid'}
class generateEval(BaseModel):
    file_name:str
    model_config={'extra':'forbid'}
class asynconfig(BaseModel):
    throttle_value:int
    max_concurrent:int
    model_config={'extra':'forbid'}

class dispalyconfig(BaseModel):
     result_dir:str
     report_dir:str

     model_config={'extra':'forbid'}

class evaluation_pip(BaseModel):
    Evaluation_model:str
    asyn_config:asynconfig
    DisplayConfig:dispalyconfig

    model_config={'extra':'forbid'}

class Gold_data(BaseModel):
  max_goldens_per_context:int
  Golden_file_name:str
  filtration_config: Filterationconfig
  evolution_config: evolutionconfig
  ContextConstruction_Config: ContextConstructionConfig
  synthesizer_model:str
  
  Generate_eval: generateEval

  model_config={'extra':'forbid'}
class params(BaseModel):
    Rag_app:rag__app
    Golden_dataset:Gold_data
    Evaluation: evaluation_pip

    model_config={'extra':'forbid'}