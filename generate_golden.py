
import os
from deepeval.synthesizer.synthesizer import Synthesizer
from deepeval.synthesizer.config import FiltrationConfig, EvolutionConfig, ContextConstructionConfig
from dotenv import load_dotenv
from deepeval.models import GeminiModel
from deepeval.synthesizer.types import Evolution

from pathlib import Path
from local_embedder import LocalSentenceTransformerEmbedding
import logging
from deepeval.models import GPTModel

load_dotenv()

Root_Dir = Path(__file__).parent
Data_Dir = Root_Dir / "datasets"
Data_Dir.mkdir(parents=True, exist_ok=True)
Doc_path = Root_Dir / "data"/"process"


Critic_model = GPTModel(
    model="Qwen/Qwen2.5-72B-Instruct",
    api_key=os.environ["HF_TOKEN"],
    base_url="https://router.huggingface.co/v1",
    temperature=0)


logging.basicConfig(level=logging.ERROR)
logging.getLogger("deepeval").setLevel(logging.ERROR)
logging.getLogger("deepeval").addHandler(logging.StreamHandler())
  # ← Use SAME key!


def get_docs(dir_path=Path|str) -> list[str]:
    directory_path= Path(dir_path)
    if directory_path.is_dir() and directory_path.exists():
        paths=directory_path.glob("*.txt")
    return [path.as_posix() for path in paths]
        



filter_config = FiltrationConfig(critic_model=Critic_model,
synthetic_input_quality_threshold=0.7)

evolution_config = EvolutionConfig(num_evolutions=1,evolutions={Evolution.MULTICONTEXT:0.7,Evolution.CONCRETIZING:0.7,Evolution.CONSTRAINED:0.7,Evolution.COMPARATIVE:0.7})

context_construction_config = ContextConstructionConfig(critic_model=Critic_model,embedder=LocalSentenceTransformerEmbedding(),max_contexts_per_document=3,max_context_length=3,context_quality_threshold=0.3)



synthesizer = Synthesizer(model=Critic_model,filtration_config=filter_config,evolution_config=evolution_config)

Goldens= synthesizer.generate_goldens_from_docs(document_paths=get_docs(Doc_path),context_construction_config=context_construction_config)

Golden_Dir = Root_Dir / "goldens"
Golden_Dir.mkdir(parents=True, exist_ok=True)

synthesizer.save_as(file_type="json",directory=Golden_Dir,file_name="golden_dataset")
