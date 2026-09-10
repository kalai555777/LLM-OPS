import mlflow
from pathlib import Path

ROOT=Path(__file__).parent

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("rag-app")
with mlflow.start_run(run_name="test_run") as run:
    mlflow.log_param('chunksize',70)
    mlflow.log_param('chunkoverlap',12)
    mlflow.log_param('output_dim',1024)
    mlflow.log_artifact(local_path=ROOT/"src"/'app'/'rag_workflow.py')
