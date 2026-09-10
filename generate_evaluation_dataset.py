from deepeval.dataset.dataset import EvaluationDataset
from deepeval.test_case import llm_test_case
from deepeval.test_case.llm_test_case import LLMTestCase
from dotenv import load_dotenv
from pathlib import Path
from src.app.rag_workflow import graph
from src.evals.individual_evals.test_recall_metric import retrieval_context

load_dotenv()


Root_dir=Path(__file__).parent
Golden_path= Root_dir/"goldens"/"golden_dataset.json"
Eval_data_path=Golden_path.parent.parent/"Eval_dataset"

Dataset=EvaluationDataset()

Dataset.add_goldens_from_json_file(file_path=Golden_path)

for golden in Dataset.goldens:
    final_state= graph.invoke(input=golden)
    test_cases=LLMTestCase(input=golden.input,actual_output=final_state.get("response"),expected_output=golden.expected_output,retrieval_context=[doc.page_content for doc in final_state.get("retrieved_docs")])

Dataset.add_test_case(test_case=test_cases)

Dataset.save_as(file_type="json",directory=Eval_data_path,file_name="evaluation_dataset",include_test_cases=True)
