from deepeval.metrics import (GEval,AnswerRelevancyMetric,FaithfulnessMetric)
from deepeval.dataset.dataset import EvaluationDataset
from deepeval.test_case.llm_test_case import LLMTestCase
from deepeval.evaluate import evaluate
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case.llm_test_case import SingleTurnParams
from deepeval.evaluate.configs import AsyncConfig,DisplayConfig
from pathlib import Path
import os
from dotenv import load_dotenv

from deepeval.test_case.llm_test_case import LLMTestCase
from deepeval.evaluate import evaluate
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case.llm_test_case import SingleTurnParams
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from deepeval.models import DeepEvalBaseLLM
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# =========================================================
# ADD ONLY THIS HF CODE
# =========================================================

class HuggingFaceModel(DeepEvalBaseLLM):

    def __init__(self):
        self.model_name = "Qwen/Qwen3.6-27B"

        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.getenv("HF_TOKEN")
        )

    def load_model(self):
        return self.client

    def get_model_name(self):
        return self.model_name

    def generate(self, prompt, schema=None):

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        output = response.choices[0].message.content

        if schema:
            return schema.model_validate_json(output)

        return output

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)


hf_model = HuggingFaceModel()

answer_relevant=AnswerRelevancyMetric(model=hf_model)
faithful_level=FaithfulnessMetric(model=hf_model,penalize_ambiguous_claims=True,truths_extraction_limit=2
)

answer_simpleness=GEval(name="simpleness",model=hf_model,evaluation_params=[SingleTurnParams.INPUT,SingleTurnParams.ACTUAL_OUTPUT],
 evaluation_steps=[
        "Assess how easy the actual output is to understand.",
        "Assess whether the response uses clear and simple language.",
        "Assess whether the response avoids unnecessary complexity and overly technical wording.",
        "Assess whether the response is concise while still answering the input question adequately."
    ],rubric=[
    Rubric(score_range= (0,3), expected_outcome="Low simplicity"),
    Rubric(score_range=(4,7),expected_outcome="Moderate simplicity"),
    Rubric(score_range=(8,10),expected_outcome="High simplicity")],verbose_mode=True
)

dataset=EvaluationDataset()

Root_path=Path(__file__).parent
Eval_path=Root_path/'datasets'/"eval_dataset.json"



dataset.add_test_cases_from_json_file(file_path=Eval_path,
input_key_name="input",actual_output_key_name="actual_output",expected_output_key_name="expected_output",retrieval_context_key_name="retrieval_context")

test_cases=dataset.test_cases

evaluate(test_cases=test_cases,
metrics=[answer_relevant,faithful_level,answer_simpleness],
async_config=AsyncConfig(throttle_value=3,max_concurrent=5),
display_config=DisplayConfig(
    results_folder=(Root_path / "evaluation_results").as_posix(),
    file_output_dir=(Root_path / "evaluation_report").as_posix()
))
