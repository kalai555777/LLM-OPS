from deepeval.models.base_model import DeepEvalBaseEmbeddingModel
from sentence_transformers import SentenceTransformer


class LocalSentenceTransformerEmbedding(DeepEvalBaseEmbeddingModel):
    """
    Free, local embedding model for DeepEval using sentence-transformers.
    No API key needed. Downloads the model once, then runs fully offline.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def load_model(self):
        return self.model

    def embed_text(self, text: str):
        return self.model.encode(text).tolist()

    def embed_texts(self, texts: list[str]):
        return self.model.encode(texts).tolist()

    async def a_embed_text(self, text: str):
        return self.embed_text(text)

    async def a_embed_texts(self, texts: list[str]):
        return self.embed_texts(texts)

    def get_model_name(self):
        return self.model_name