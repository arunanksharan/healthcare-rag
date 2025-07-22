import logging
from openai import OpenAI

from data_retrieval.core.settings import settings

logger = logging.getLogger(__name__)
client = OpenAI(api_key=settings.openai_api_key)


def get_openai_embedding(text: str, model: str = "text-embedding-ada-002") -> list:
    """
    Generate an embedding for the input text using OpenAI's embedding model.
    """
    try:
        response = client.embeddings.create(input=text, model=model)
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        logger.error("Error generating OpenAI embedding: %s", e)
        raise e
