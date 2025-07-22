from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    qdrant_url: str
    redis_url: str
    qdrant_collection_name: str
    llama_cloud_api_key: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
