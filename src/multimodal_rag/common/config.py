from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application Configuration
    data_dir: str = Field(..., env="DATA_DIR")
    raw_subdir: str = Field(..., env="RAW_SUBDIR")
    image_subdir: str = Field(..., env="IMAGE_SUBDIR")
    processed_subdir: str = Field(..., env="PROCESSED_SUBDIR")
    text_content_file_name: str = Field(..., env="TEXT_CONTENT_FILE_NAME")
    image_content_file_name: str = Field(..., env="IMAGE_CONTENT_FILE_NAME")

    # CLIP Configuration
    embedding_model: str = Field("ViT-B/32", env="EMBEDDING_MODEL")

    # Milvus Configuration
    db_file: str = Field(..., env="DB_FILE")
    text_collection_name: str = Field(..., env="TEXT_COLLECTION_NAME")
    image_collection_name: str = Field(..., env="IMAGE_COLLECTION_NAME")
    dimension: int = Field(default=512)
    

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get application settings instance."""
    return settings
