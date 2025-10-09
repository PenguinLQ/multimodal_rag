from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application Configuration
    data_dir: str = Field(..., env="DATA_DIR")

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get application settings instance."""
    return settings
