"""
Configuration for OpenAI embedding models.
Centralized settings for vector embedding operations.
"""
import os
from dataclasses import dataclass, field


def _env_bool(key: str, default: str = "false") -> bool:
    """Parse environment boolean values consistently."""
    return os.getenv(key, default).lower() == "true"


@dataclass
class EmbeddingConfig:
    """Configuration for OpenAI embedding models."""

    # Provider selection
    provider: str = field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "openai"))

    # Model configurations
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))

    # Vector dimensions
    openai_embedding_dim: int = 1536

    # Performance settings
    batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))

    # OpenAI API retry and timeout configuration
    max_retries: int = int(os.getenv("EMBEDDING_MAX_RETRIES", "6"))
    request_timeout: int = int(os.getenv("EMBEDDING_REQUEST_TIMEOUT", "30"))
    retry_min_seconds: int = int(os.getenv("EMBEDDING_RETRY_MIN_SECONDS", "1"))
    retry_max_seconds: int = int(os.getenv("EMBEDDING_RETRY_MAX_SECONDS", "20"))

    # Concurrency control
    max_concurrency: int = int(os.getenv("EMBEDDING_MAX_CONCURRENCY", "5"))

    # Logging
    log_similarity_matches: bool = _env_bool("LOG_SIMILARITY_MATCHES", "false")

    @property
    def model_name(self) -> str:
        if self.provider == "openai":
            return self.openai_model
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

    @property
    def embedding_dimension(self) -> int:
        if self.provider == "openai":
            return self.openai_embedding_dim
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def get_model_name(self) -> str:
        return self.model_name

    def get_embedding_dimension(self) -> int:
        return self.embedding_dimension


# Default configuration instance
DEFAULT_CONFIG = EmbeddingConfig()

