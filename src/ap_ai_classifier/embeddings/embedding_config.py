"""
Configuration for OpenAI embedding models.
Centralized settings for vector embedding operations.
"""
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, Any


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

    # Cache configuration (simplified - no Redis for now)
    cache_prefix: str = "embeddings_similarity"
    cache_expiration_seconds: int = int(os.getenv("EMBEDDING_CACHE_EXPIRATION", str(60 * 60 * 24 * 30)))

    # Similarity thresholds
    high_similarity_threshold: float = float(os.getenv("HIGH_SIMILARITY_THRESHOLD", "0.55"))
    medium_similarity_threshold: float = float(os.getenv("MEDIUM_SIMILARITY_THRESHOLD", "0.35"))

    # Vector dimensions
    openai_embedding_dim: int = 1536

    # Performance settings
    batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))
    max_text_length: int = 8192

    # OpenAI API retry and timeout configuration
    max_retries: int = int(os.getenv("EMBEDDING_MAX_RETRIES", "6"))
    request_timeout: int = int(os.getenv("EMBEDDING_REQUEST_TIMEOUT", "30"))
    retry_min_seconds: int = int(os.getenv("EMBEDDING_RETRY_MIN_SECONDS", "1"))
    retry_max_seconds: int = int(os.getenv("EMBEDDING_RETRY_MAX_SECONDS", "20"))

    # Concurrency control (simplified - no semaphores)
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

    def to_dict(self) -> Dict[str, Any]:
        """Return a logging-friendly flattened dict."""
        base = asdict(self)
        base.update({
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "cache_expiration_hours": self.cache_expiration_seconds // 3600,
            "thresholds": {
                "high_similarity": self.high_similarity_threshold,
                "medium_similarity": self.medium_similarity_threshold,
            },
            "retry_config": {
                "max_retries": self.max_retries,
                "request_timeout": self.request_timeout,
                "retry_min_seconds": self.retry_min_seconds,
                "retry_max_seconds": self.retry_max_seconds,
            },
        })
        return base

    @classmethod
    def from_environment(cls) -> "EmbeddingConfig":
        """Return a new instance using environment variables."""
        return cls()


# Default configuration instance
DEFAULT_CONFIG = EmbeddingConfig()

