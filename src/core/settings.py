"""
Settings Module - Configuration Loading and Validation

This module handles loading and validating the configuration from config/settings.yaml
It provides the Settings dataclass and functions to load/validate configuration.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

import yaml
from pydantic import BaseModel, Field, ConfigDict


class LLMSettings(BaseModel):
    """LLM provider configuration"""
    provider: str
    model: str
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    azure_endpoint: Optional[str] = None
    api_version: Optional[str] = None
    deployment_name: Optional[str] = None
    base_url: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class EmbeddingSettings(BaseModel):
    """Embedding provider configuration"""
    provider: str
    model: str
    api_key: Optional[str] = None
    dimensions: int = 1536
    azure_endpoint: Optional[str] = None
    api_version: Optional[str] = None
    deployment_name: Optional[str] = None
    base_url: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class VisionLLMSettings(BaseModel):
    """Vision LLM configuration for image captioning"""
    enabled: bool = False
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    api_version: Optional[str] = None
    deployment_name: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class VectorStoreSettings(BaseModel):
    """Vector store configuration"""
    provider: str = "chroma"
    path: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class RetrievalSettings(BaseModel):
    """Retrieval engine configuration"""
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_final: int = 10
    rrf_k: int = 60

    model_config = ConfigDict(extra="allow")


class RerankerSettings(BaseModel):
    """Reranker configuration"""
    enabled: bool = False
    provider: str = "none"

    model_config = ConfigDict(extra="allow")


class EvaluationSettings(BaseModel):
    """Evaluation configuration"""
    enabled: bool = False
    providers: list = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class IngestionSettings(BaseModel):
    """Ingestion pipeline configuration"""
    chunking: Dict[str, Any] = Field(default_factory=dict)
    chunk_refiner: Dict[str, Any] = Field(default_factory=dict)
    metadata_enricher: Dict[str, Any] = Field(default_factory=dict)
    image_captioning: Dict[str, Any] = Field(default_factory=dict)
    batch_size: int = 10

    model_config = ConfigDict(extra="allow")


class ObservabilitySettings(BaseModel):
    """Observability (logging, tracing, dashboard) configuration"""
    logging: Dict[str, Any] = Field(default_factory=dict)
    tracing: Dict[str, Any] = Field(default_factory=dict)
    dashboard: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


class StorageSettings(BaseModel):
    """Storage paths configuration"""
    documents_path: str = "data/documents"
    images_path: str = "data/images"
    db_path: str = "data/db"
    cache_path: str = "cache"
    logs_path: str = "logs"
    sqlite: Dict[str, str] = Field(default_factory=dict)
    bm25: Dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


class Settings(BaseModel):
    """Root settings configuration"""
    llm: LLMSettings
    embedding: EmbeddingSettings
    vision_llm: VisionLLMSettings = Field(default_factory=VisionLLMSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)

    model_config = ConfigDict(extra="allow")


def _resolve_env_vars(value: Any) -> Any:
    """
    Recursively resolve environment variables in configuration.
    Supports ${ENV_VAR} syntax.
    """
    if isinstance(value, str):
        if value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.getenv(env_var, value)
        return value
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(v) for v in value]
    return value


def load_settings(config_path: Optional[str] = None) -> Settings:
    """
    Load settings from YAML configuration file.

    Args:
        config_path: Path to settings.yaml (defaults to config/settings.yaml)

    Returns:
        Settings object with resolved environment variables

    Raises:
        FileNotFoundError: If config file not found
        ValueError: If config validation fails
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    if not raw_config:
        raise ValueError("Configuration file is empty")

    # Resolve environment variables
    resolved_config = _resolve_env_vars(raw_config)

    try:
        return Settings(**resolved_config)
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}")


def validate_settings(settings: Settings) -> None:
    """
    Validate critical settings.

    Args:
        settings: Settings object to validate

    Raises:
        ValueError: If validation fails
    """
    # Check required LLM settings
    if not settings.llm.provider:
        raise ValueError("LLM provider is required")
    if not settings.llm.model:
        raise ValueError("LLM model is required")

    # Check required Embedding settings
    if not settings.embedding.provider:
        raise ValueError("Embedding provider is required")
    if not settings.embedding.model:
        raise ValueError("Embedding model is required")

    # Check VectorStore
    if not settings.vector_store.provider:
        raise ValueError("VectorStore provider is required")

    # Validate storage paths
    storage = settings.storage
    required_paths = [storage.documents_path, storage.db_path, storage.logs_path]
    for path in required_paths:
        Path(path).mkdir(parents=True, exist_ok=True)
