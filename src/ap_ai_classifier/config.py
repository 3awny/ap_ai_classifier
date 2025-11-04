from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
BASE_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BASE_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)

# Project paths
DATA_DIR = BASE_DIR / "data"
RAW_CSV_PATH = DATA_DIR / "line_items.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Embedding configuration
# Note: SentenceTransformer config is here for backward compatibility
# OpenAI embedding config is in embeddings/embedding_config.py
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # For SentenceTransformer backend
EMBEDDING_DIM = 384  # SentenceTransformer dimension

# Retrieval configuration
TOP_K = 5
MIN_SIM_TO_ACCEPT = 0.45

# LLM configuration
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# Evaluation target
# F1-Macro target of 0.70 is specified in the assignment requirements.
# Rationale: 
# - F1-macro averages F1 scores across all classes (handles class imbalance)
# - 0.70 means ~70% precision and recall balanced across all classes
# - For AP classification: ~70% correct predictions minimizes manual review workload
#   while maintaining acceptable accuracy for financial data
# - Threshold balances automation benefits vs. risk of misclassification
# - Typical business ML thresholds: 0.60-0.80 (0.70 is mid-range, achievable but not trivial)
TARGET_F1_MACRO = 0.70
