# config.py - Configuration loaded from config.yaml
# This file loads configuration from config.yaml and exports it as Python constants

import yaml
from pathlib import Path

# Load YAML configuration file
config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'

try:
    with open(config_path, 'r') as f:
        _config = yaml.safe_load(f)
except FileNotFoundError:
    raise FileNotFoundError(
        f"config.yaml not found at {config_path}. "
        "Please ensure config.yaml exists in the project root."
    )
except yaml.YAMLError as e:
    raise ValueError(f"Error parsing config.yaml: {e}")

# Export configuration as constants
# This maintains backward compatibility with existing code that imports config

# Embedding Configuration
EMBEDDING_MODEL = _config.get('embedding', {}).get(
    'model', 'sentence-transformers/all-MiniLM-L6-v2'
)

# Database Configuration
CHROMA_COLLECTION_NAME = _config.get('database', {}).get(
    'collection_name', 'rag_documents'
)
CHROMA_DB_PATH = _config.get('database', {}).get('path', './chroma_db')

# LLM Configuration
DEFAULT_LLM_TEMPERATURE = _config.get('llm', {}).get('temperature', 0.0)

# File Paths
DATA_DIRECTORY = _config.get('paths', {}).get('data_directory', 'data')

# Export the raw config dict for advanced users who want nested access
CONFIG = _config
