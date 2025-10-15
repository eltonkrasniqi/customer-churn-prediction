"""Utility functions for configuration, logging, and versioning."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger for the churn prediction system.
    
    Args:
        level: Logging level (default: logging.INFO)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("churn_prediction")
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration
        
    Raises:
        FileNotFoundError: If configuration file does not exist
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def get_model_version() -> str:
    """Generate a timestamp-based model version string.
    
    Returns:
        Version string in format YYYYMMDD_HHMM
    """
    return datetime.now().strftime("%Y%m%d_%H%M")


def ensure_dir(path: Path) -> Path:
    """Create directory if it does not exist.
    
    Args:
        path: Directory path to create
        
    Returns:
        The path object
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
