from pathlib import Path

from src.ml.train import main, MODEL_PATH


def test_training_creates_model():
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
    
    main()
    
    assert MODEL_PATH.exists()
