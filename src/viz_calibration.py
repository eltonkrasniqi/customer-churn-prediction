"""Visualization utilities for model calibration."""

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve

from src.core.config import CALIBRATION_BINS, PLOT_DPI
from src.utils import setup_logging


logger = setup_logging()


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str,
    n_bins: int = CALIBRATION_BINS,
    strategy: str = "uniform"
) -> None:
    """
    Generate and save a calibration (reliability) diagram.
    
    A well-calibrated model should have predicted probabilities that match
    the observed frequencies. The closer the curve is to the diagonal, the
    better the calibration.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        output_path: Path to save the plot
        n_bins: Number of bins for calibration curve
        strategy: Binning strategy ('uniform' or 'quantile')
    """
    logger.info(f"Generating calibration plot with {n_bins} bins")
    
    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy=strategy
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=2)
    
    # Plot model calibration
    ax.plot(prob_pred, prob_true, "o-", label="Model", linewidth=2, markersize=8)
    
    # Formatting
    ax.set_xlabel("Predicted Probability", fontsize=12)
    ax.set_ylabel("Observed Frequency", fontsize=12)
    ax.set_title("Calibration Plot (Reliability Diagram)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    
    plt.tight_layout()
    
    # Save figure
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Calibration plot saved to: {output_path}")


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Tuple[float, float]:
    """
    Compute calibration metrics: Brier score and ECE.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        
    Returns:
        Tuple of (brier_score, expected_calibration_error)
    """
    from sklearn.metrics import brier_score_loss
    
    # Brier score (lower is better, perfect = 0)
    brier = brier_score_loss(y_true, y_prob)
    
    # Expected Calibration Error (ECE)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=CALIBRATION_BINS, strategy="uniform")
    ece = np.mean(np.abs(prob_true - prob_pred))
    
    logger.info(f"Brier score: {brier:.4f}, ECE: {ece:.4f}")
    
    return brier, ece

