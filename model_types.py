from dataclasses import dataclass

@dataclass
class TrainingMetrics:
    """Stores training metrics for each epoch."""
    epoch: int
    loss: float
    accuracy: float