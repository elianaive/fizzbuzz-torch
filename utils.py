from typing import Dict, Any
import torch.nn as nn
import wandb

def setup_wandb(config: Dict[str, Any], project_name: str) -> None:
    """Initialize a wandb run with given config."""
    wandb.init(project=project_name, config=config)

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)