from typing import List, Dict, Optional
import torch
import torch.nn as nn
import numpy as np
from data import FizzBuzzLabeler

class ResidualBlock(nn.Module):
    """Residual block with optional bottleneck."""
    def __init__(self, size: int, bottleneck_size: Optional[int] = None,
                 dropout: float = 0.0, activation: str = 'relu'):
        super().__init__()
        
        act_fns = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'selu': nn.SELU(),
            'tanh': nn.Tanh()
        }
        self.act_fn = act_fns[activation]
        
        if bottleneck_size is None:
            self.block = nn.Sequential(
                nn.Linear(size, size),
                nn.LayerNorm(size),
                self.act_fn,
                nn.Dropout(dropout),
                nn.Linear(size, size),
                nn.LayerNorm(size)
            )
        else:
            self.block = nn.Sequential(
                nn.Linear(size, bottleneck_size),
                nn.LayerNorm(bottleneck_size),
                self.act_fn,
                nn.Dropout(dropout),
                nn.Linear(bottleneck_size, size),
                nn.LayerNorm(size)
            )
            
    def forward(self, x):
        orig_shape = x.shape
        out = self.block(x)
        assert out.shape == orig_shape, f"Shape mismatch in ResidualBlock: input {orig_shape}, output {out.shape}"
        return self.act_fn(x + out)

class LinearFizzBuzz(nn.Module):
    def __init__(self, 
                 input_size: int,
                 hidden_sizes: List[int] = None,
                 dropout: float = 0.0,
                 activation: str = 'relu',
                 init_scale: float = 1.0,
                 use_residual: bool = False,
                 use_layer_norm: bool = True,
                 bottleneck_factor: float = 0.25,
                 num_residual_blocks: int = 2):
        super().__init__()
        print(f"Initializing model with input_size={input_size}, hidden_sizes={hidden_sizes}")
        
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.input_size = input_size
        
        # Activation function mapping
        act_fns = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'selu': nn.SELU(),
            'tanh': nn.Tanh()
        }
        self.act_fn = act_fns[activation]
        
        # Initialize layers
        self.layers = self._build_layers(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            init_scale=init_scale,
            bottleneck_factor=bottleneck_factor,
            num_residual_blocks=num_residual_blocks
        )
        
        # Initialize with better scaling
        self._reset_parameters()
        
        print(f"Model architecture:")
        for i, layer in enumerate(self.layers):
            print(f"  Layer {i}: {layer}")
    
    def _init_layer(self, layer: nn.Linear, scale: float = 1.0) -> None:
        """Initialize a linear layer with scaled initialization."""
        if isinstance(self.act_fn, nn.GELU):
            # Special init for GELU
            nn.init.xavier_uniform_(layer.weight, gain=scale * 1.2)
        elif isinstance(self.act_fn, nn.SELU):
            # SELU paper initialization
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='linear')
            scale = scale * np.sqrt(1/3)
            with torch.no_grad():
                layer.weight.data *= scale
        else:
            # Default Xavier initialization
            nn.init.xavier_uniform_(layer.weight, gain=scale)
        
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    
    def _build_layers(self, input_size: int, hidden_sizes: List[int],
                     dropout: float, init_scale: float,
                     bottleneck_factor: float, num_residual_blocks: int) -> nn.Sequential:
        """Build the layer structure."""
        layers = []
        
        if hidden_sizes is None or len(hidden_sizes) == 0:
            layers.append(nn.Linear(input_size, FizzBuzzLabeler.NUM_CLASSES))
            return nn.Sequential(*layers)
        
        prev_size = input_size
        
        # Input projection with layer norm
        input_layer = nn.Linear(input_size, hidden_sizes[0])
        self._init_layer(input_layer, init_scale)
        layers.extend([
            input_layer,
            nn.LayerNorm(hidden_sizes[0]) if self.use_layer_norm else nn.Identity(),
            self.act_fn,
            nn.Dropout(dropout)
        ])
        prev_size = hidden_sizes[0]
        
        # Hidden layers
        for i, hidden_size in enumerate(hidden_sizes[1:], 1):
            if self.use_residual and prev_size == hidden_size:
                # Add residual blocks when sizes match
                for _ in range(num_residual_blocks):
                    bottleneck_size = int(hidden_size * bottleneck_factor)
                    layers.append(
                        ResidualBlock(
                            size=hidden_size,
                            bottleneck_size=bottleneck_size,
                            dropout=dropout,
                            activation=type(self.act_fn).__name__.lower()
                        )
                    )
            
            # Regular layer
            linear = nn.Linear(prev_size, hidden_size)
            self._init_layer(linear, init_scale)
            
            layers.extend([
                linear,
                nn.LayerNorm(hidden_size) if self.use_layer_norm else nn.Identity(),
                self.act_fn,
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Output projection
        output_layer = nn.Linear(prev_size, FizzBuzzLabeler.NUM_CLASSES)
        self._init_layer(output_layer, init_scale/2.0)
        layers.append(output_layer)
        
        return nn.Sequential(*layers)
    
    def _reset_parameters(self) -> None:
        """Apply custom initialization to all parameters."""
        if self.use_residual:
            for m in self.modules():
                if isinstance(m, ResidualBlock):
                    if hasattr(m.block[-2], 'weight'):
                        with torch.no_grad():
                            m.block[-2].weight.data *= 0.1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        if len(x.shape) == 1:
            x = x.view(1, -1)
        elif len(x.shape) > 2:
            x = x.view(batch_size, -1)
            
        assert x.shape[1] == self.input_size, \
            f"Input shape mismatch: expected {self.input_size} features, got {x.shape[1]}"
            
        #print(f"Model forward input shape: {x.shape}")
        out = self.layers(x)
        #print(f"Model forward output shape: {out.shape}")
        
        return out

    def get_complexity_stats(self) -> Dict[str, int]:
        """Return model complexity statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 32 / (8 * 1024 * 1024),
            'residual_blocks': sum(1 for m in self.modules() if isinstance(m, ResidualBlock))
        }