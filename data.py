from typing import List, Tuple, Callable, Union
import torch
from torch.utils.data import Dataset

class BinaryEncoder:
    """Handles binary encoding of numbers."""
    def __init__(self, bits: int):
        self.bits = bits
        self.mask = 2**torch.arange(bits-1, -1, -1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Convert numbers to binary representation."""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        
        # Convert to binary representation
        encoded = x.unsqueeze(-1).bitwise_and(self.mask).ne(0).float()
        return encoded.reshape(encoded.size(0), -1)  # Flatten output

    def get_input_size(self) -> int:
        """Return the size of the binary representation."""
        return self.bits


class FizzBuzzLabeler:
    """Handles FizzBuzz labeling logic."""
    NUM_CLASSES = 4

    @staticmethod
    def encode(x: int) -> int:
        """Convert a number to its FizzBuzz class (0-3)."""
        if x % 15 == 0:
            return 3  # FizzBuzz
        elif x % 5 == 0:
            return 2  # Buzz
        elif x % 3 == 0:
            return 1  # Fizz
        return 0  # Number

    @staticmethod
    def decode(label: int) -> Callable[[int], str]:
        """Convert a class label to its string representation function."""
        return {
            0: lambda x: str(x),
            1: lambda _: "Fizz",
            2: lambda _: "Buzz",
            3: lambda _: "FizzBuzz"
        }[label]


class FizzBuzzDataset(Dataset):
    """Dataset for FizzBuzz training."""
    def __init__(self, 
                 range_spec: Union[Tuple[int, int], List[int]], 
                 encoder: BinaryEncoder):
        """Initialize dataset."""
        if isinstance(range_spec, tuple):
            start, end = range_spec
            self.data = torch.arange(start, end)
        else:
            self.data = torch.tensor(range_spec)

        # Encode data and create labels
        self.binary = encoder.encode(self.data)
        self.labels = torch.tensor([FizzBuzzLabeler.encode(x.item()) 
                                    for x in self.data])

        # Verify shapes
        assert len(self.binary.shape) == 2, f"Binary shape should be 2D, got {self.binary.shape}"
        assert self.binary.shape[0] == len(self.data), f"Binary batch size mismatch: {self.binary.shape[0]} vs {len(self.data)}"
        assert self.binary.shape[1] == encoder.bits, f"Binary width should be {encoder.bits}, got {self.binary.shape[1]}"

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.binary[idx], self.labels[idx]
