from typing import List
from data import FizzBuzzLabeler
from model_types import TrainingMetrics

def evaluate_model_accuracy(results: List[str], start: int, end: int) -> float:
    """Calculate the accuracy of model predictions against true FizzBuzz output."""
    correct = 0
    total = 0
    for i, pred in zip(range(start, end), results):
        true_label = FizzBuzzLabeler.decode(FizzBuzzLabeler.encode(i))(i)
        if pred == true_label:
            correct += 1
        total += 1
    return (correct / total) * 100