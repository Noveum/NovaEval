"""
Datasets package for NovaEval.

This package contains dataset loaders and processors for various
evaluation datasets.
"""

from novaeval.datasets.base import BaseDataset
from novaeval.datasets.mmlu import MMLUDataset
from novaeval.datasets.huggingface import HuggingFaceDataset
from novaeval.datasets.custom import CustomDataset

__all__ = [
    "BaseDataset",
    "MMLUDataset", 
    "HuggingFaceDataset",
    "CustomDataset",
]

