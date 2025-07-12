"""
Datasets package for NovaEval.

This package contains dataset loaders and processors for various
evaluation datasets.
"""

from novaeval.datasets.base import BaseDataset
from novaeval.datasets.custom import CustomDataset
from novaeval.datasets.huggingface import HuggingFaceDataset
from novaeval.datasets.mmlu import MMLUDataset

__all__ = [
    "BaseDataset",
    "CustomDataset",
    "HuggingFaceDataset",
    "MMLUDataset",
]
