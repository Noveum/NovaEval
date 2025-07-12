"""
Integrations package for NovaEval.

This package contains integrations with external services and platforms,
including the Noveum.ai platform.
"""

from novaeval.integrations.credentials import CredentialManager
from novaeval.integrations.noveum import NoveumIntegration
from novaeval.integrations.s3 import S3Integration

__all__ = [
    "CredentialManager",
    "NoveumIntegration",
    "S3Integration",
]
