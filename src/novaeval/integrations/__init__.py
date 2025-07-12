"""
Integrations package for NovaEval.

This package contains integrations with external services and platforms,
including the Noveum.ai platform.
"""

from novaeval.integrations.noveum import NoveumIntegration
from novaeval.integrations.s3 import S3Integration
from novaeval.integrations.credentials import CredentialManager

__all__ = [
    "NoveumIntegration",
    "S3Integration", 
    "CredentialManager",
]

