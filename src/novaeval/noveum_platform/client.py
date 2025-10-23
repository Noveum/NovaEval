"""
Noveum Platform API Client.

This module provides a synchronous client for interacting with the Noveum Platform
API. It handles authentication, request/response processing, and error handling for
traces, datasets, and scorer results.
"""

import os
from typing import Optional

from dotenv import load_dotenv

from novaeval.utils.logging import get_logger

# All exceptions and models are now imported through the API classes
from .noveum_datasets_api import DatasetsAPI
from .noveum_scorer_results_api import ScorerResultsAPI
from .noveum_traces_api import TracesAPI
from .utils import create_authenticated_session

# Load environment variables
load_dotenv()

logger = get_logger(__name__)


class NoveumClient(TracesAPI, DatasetsAPI, ScorerResultsAPI):
    """
    Unified client for Noveum Platform API.

    Provides methods for traces, datasets, and scorer results with clear
    prefixed method names to avoid ambiguity.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.noveum.ai",
        timeout: float = 30.0,
    ):
        """
        Initialize the NoveumClient.

        Args:
            api_key: Noveum API key. If not provided, will try to load from
                     NOVEUM_API_KEY environment variable.
            base_url: Base URL for the Noveum API. Defaults to https://api.noveum.ai
            timeout: Request timeout in seconds. Defaults to 30.0.
        """
        self.api_key = api_key or os.getenv("NOVEUM_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "API key is required. Provide it directly or set NOVEUM_API_KEY environment variable."
            )

        # Setup session with authentication
        self.session = create_authenticated_session(self.api_key)

        # Initialize parent API classes with shared session
        super().__init__(self.session, self.base_url, self.timeout)
