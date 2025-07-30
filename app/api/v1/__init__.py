"""
API v1 router configuration.

This module sets up the main API router for version 1 endpoints.
"""

from fastapi import APIRouter

# Import and include route modules
from app.api.v1 import components, datasets, evaluations, models, scorers

# Create the main API router for v1
api_router = APIRouter()

api_router.include_router(components.router, prefix="/components", tags=["Components"])
api_router.include_router(models.router, prefix="/models", tags=["Models"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["Datasets"])
api_router.include_router(scorers.router, prefix="/scorers", tags=["Scorers"])
api_router.include_router(
    evaluations.router, prefix="/evaluations", tags=["Evaluations"]
)


@api_router.get("/ping", tags=["Health"])
async def ping():
    """
    Simple ping endpoint for API v1.

    Returns:
        dict: Pong response
    """
    return {"message": "pong", "version": "v1"}
