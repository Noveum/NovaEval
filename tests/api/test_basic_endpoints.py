"""
Basic tests for simple API endpoints.
"""


class TestBasicEndpoints:
    """Test basic API endpoints that don't require complex mocking."""

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data

    def test_ping_endpoint(self, client):
        """Test API ping endpoint."""
        response = client.get("/api/v1/ping")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "novaeval-api"

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint returns prometheus format."""
        response = client.get("/metrics")
        assert response.status_code == 200
        content = response.text
        assert "novaeval_http_requests_total" in content
        assert "TYPE" in content or "HELP" in content

    def test_openapi_endpoints(self, client):
        """Test OpenAPI documentation endpoints."""
        # Test OpenAPI JSON
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data
        assert "components" in data

        # Test docs
        response = client.get("/docs")
        assert response.status_code == 200

        # Test redoc
        response = client.get("/redoc")
        assert response.status_code == 200

    def test_404_handling(self, client):
        """Test 404 error handling."""
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_component_endpoints_basic(self, client):
        """Test basic component endpoints."""
        # Test main components endpoint
        response = client.get("/api/v1/components/")
        assert response.status_code == 200

        # Test individual component type listings
        for component_type in ["models", "datasets", "scorers"]:
            response = client.get(f"/api/v1/components/{component_type}")
            assert response.status_code == 200
            data = response.json()
            assert "components" in data
