"""
Tests for components API endpoints.
"""


class TestComponentsEndpoints:
    """Test components discovery and management endpoints."""

    def test_list_components(self, client):
        """Test listing all components."""
        response = client.get("/api/v1/components/")
        assert response.status_code == 200
        data = response.json()
        assert "types" in data
        assert "counts" in data

    def test_list_models(self, client):
        """Test listing model components."""
        response = client.get("/api/v1/components/models")
        assert response.status_code == 200
        data = response.json()
        assert "components" in data
        assert isinstance(data["components"], list)

    def test_list_datasets(self, client):
        """Test listing dataset components."""
        response = client.get("/api/v1/components/datasets")
        assert response.status_code == 200
        data = response.json()
        assert "components" in data
        assert isinstance(data["components"], list)

    def test_list_scorers(self, client):
        """Test listing scorer components."""
        response = client.get("/api/v1/components/scorers")
        assert response.status_code == 200
        data = response.json()
        assert "components" in data
        assert isinstance(data["components"], list)

    def test_get_nonexistent_component(self, client):
        """Test getting nonexistent component returns 404."""
        response = client.get("/api/v1/components/models/nonexistent")
        assert response.status_code == 404

    def test_reload_components(self, client):
        """Test reloading component registry."""
        response = client.post("/api/v1/components/reload")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data

    def test_invalid_component_type(self, client):
        """Test invalid component type returns 404."""
        response = client.get("/api/v1/components/invalid/test")
        assert response.status_code == 404
