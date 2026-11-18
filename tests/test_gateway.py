from fastapi.testclient import TestClient

from gateway.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_models():
    response = client.get("/models")
    assert response.status_code == 200
    assert len(response.json()["available_tasks"]) == 5


def test_invalid_task():
    response = client.post(
        "/chat",
        json={
            "task_type": "unknown",
            "message": "Hello",
        },
    )
    assert response.status_code == 400


def test_empty_message():
    response = client.post(
        "/chat",
        json={
            "task_type": "general",
            "message": "",
        },
    )
    assert response.status_code == 422
