from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# -----------------------------------------------------------------------------
# Smoke test : l'API est disponible
# -----------------------------------------------------------------------------
def test_predict_smoke():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "API is up and running!"


# -----------------------------------------------------------------------------
# Exercice 1.1 : prédiction correcte avec [1.0, 2.0, 3.0]
# -----------------------------------------------------------------------------
def test_predict_success_1_2_3():
    response = client.post("/predict", json={"features": [1.0, 2.0, 3.0]})
    assert response.status_code == 200
    assert response.json() == {"predictions": [2.0, 4.0, 6.0]}


# -----------------------------------------------------------------------------
# Exercice 1.2 : prédiction incorrecte
# On vérifie que l'API NE renvoie PAS un résultat volontairement faux.
# Le test réussit tant que l'implémentation est correcte ; il servirait
# aussi de régression si quelqu'un cassait la logique de prédiction.
# -----------------------------------------------------------------------------
def test_predict_does_not_return_wrong_result():
    response = client.post("/predict", json={"features": [1.0, 2.0, 3.0]})
    wrong_expected = {"predictions": [3.0, 5.0, 7.0]}  # résultat volontairement faux
    assert response.status_code == 200
    assert response.json() != wrong_expected


# -----------------------------------------------------------------------------
# Exercice 1.3 : JSON incorrect — champ "features" manquant
# -----------------------------------------------------------------------------
def test_predict_missing_features_field():
    response = client.post("/predict", json={
        "feature1": 3.5,
        "feature2": 1.2,
        "feature3": 4.9
    })
    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"] == "Field required"


# -----------------------------------------------------------------------------
# Cas nominal repris du TD
# -----------------------------------------------------------------------------
def test_predict_success():
    response = client.post("/predict", json={"features": [3.5, 1.2, 4.9]})
    assert response.status_code == 200
    assert response.json() == {"predictions": [7.0, 2.4, 9.8]}

