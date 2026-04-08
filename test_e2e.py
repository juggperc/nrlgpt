import asyncio
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_api():
    print("--- Testing API Information (/api/info) ---")
    res = client.get("/api/info")
    print(res.json())

    req_data = {"home_team": "Storm", "away_team": "Panthers", "venue": "Suncorp"}

    print("\n--- Testing Prediction (/api/predict) ---")
    res = client.post("/api/predict", json=req_data)
    print(res.json())

    print("\n--- Testing SGM Generation (/api/sgm) ---")
    res = client.post("/api/sgm", json=req_data)
    print(res.json())

    print("\n--- Testing Simulation (/api/simulate) ---")
    res = client.post("/api/simulate", json={**req_data, "plays": 5})

    sim_data = res.json()
    if "plays" in sim_data:
        print(f"Generated {len(sim_data['plays'])} plays.")
        for p in sim_data["plays"]:
            print(f"Minute {p['minute']}: {p['commentary']}")
    else:
        print(sim_data)


if __name__ == "__main__":
    test_api()
