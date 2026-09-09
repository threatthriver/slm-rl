"""
Integration tests for Cognition Studio FastAPI backend.
Validates all 6 core API endpoints and responses.
"""

import pytest
from fastapi.testclient import TestClient
from app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "device" in data
    assert "format_compliance" in data
    assert "gsm8k_pass_rate" in data
    assert "lora_parameters" in data


def test_generate_endpoint(client):
    req = {
        "prompt": "Betty has $50 and her friend gives her $20. How much does she have?",
        "temperature": 0.1,
        "max_tokens": 64,
    }
    res = client.post("/api/generate", json=req)
    assert res.status_code == 200
    data = res.json()
    assert "raw_text" in data
    assert "think" in data
    assert "answer" in data
    assert "latency_ms" in data
    assert "tokens_per_sec" in data


def test_analyze_csao_steps_with_error(client):
    chain = (
        "<think>\n"
        "Step 1: Natalia sold 48 clips in April.\n"
        "Step 2: In May, she sold half as many clips: 48 / 2 = 24.\n"
        "Step 3: Altogether she sold 48 + 24 = 82 clips.\n"
        "Step 4: Therefore Natalia has 82 clips.\n"
        "</think>\n"
        "<answer>82</answer>"
    )
    res = client.post("/api/analyze-steps", json={"chain": chain, "gamma": 0.9})
    assert res.status_code == 200
    data = res.json()
    assert data["pivot_localized"] is True
    assert data["pivot_step"] == 3
    assert len(data["steps"]) == 4
    # Step 1 & 2 should be valid with positive advantage
    assert data["steps"][0]["advantage"] == 0.5
    assert data["steps"][1]["advantage"] == 0.5
    # Step 3 is fatal pivot error
    assert data["steps"][2]["advantage"] == -1.0
    assert data["steps"][2]["status"] == "pivot_error"
    # Step 4 is downstream discounted
    assert data["steps"][3]["advantage"] < 0
    assert data["steps"][3]["status"] == "downstream_fallout"


def test_analyze_csao_steps_clean(client):
    chain = (
        "<think>\n"
        "Step 1: John buys 5 apples.\n"
        "Step 2: Each apple is $2: 5 * 2 = 10.\n"
        "</think>\n"
        "<answer>10</answer>"
    )
    res = client.post("/api/analyze-steps", json={"chain": chain, "gamma": 0.9})
    assert res.status_code == 200
    data = res.json()
    assert data["pivot_localized"] is False
    assert len(data["steps"]) == 2
    assert data["steps"][0]["status"] == "valid"
    assert data["steps"][1]["status"] == "valid"


def test_duel_endpoint(client):
    res = client.post("/api/duel", json={"round_idx": 1})
    assert res.status_code == 200
    data = res.json()
    assert "prover_output" in data
    assert "skeptic_output" in data
    assert "prover_payoff" in data
    assert "skeptic_payoff" in data
    assert "winner" in data


def test_curriculum_endpoint(client):
    res = client.get("/api/curriculum")
    assert res.status_code == 200
    data = res.json()
    assert len(data["tiers"]) == 5
    assert data["current_active_tier"] == 5


def test_tier_test_endpoint(client):
    res = client.post("/api/test-tier", json={"tier": 1})
    assert res.status_code == 200
    data = res.json()
    assert "problem" in data
    assert "expected_target" in data
    assert "output" in data


def test_telemetry_endpoint(client):
    res = client.get("/api/telemetry")
    assert res.status_code == 200
    data = res.json()
    assert "history" in data
    assert "peak_reward" in data
    assert "format_compliance_pct" in data


def test_simulate_reward_endpoint(client):
    req = {
        "math_weight": 0.4,
        "fmt_weight": 0.3,
        "aha_weight": 0.15,
        "comp_weight": 0.15,
        "is_correct": True,
        "is_formatted": True,
        "has_aha": True,
        "tokens_used": 35,
    }
    res = client.post("/api/simulate-reward", json=req)
    assert res.status_code == 200
    data = res.json()
    assert data["total_reward"] > 0.8
    assert data["badge"] == "Optimal Deliberation"


def test_compare_endpoint(client):
    res = client.get("/api/compare")
    assert res.status_code == 200
    data = res.json()
    assert "pillars" in data
    assert len(data["pillars"]) >= 5
    assert "empirical_metrics" in data
