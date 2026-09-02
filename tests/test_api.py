"""API contract: status codes, validation, and response shapes."""
from __future__ import annotations

import pytest

from tests.conftest import requires_intelligence, requires_model


class TestMeta:
    def test_root_lists_endpoints(self, api_client):
        response = api_client.get("/")
        assert response.status_code == 200
        assert "endpoints" in response.json()

    def test_health_reports_readiness(self, api_client):
        response = api_client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] in {"healthy", "degraded"}
        assert "model_available" in body

    def test_openapi_schema_is_served(self, api_client):
        assert api_client.get("/openapi.json").status_code == 200


@requires_model
class TestPredictionEndpoint:
    def test_valid_request_returns_200(self, api_client, valid_payload):
        response = api_client.post("/predict/attrition", json=valid_payload)
        assert response.status_code == 200

        body = response.json()
        assert body["employee_id"] == valid_payload["EmployeeID"]
        assert 0.0 <= body["attrition_probability"] <= 1.0
        assert body["risk_level"] in {"HIGH", "MEDIUM", "LOW"}
        assert body["model_version"]

    @pytest.mark.parametrize(
        "field,value",
        [
            ("Age", 7),                    # below minimum
            ("Age", 200),                  # above maximum
            ("JobSatisfaction", 9),        # outside the 1-4 Likert scale
            ("OverTime", "Sometimes"),     # outside the allowed literal
            ("Department", "Marketing"),   # not a known department
            ("MonthlyIncome", -100),       # negative pay
            ("WorkLifeBalance", 0),        # below the scale
        ],
    )
    def test_invalid_field_returns_422(self, api_client, valid_payload, field, value):
        payload = {**valid_payload, field: value}
        assert api_client.post("/predict/attrition", json=payload).status_code == 422

    def test_missing_required_field_returns_422(self, api_client, valid_payload):
        payload = {k: v for k, v in valid_payload.items() if k != "Age"}
        assert api_client.post("/predict/attrition", json=payload).status_code == 422

    def test_unknown_field_is_rejected(self, api_client, valid_payload):
        """extra='forbid' - a typo in a field name must not be silently ignored."""
        payload = {**valid_payload, "Salary": 50_000}
        assert api_client.post("/predict/attrition", json=payload).status_code == 422

    def test_empty_body_returns_422(self, api_client):
        assert api_client.post("/predict/attrition", json={}).status_code == 422

    def test_known_employee_returns_200(self, api_client, employees):
        employee_id = int(employees["EmployeeID"].iloc[0])
        response = api_client.get(f"/predict/attrition/{employee_id}")
        assert response.status_code == 200
        assert response.json()["employee_id"] == employee_id

    def test_unknown_employee_returns_404(self, api_client):
        assert api_client.get("/predict/attrition/999999").status_code == 404

    def test_model_info_returns_metrics(self, api_client):
        body = api_client.get("/predict/model/info").json()
        assert body["algorithm"]
        assert 0.0 <= body["roc_auc"] <= 1.0
        assert 0.0 < body["decision_threshold"] < 1.0


@requires_intelligence
class TestDashboardEndpoints:
    def test_summary_shape(self, api_client):
        body = api_client.get("/dashboard/summary").json()
        for key in ["total_employees", "high_risk_employees", "average_engagement"]:
            assert key in body
        assert body["total_employees"] > 0

    def test_risk_counts_sum_to_headcount(self, api_client):
        body = api_client.get("/dashboard/summary").json()
        total = (body["high_risk_employees"] + body["medium_risk_employees"]
                 + body["low_risk_employees"])
        assert total == body["total_employees"]

    def test_attrition_by_department(self, api_client):
        rows = api_client.get("/dashboard/attrition-by-department").json()
        assert rows
        assert {"Department", "headcount", "high_risk_pct"} <= set(rows[0])

    def test_risk_distribution_covers_all_bands(self, api_client):
        rows = api_client.get("/dashboard/risk-distribution").json()
        assert {r["risk_level"] for r in rows} == {"HIGH", "MEDIUM", "LOW"}

    def test_skill_gaps_respects_limit(self, api_client):
        assert len(api_client.get("/dashboard/skill-gaps?limit=5").json()) <= 5

    def test_skill_gaps_rejects_bad_limit(self, api_client):
        assert api_client.get("/dashboard/skill-gaps?limit=0").status_code == 422

    def test_recommendations_filter_by_risk(self, api_client):
        rows = api_client.get("/dashboard/recommendations?risk_level=HIGH&limit=10").json()
        assert all(r["RiskLevel"] == "HIGH" for r in rows)

    def test_recommendations_reject_invalid_risk(self, api_client):
        assert api_client.get("/dashboard/recommendations?risk_level=CRITICAL").status_code == 422


@requires_intelligence
class TestWorkforceEndpoints:
    def test_employee_record_returns_full_detail(self, api_client, employees):
        employee_id = int(employees["EmployeeID"].iloc[0])
        body = api_client.get(f"/employees/{employee_id}").json()
        assert body["EmployeeID"] == employee_id
        assert "skill_gap_detail" in body
        assert "recommendations" in body

    def test_unknown_employee_returns_404(self, api_client):
        assert api_client.get("/employees/999999").status_code == 404

    def test_employee_list_paginates(self, api_client):
        first = api_client.get("/employees?limit=5").json()
        second = api_client.get("/employees?limit=5&offset=5").json()
        assert len(first["employees"]) == 5
        ids_first = {e["EmployeeID"] for e in first["employees"]}
        ids_second = {e["EmployeeID"] for e in second["employees"]}
        assert not (ids_first & ids_second)

    def test_employee_list_filters_by_department(self, api_client):
        body = api_client.get("/employees?department=Sales&limit=10").json()
        assert all(e["Department"] == "Sales" for e in body["employees"])

    def test_role_requirements(self, api_client):
        body = api_client.get("/skills/role/Research Scientist").json()
        assert body["job_role"] == "Research Scientist"
        assert body["required_skills"]

    def test_unknown_role_returns_404(self, api_client):
        assert api_client.get("/skills/role/Astronaut").status_code == 404

    def test_employee_skills_split_held_and_missing(self, api_client, employees):
        employee_id = int(employees["EmployeeID"].iloc[0])
        body = api_client.get(f"/skills/employee/{employee_id}").json()
        assert "skills_held" in body
        assert "skills_missing" in body

    def test_engagement_summary(self, api_client):
        body = api_client.get("/engagement/summary").json()
        assert 0 <= body["average_engagement"] <= 100
