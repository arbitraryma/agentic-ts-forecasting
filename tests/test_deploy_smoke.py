"""Lightweight checks for the API, service, graph, and MCP import."""

import os
import unittest

from fastapi.testclient import TestClient

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


class DeploySmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from api.app import app

        cls.client = TestClient(app)

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_invalid_csv_target_column(self):
        response = self.client.post(
            "/forecast/csv",
            files={
                "file": (
                    "series.csv",
                    "sales,other\n1,2\n3,4\n",
                    "text/csv",
                )
            },
            data={
                "target_column": "missing",
                "horizon": "3",
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("missing", response.json()["detail"])

    def test_forecast_endpoint_cheap_models(self):
        payload = {
            "values": [float(10 + (index % 12)) for index in range(36)],
            "horizon": 3,
            "target_name": "smoke",
            "models": ["ets"],
        }

        if not os.getenv("DASHSCOPE_API_KEY"):
            response = self.client.post("/forecast", json=payload)
            self.assertEqual(response.status_code, 503)
            return

        response = self.client.post("/forecast", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body["forecast"]), 3)
        self.assertIn(
            body["critic_decision"],
            {"ACCEPT", "RETRY"},
        )

    def test_retry_count_cannot_exceed_one(self):
        from langgraph.graph import END

        from agent.graph import route_after_critic

        self.assertEqual(
            route_after_critic({
                "critic_result": {"decision": "RETRY"},
                "retry_count": 0,
            }),
            "forecasting",
        )
        self.assertEqual(
            route_after_critic({
                "critic_result": {"decision": "RETRY"},
                "retry_count": 1,
            }),
            END,
        )
        self.assertEqual(
            route_after_critic({
                "critic_result": {"decision": "ACCEPT"},
                "retry_count": 0,
            }),
            END,
        )

    def test_mcp_server_imports(self):
        import mcp_server.server as server

        self.assertTrue(hasattr(server, "mcp"))
        self.assertTrue(
            hasattr(server, "run_full_forecasting_pipeline")
        )

    def test_forecast_length_matches_horizon(self):
        from services.forecasting_service import (
            forecast_from_series,
        )

        horizon = 3
        result = forecast_from_series(
            [float(10 + (index % 12)) for index in range(36)],
            horizon=horizon,
            target_name="smoke",
            seasonal_period=12,
            model_allowlist=["seasonal_naive"],
            compute_budget="low",
            fast=True,
            n_windows=1,
            verbose=False,
        )
        self.assertEqual(len(result["forecast"]), horizon)


if __name__ == "__main__":
    unittest.main()
