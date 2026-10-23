"""Validate the HTTP contract and real serialized-pipeline inference."""

import hashlib
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.modelling.training import train_model
from src.modelling.utils import pickle_object
from src.web_service.lib.inference import FEATURE_COLUMNS
from src.web_service.lib.models import AbaloneFeatures
from src.web_service.main import create_app
from tests.fixtures import training_data

EXAMPLE = json.loads(
    (Path(__file__).resolve().parents[1] / "examples/predict.json").read_text()
)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "model.pkl"
        features, target = training_data()
        self.model = train_model(features, target)
        pickle_object(self.model, self.path)

    def test_prediction_matches_pipeline_and_artifact_hash(self):
        with TestClient(create_app(self.path)) as client:
            response = client.post("/predict", json=EXAMPLE)
            self.assertEqual(response.status_code, 200)
            expected = self.model.predict(
                pd.DataFrame([EXAMPLE]).rename(columns=FEATURE_COLUMNS)
            )[0]
            self.assertAlmostEqual(response.json()["age_years"], expected)
            checksum = hashlib.sha256(self.path.read_bytes()).hexdigest()
            self.assertEqual(response.json()["model_sha256"], checksum)
            self.assertEqual(
                client.get("/health").json(),
                {
                    "status": "ok",
                    "model_sha256": checksum,
                },
            )
            home = client.get("/")
            self.assertEqual(home.status_code, 200)
            self.assertIn("text/html", home.headers["content-type"])
            self.assertIn("Specimen measurements", home.text)
            self.assertIn("default-src 'self'", home.headers["content-security-policy"])
            self.assertEqual(client.get("/docs").status_code, 200)
            schema = client.get("/openapi.json").json()
            self.assertIn("/predict", schema["paths"])

    def test_interface_assets_and_private_file_isolation(self):
        with TestClient(create_app(self.path)) as client:
            for name in ["app.css", "app.js", "shell.svg"]:
                with self.subTest(asset=name):
                    response = client.get(f"/static/{name}")
                    self.assertEqual(response.status_code, 200)
                    self.assertTrue(response.content)
            for path in [
                "/static/../local_objects/model.pkl",
                "/static/%2e%2e/local_objects/model.pkl",
                "/.git/config",
            ]:
                with self.subTest(path=path):
                    self.assertEqual(client.get(path).status_code, 404)

    def test_reject_invalid_inputs(self):
        cases = [
            {**EXAMPLE, "sex": "unknown"},
            {**EXAMPLE, "length": -1},
            {**EXAMPLE, "length": "0.455"},
            {**EXAMPLE, "length": True},
            {**EXAMPLE, "length": None},
            {**EXAMPLE, "Rings": 15},
            {**EXAMPLE, "Age": 16.5},
            {**EXAMPLE, "typo": 1},
            {key: value for key, value in EXAMPLE.items() if key != "sex"},
            {},
        ]
        with TestClient(create_app(self.path)) as client:
            for payload in cases:
                with self.subTest(payload=payload):
                    self.assertEqual(
                        client.post("/predict", json=payload).status_code, 422
                    )
            self.assertEqual(
                client.post("/predict", content="{bad json").status_code, 422
            )
            for value in [float("nan"), float("inf"), float("-inf")]:
                response = client.post(
                    "/predict",
                    content=json.dumps({**EXAMPLE, "length": value}),
                    headers={"Content-Type": "application/json"},
                )
                self.assertEqual(response.status_code, 422)

    def test_nonfinite_measurements_rejected(self):
        from pydantic import ValidationError

        for value in [float("nan"), float("inf"), float("-inf")]:
            with self.subTest(value=value), self.assertRaises(ValidationError):
                AbaloneFeatures(**{**EXAMPLE, "length": value})

    def test_zero_height_is_accepted_like_training_data(self):
        with TestClient(create_app(self.path)) as client:
            response = client.post("/predict", json={**EXAMPLE, "height": 0})
            self.assertEqual(response.status_code, 200)

    def test_missing_model_fails_startup(self):
        with (
            self.assertRaisesRegex(FileNotFoundError, "Train with"),
            TestClient(create_app(self.path.parent / "missing.pkl")),
        ):
            self.fail("Missing models must not report readiness")

    def test_corrupt_and_wrong_models_fail_startup(self):
        for artifact in [
            b"not a pickle",
            pickle.dumps({}),
            pickle.dumps(Pipeline([("model", LinearRegression())])),
        ]:
            with self.subTest(artifact=artifact[:10]):
                self.path.write_bytes(artifact)
                with (
                    self.assertRaises((TypeError, ValueError, pickle.UnpicklingError)),
                    TestClient(create_app(self.path)),
                ):
                    self.fail("Invalid model accepted")

    def test_wrong_training_schema_fails_startup(self):
        features, target = training_data()
        pickle_object(train_model(features.drop(columns="Sex"), target), self.path)
        with (
            self.assertRaisesRegex(ValueError, "feature names"),
            TestClient(create_app(self.path)),
        ):
            self.fail("Wrong feature schema accepted")

    def test_load_once_and_reload_only_on_restart(self):
        with TestClient(create_app(self.path)) as client:
            old = client.post("/predict", json=EXAMPLE).json()
            features, target = training_data()
            pickle_object(train_model(features, target + 10), self.path)
            self.assertEqual(client.post("/predict", json=EXAMPLE).json(), old)
        with TestClient(create_app(self.path)) as client:
            new = client.post("/predict", json=EXAMPLE).json()
            self.assertNotEqual(old["model_sha256"], new["model_sha256"])
            self.assertAlmostEqual(new["age_years"] - old["age_years"], 10)

    def test_prediction_failure_is_generic(self):
        with TestClient(create_app(self.path)) as client:
            with (
                patch(
                    "src.web_service.main.predict_age",
                    side_effect=ValueError("private path"),
                ),
                self.assertLogs("src.web_service.main", level="ERROR"),
            ):
                response = client.post("/predict", json=EXAMPLE)
            self.assertEqual(response.status_code, 500)
            self.assertEqual(response.json(), {"detail": "Model prediction failed"})

    def test_nonfinite_predictions_fail_safely(self):
        with TestClient(create_app(self.path)) as client:
            with (
                patch.object(
                    client.app.state.model, "predict", return_value=np.array([np.nan])
                ),
                self.assertLogs("src.web_service.main", level="ERROR"),
            ):
                response = client.post("/predict", json=EXAMPLE)
            self.assertEqual(response.status_code, 500)

    def test_environment_model_path(self):
        with (
            patch.dict("os.environ", {"MODEL_PATH": str(self.path)}),
            TestClient(create_app()) as client,
        ):
            self.assertEqual(client.get("/health").status_code, 200)
