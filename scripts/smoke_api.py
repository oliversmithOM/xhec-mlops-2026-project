"""Check a running API against the mounted model, including validation and docs."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from src.modelling.predicting import predict
from src.web_service.lib.inference import FEATURE_COLUMNS


def main(base_url: str, model_path: Path) -> None:
    for attempt in range(60):
        try:
            with urlopen(f"{base_url}/health", timeout=3) as response:
                health = json.load(response)
            break
        except (URLError, TimeoutError, ConnectionError):
            if attempt == 59:
                raise
            time.sleep(1)
    checksum = hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert health == {"status": "ok", "model_sha256": checksum}, health
    example = json.loads(
        (Path(__file__).resolve().parents[1] / "examples/predict.json").read_text()
    )
    request = Request(
        f"{base_url}/predict",
        data=json.dumps(example).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=10) as response:
        result = json.load(response)
    expected = predict(
        model_path, pd.DataFrame([example]).rename(columns=FEATURE_COLUMNS)
    )[0]
    assert math.isclose(result["age_years"], expected, rel_tol=1e-10, abs_tol=1e-10), (
        result
    )
    assert result["model_sha256"] == checksum, result
    request = Request(
        f"{base_url}/predict",
        data=b'{"sex":"invalid"}',
        headers={"Content-Type": "application/json"},
    )
    try:
        urlopen(request, timeout=10)
    except HTTPError as exc:
        assert exc.code == 422, exc.code
    else:
        raise AssertionError("Invalid inputs were accepted")
    with urlopen(f"{base_url}/docs", timeout=10) as response:
        assert response.status == 200
    with urlopen(f"{base_url}/", timeout=10) as response:
        assert "Specimen measurements" in response.read().decode()
    with urlopen(f"{base_url}/static/app.js", timeout=10) as response:
        assert response.status == 200
    print(f"API smoke test passed: {json.dumps(result)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("src/web_service/local_objects/model.pkl"),
    )
    args = parser.parse_args()
    main(args.base_url.rstrip("/"), args.model_path)
