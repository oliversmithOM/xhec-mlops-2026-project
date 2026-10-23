# Abalone Age Prediction

Predict abalone age from physical measurements: explore the data, train a
linear-regression pipeline, schedule retraining with Prefect, and serve the
saved pipeline through FastAPI. Predictions are **age in years** (`Rings + 1.5`),
not ring counts.

## Team

| Name | GitHub |
| --- | --- |
| Andreea Patarlageanu | [AndreeaPatarlageanu](https://github.com/AndreeaPatarlageanu) |
| Oliver Smith | [oliversmith13](https://github.com/oliversmith13) (contributions), [oliversmithOM](https://github.com/oliversmithOM) (repository owner) |
| Rémi Laget | [Remilaget](https://github.com/Remilaget) |
| Ryan Balech | [RyanBalech](https://github.com/RyanBalech) |

---

## Quick Start (Fast Setup)

Prerequisites: Git, **Python 3.11**, and [uv](https://docs.astral.sh/uv/getting-started/installation/). Run all commands from the repository root:

```bash
git clone https://github.com/oliversmithOM/xhec-mlops-2026-project.git
cd xhec-mlops-2026-project
uv sync --locked --python 3.11
uv run python -m scripts.download_data
uv run python -m src.modelling.main data/abalone.csv
uv run uvicorn src.web_service.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/docs** to test predictions interactively via Swagger UI, or send a request from another terminal:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary @examples/predict.json
```

*(Stop the local server with `Ctrl+C` before proceeding to Docker to free port 8000).*

---

## Project Walkthrough

The project is structured around the 4 main stages of the MLOps lifecycle:

```
Part 1: Exploration & MLflow Experiments
   ↓
Parts 2 & 3: Modular Training Pipeline & Scheduled Retraining (Prefect)
   ↓
Part 4: Serving & Containerization (FastAPI & Docker)
```

### Part 1: Exploration and MLflow Experiments

Install notebook dependencies into the environment:

```bash
uv sync --locked --python 3.11 --group notebook
```

In VS Code or your preferred IDE, select the project's `.venv` interpreter as the notebook kernel:
1. Run `notebooks/eda.ipynb`: Covers distributions, missing values, duplicates, categorical counts, outliers, and correlations.
2. Run `notebooks/modelling.ipynb`: Compares the baseline linear regression with a log-target experiment. Both runs are logged to a local SQLite MLflow tracking store.

Launch the MLflow UI to compare metrics and parameters:

```bash
uv run --group notebook mlflow ui --backend-store-uri sqlite:///data/mlflow.db --host 127.0.0.1 --port 5000
```

Open **http://127.0.0.1:5000**. The SQLite database and artifacts live under `data/`.

---

### Parts 2 and 3: Training Pipeline and Scheduled Retraining

`src/modelling/` contains modular scripts for preprocessing, training/evaluation, serialization, and batch inference:
- Preprocessing fits only on the training split (80/20 train/test split, `random_state=42`).
- Target columns (`Rings` and `Age`) are excluded from features.
- Categoricals are one-hot encoded, and numeric inputs are scaled.

#### Train the model and run offline batch predictions

```bash
# Train pipeline and save artifact to src/web_service/local_objects/model.pkl
uv run python -m src.modelling.main data/abalone.csv

# Run batch predictions on CSV
uv run python -m src.modelling.predicting src/web_service/local_objects/model.pkl data/abalone.csv
```

Use `--model-path path/to/model.pkl` to specify an alternate output path.

#### Scheduled retraining with Prefect

For automated daily retraining, start the local Prefect server in one terminal:

```bash
uv run prefect server start --host 127.0.0.1 --port 4200
```

In a second terminal, configure the API endpoint and register the deployment:

```powershell
# PowerShell
$env:PREFECT_API_URL = "http://127.0.0.1:4200/api"
uv run python -m src.modelling.deployment
```

```bash
# macOS/Linux
PREFECT_API_URL=http://127.0.0.1:4200/api uv run python -m src.modelling.deployment
```

The flow is scheduled to retrain daily at **02:00 UTC**. View dashboard runs at **http://127.0.0.1:4200**. To trigger an immediate retraining run from a separate terminal:

```bash
uv run prefect deployment run "Abalone model training/daily-abalone-retraining" --watch
```

Stop the server/runner with `Ctrl+C` when finished.

---

### Part 4: Serving and Containerization

Part 4 exposes the trained model via a production-grade FastAPI web service and packages it inside a lightweight, non-root Docker container.

#### 1. API input and model lifecycle

* **Endpoint**: `POST /predict` accepts 8 fields matching the Kaggle training data: `sex` (`M`, `F`, or `I`), `length`, `diameter`, `height`, `whole_weight`, `shucked_weight`, `viscera_weight`, and `shell_weight`.
* **Validation**: Missing fields, extra fields (like `Rings`), negative numbers, numeric strings, and invalid categories are rejected with HTTP **422**.
* **Model Loading**: The model pipeline is loaded **once at startup**. Missing or corrupt model artifacts fail startup immediately.
* **Integrity**: `/health` and `/predict` return the model's SHA-256 hash so you can verify which model version is actively serving predictions.

Test via PowerShell:
```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -Body (Get-Content examples/predict.json -Raw)
```

#### 2. Run the API in Docker

Make sure Docker Desktop (or Docker Engine) is running. Build the image:

```bash
docker build -f Dockerfile.app -t abalone-api .
```

> [!IMPORTANT]
> **Port availability check:**
> The Docker container maps to port `8000` by default. If your local Uvicorn process is still running, Docker will fail to bind port 8000.
> - **Check if port 8000 is occupied:**
>   - PowerShell: `Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue`
>   - macOS/Linux: `lsof -i :8000`
> - **Free port 8000:** Stop the running Uvicorn terminal with `Ctrl+C`, or terminate it:
>   - PowerShell: `Stop-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess -Force`
>   - macOS/Linux: `kill -9 $(lsof -t -i:8000)`
> - **Alternative (change host port):** You can also bind to another host port such as `8080` by changing `-p 127.0.0.1:8000:8001` to `-p 127.0.0.1:8080:8001`.

Run the container with the model directory mounted read-only:

**PowerShell:**
```powershell
docker run --rm --name abalone-api -p 127.0.0.1:8000:8001 --mount "type=bind,source=$((Get-Location).Path)/src/web_service/local_objects,target=/app/src/web_service/local_objects,readonly" abalone-api
```

**macOS/Linux:**
```bash
docker run --rm --name abalone-api -p 127.0.0.1:8000:8001 --mount "type=bind,source=$PWD/src/web_service/local_objects,target=/app/src/web_service/local_objects,readonly" abalone-api
```

#### 3. Verify container health and predictions

From a second terminal:

```bash
# Run HTTP smoke test against the live container
uv run python -m scripts.smoke_api
```

Check the Docker health status:

* **PowerShell:**
  ```powershell
  docker inspect --format "{{.State.Health.Status}}" abalone-api
  ```
* **macOS/Linux:**
  ```bash
  docker inspect --format '{{.State.Health.Status}}' abalone-api
  ```

*(Health starts as `starting` and transitions to `healthy` once the endpoint passes health checks).*

Stop the container when finished:
```bash
docker stop abalone-api
```

---

## Results and Limitations

The 4,177-row dataset has no missing values or duplicate rows. We preserve the Part 1 rows, including outliers and zero heights. With a reproducible 80/20 split (`random_state=42`), 836 rows are held out:

| Model | RMSE (years) | MAE (years) | R² |
| --- | ---: | ---: | ---: |
| Linear regression baseline (served) | 2.212 | 1.593 | 0.548 |
| Log-target experiment | 2.220 | 1.570 | 0.545 |

The baseline is retained for its slightly lower RMSE and simplicity. These are single-split results, not guarantees for new populations. Linear regression may extrapolate poorly outside the training distribution. There is no automatic model selection, model promotion, or prediction interval.

---

## Checks and Continuous Integration

Each teammate runs `uv run pre-commit install` once in their own clone.

```bash
uv run pre-commit run --all-files
uv run python -m unittest discover -s tests
```

GitHub runs checks on pushes and pull requests: formatting/linting, training and API tests, Docker build, and an actual container HTTP smoke test. CI uses a deterministic synthetic model fixture for the container test, so it does not depend on Kaggle availability. Real-data accuracy is evaluated separately by the training command.

If GNU Make is installed, `make setup`, `make data`, `make train`, `make api`, `make test`, `make check`, and `make docker-build` wrap the same commands. Make is optional; all direct commands above work in PowerShell.

---

## Repository Layout and Contributions

- `notebooks/`: Part 1 EDA and MLflow experiments.
- `src/modelling/`: Parts 2–3 training modules, Prefect flow and deployment.
- `src/web_service/`: Part 4 validated API and model loading.
- `Dockerfile.app`, `bin/run_services.sh`: container build and startup.
- `scripts/`: verified data download and live API smoke check.
- `tests/`: training, serialization, API validation, and lifecycle tests.
- `examples/predict.json`: runnable API request.
- `pyproject.toml`, `uv.lock`: Part 0 dependency management.
- `.github/workflows/ci.yaml`: checks and container verification.

All assessed work must be merged into `main`. Every student must contribute meaningful commits using their GitHub-linked identity. Merge pull requests with **Create a merge commit**, never squash, to preserve individual history.
