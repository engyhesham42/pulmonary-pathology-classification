# Chest X-ray Multi-Label Classification (Educational Prototype)

NIH ChestX-ray14 subset (13 diseases + *No Finding*) · DenseNet121 (TensorFlow/Keras)
· MLflow tracking · FastAPI + Docker · Streamlit UI with Grad-CAM.

> **Not for clinical use.** Educational prototype; no claims of clinical readiness.

## Architecture
```
Data audit + EDA → Preprocessing → Tracked models (MLflow: Run1/2/3)
  → Validation + per-class thresholds + XAI (Grad-CAM)
  → Versioned artifact (run3_finetuned.keras + inference_config.json)
  → Docker REST API (FastAPI) → Streamlit UI → Business decisions
```
```
xray-project/
├── notebooks/grad_project_improved.ipynb   data audit, EDA, training, MLflow, evaluation
├── models/        run3_finetuned.keras + inference_config.json   (from the notebook)
├── api/           FastAPI service (+ Dockerfile); guidance.py + llm_service.py = AI explanation layer
├── ui/            Streamlit app (dark theme, ui/.streamlit/config.toml) (+ Dockerfile)
├── reports/       run3_test_metrics.csv (+ add plots)
├── scripts/       make_dummy_model.py (placeholder for pipeline testing only)
├── tests/         API tests
└── docker-compose.yml
```

## Required files (from the notebook's Drive folder `x_ray_ds/models/`)
Copy into `models/`:
- `run3_finetuned.keras`
- `inference_config.json`

## Run
(UI: dark ChestAI theme with score bars, per-finding threshold ticks, Grad-CAM class selector, review priority, AI explanation and follow-up.)

```bash
docker compose up --build
```
UI http://localhost:8501 · API docs http://localhost:8000/docs · `curl localhost:8000/health`

```bash
curl -F "file=@sample.png" "http://localhost:8000/predict?explain=true&explain_class=Effusion"
```

## AI explanation + follow-up (Groq, optional)
`POST /predict?llm_explain=true` adds two fields to the response:
- `follow_up`: specialty suggestion from the **predefined mapping** in `api/guidance.py`
  (draft; must be reviewed by a clinician before any real use).
- `explanation`: plain-language text from Groq, written only from class names, scores and
  the threshold. The LLM never sees the image, cannot change predictions or referrals, and its
  output must pass a schema + rule checks (no treatment, urgency, invented numbers or
  unsupplied specialties). On any failure the result is `unavailable` and predictions are untouched.

Setup: copy `.env.example` to `.env` and set `GROQ_API_KEY` (docker compose reads it
automatically). Without a key everything else works and the explanation shows as unavailable.
**Never commit `.env`.** For per-class thresholds, the explanation uses the selected class's own threshold.

## How inference matches training
RGB → resize to 224×224 → divide by 255. ImageNet normalization is inside the model.
A class is *detected* when its sigmoid probability ≥ its per-class threshold
(`inference_config.json`, tuned for F1 on the validation set).

Review priority (rule-based; tune `PRIORITY_HIGH_RATIO` from validation results and justify it):
- **High:** an abnormal class is detected and probability/threshold ≥ ratio (default 1.25)
- **Medium:** abnormal detected below that ratio, or nothing clearly detected
- **Low:** only *No Finding* detected

## Results (test set, Run 3: fine-tuned DenseNet121)
| | Threshold 0.5 | Tuned thresholds |
|---|---|---|
| Macro AUC | 0.766 | 0.766 |
| Macro F1 | 0.252 | 0.283 |
| Macro precision | 0.182 | 0.234 |
| Macro recall | 0.538 | 0.377 |

Per-disease results: `reports/run3_test_metrics.csv`. Validation macro AUC across runs:
Run 1 baseline ≈ 0.69 → Run 2 (fixed, frozen) 0.705 → Run 3 (fine-tuned) 0.780.
MLflow UI: `mlflow ui --backend-store-uri sqlite:///mlflow.db` (in the folder with `mlflow.db`).

## Tests
```bash
pip install -r api/requirements.txt pytest httpx
pytest tests/ -q          # needs the model files in models/
```

## Configuration (env vars on the `api` service)
`MODEL_PATH`, `CONFIG_PATH`, `PRIORITY_HIGH_RATIO`, `MAX_UPLOAD_MB`, `GROQ_API_KEY`, `GROQ_MODEL`, `GROQ_TIMEOUT`.
`api/requirements.txt` pins `tensorflow-cpu==2.21.0` to match the Colab version that saved the model.

## Limitations
- Weak classes: Pneumonia (AUC 0.63, F1 0.07), Nodule (0.66), Fibrosis (0.71).
- Labels are text-mined and noisy; low precision overall.
- Val/test are ~56–60% *No Finding* vs 19% in train (distribution shift).
- No external validation; Grad-CAM shows attention, not causality.

## TODO before submission
- [ ] Export EDA/audit plots and Grad-CAM examples into `reports/`
- [ ] Add cost assumptions to the UI "Business insights" tab
- [ ] Justify priority ratio from validation data
- [ ] Update architecture diagram to match; test from a clean clone; record a fallback demo
- [ ] Team contribution summary
