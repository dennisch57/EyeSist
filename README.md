# EyeSist

Eye gaze classification system with per-user Ridge Regression calibration and an automated MLOps retraining pipeline.

## Architecture overview

```
Backend
  └─ YOLO eye detection → crop eyes
  └─ POST /calibrate  → per-user Ridge Regression (fitted in ~40ms)
  └─ POST /predict    → gaze direction (6 classes: closed, down, left, right, straight, up)

Azure Blob Storage ("eyesist" container)
  ├─ calibration-data/{session_id}/images/{label}_{n}.jpg
  ├─ calibration-data/{session_id}/meta.json
  ├─ models/backbone/ethxgaze_backbone.pth          ← production model
  ├─ models/backbone/candidate/ethxgaze_candidate_*.pth
  ├─ models/backbone/archive/ethxgaze_backbone_*.pth
  └─ models/backbone/promotion_log.json

ClearML (nightly at 2am UTC)
  └─ pipeline: check_retrain → train_model → evaluate_promote
```

## Backend

### Setup

```bash
cd backend
pip install -r requirements.txt
```

Set environment variables:

| Variable | Default | Description |
|---|---|---|
| `AZURE_STORAGE_ACCOUNT_NAME` | `anonifyme` | Azure storage account |
| `EYESIST_DEVICE` | `auto` | PyTorch device: `auto`, `cpu`, `cuda`, `mps` |

### Run

```bash
uvicorn main:app --reload
```

### API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/calibrate` | Fit per-user Ridge model. Body: `{crops: {label: [base64, ...]}}` |
| `POST` | `/predict` | Predict gaze direction. Body: `{image: base64, session_id?: string}` |
| `GET` | `/health` | Health check |

### Model

- **Backbone**: ETH-XGaze pretrained ResNet-50 (outputs 2048-d feature vector, `fc` layer never called)
- **Head**: configurable dense layers → `Linear(2048, 6)` (best config uses no intermediate layers)
- **Per-user calibration**: Ridge Regression (`alpha=1.0`, `StandardScaler`, `class_weight="balanced"`) fitted on backbone features — ~40ms fit time
- **Session storage**: Ridge model saved locally at `/tmp/eyesist_sessions/<uuid>.pkl`; 

## MLOps pipeline

### Overview

Retraining is triggered automatically when the base model's per-session accuracy degrades. It never runs from a user calibration — calibration is always Ridge-only. Retraining rebuilds the full ResNet-50 + head from the current production weights.

### Trigger condition

2 or more user sessions where the base model pre-calibration accuracy falls below 50%.

### Pipeline steps

```
pipeline_controller.py
  ├─ step 1: check_retrain
  │    - Scans all session meta.json in Azure
  │    - If 2+ sessions below 50%: assigns permanent train/test splits (80/20)
  │    - Returns (needs_retrain, train_session_ids)
  │
  ├─ step 2: train_model  (only runs if needs_retrain=True)
  │    - Loads production weights from Azure as starting point
  │    - Combines original TRAIN_DIR + Azure user session images (split=train)
  │    - Phase 1: freeze backbone, train head only
  │    - Phase 2: unfreeze from layer3, fine-tune with differential LR
  │    - Uploads candidate checkpoint to Azure
  │
  └─ step 3: evaluate_promote
       - Evaluates candidate vs production on original TEST_DIR + user test sessions
       - Promotes if: candidate_test_acc > production_test_acc + 1% AND > 60% floor
       - Archives old backbone before overwriting production
       - Logs every promotion decision to models/backbone/promotion_log.json
```

### Data split strategy

| Data source | Splits |
|---|---|
| Original dataset | Fixed `train` / `val` / `test` — never reassigned |
| User sessions | `train` or `test` only — assigned permanently at first retrain trigger |
| Val set | Original only — user sessions never go into val |

### Running the pipeline

**Local debug (all steps run in-process):**
```bash
cd backend/pipeline
python pipeline_controller.py --run-local
```

**Enqueue on ClearML agents:**
```bash
python pipeline_controller.py --run-remote
```

**Register nightly scheduler (run once to activate):**
```bash
python pipeline_controller.py --schedule --cron "0 2 * * *"
```

### ClearML agent setup

Each agent machine needs:
1. ClearML credentials configured (`clearml-agent init`)
2. Backend dependencies installed (`pip install -r backend/requirements.txt`)
3. `EYESIST_BACKEND_DIR` env var pointing to the `backend/` directory
4. `AZURE_STORAGE_ACCOUNT_NAME` env var set
5. Agent running in the `default` queue: `clearml-agent daemon --queue default`

### Promotion logic

| Condition | Result |
|---|---|
| candidate test acc ≥ production test acc + 5% AND ≥ 60% | Promoted — overwrites production blob, archives old |
| candidate test acc < production test acc + 5% | Not promoted — not meaningfully better |
| candidate test acc < 60% | Not promoted — below quality floor |

After promotion the FastAPI server clears its in-memory model singleton and reloads from Azure on the next request.

## Project structure

```
backend/
├── main.py                        # FastAPI app
├── model.py                       # GazeClassifier, ResNet-50, feature extraction
├── eye_detector.py                # YOLO eye cropping
├── azure_storage.py               # Azure Blob Storage helpers
├── runtime_config.py              # Device selection (EYESIST_DEVICE)
├── training_config.py             # LABELS, RETRAIN_EXPERIMENT_CONFIG, SEED
├── requirements.txt
└── pipeline/
    ├── pipeline_controller.py     # ClearML PipelineDecorator + TaskScheduler
    ├── step_check_retrain.py      # Check trigger condition, assign splits
    ├── step_train_base_model.py   # Retrain from production weights
    └── step_evaluate_promote.py   # Evaluate candidate, promote if better

frontend/
└── ...

Experiments/
└── ETHGaze_Transfer_Learning.ipynb  # Original training notebook (reference)
```
