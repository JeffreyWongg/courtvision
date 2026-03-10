# CourtVision

CourtVision is a small end-to-end sandbox for **basketball court-state interaction + ML-style inference**:

- A **React** web UI where you place **5 offensive players, 5 defensive players, and a ball** on an NBA half court.
- A **FastAPI** backend that exposes:
  - **`POST /api/v1/predict`** — returns “next-frame” player coordinates (currently a **stub** predictor unless a real model is provided).
  - **`POST /api/v1/match`** — matches a trajectory against a tracking-play catalog (currently uses a **stub** dataset unless a JSONL dataset is provided).

The repo also includes:

- `data-generation/`: a procedural **JSONL dataset generator** for basketball tactics prompt/completion pairs (LLM fine-tuning format).
- `model-training/`: a script to train **LoRA adapters** for a Llama-style causal LM on a JSONL dataset.

---

## Brief summary

Use the UI to create a 10-player court state, send it to the API, and visualize the returned predicted positions. Optionally, run “motion matching” against a play-trajectory dataset.

---

## How it works

### Frontend (`frontend-web/`)

- Built with Vite + React + TypeScript.
- You place tokens via an **11-click flow** (offense → defense → ball), then you can **drag** tokens to adjust positions.
- Clicking **Run Predict** sends the current 10-player state to `POST /api/v1/predict`.
- The UI updates the on-court dots to the returned coordinates and displays the backend’s `model_version`.

**Coordinate system**

- Backend speaks **full-court** NBA coordinates: \(x \in [0,94]\), \(y \in [0,50]\) (feet).
- Frontend renders a **half-court** view: \(x \in [0,47]\), \(y \in [0,50]\).
- The frontend scales half-court \(x\) to full-court \(x\) when calling the API, and scales back when rendering predictions.

### Backend (`backend-api/`)

- FastAPI app with two routers:
  - `POST /api/v1/predict`: validates a `CourtStateRequest` (5 offense + 5 defense) and returns a `PredictionResponse`.
  - `POST /api/v1/match`: validates a `MatchRequest` (list of `[dx, dy]` displacement vectors) and returns a `MatchResponse`.
- Dependency injection creates singleton service instances:
  - `PredictionService` reads `COURTVISION_MODEL_PATH` (optional).
  - `MotionMatcher` reads `COURTVISION_DATASET_PATH` (optional).

**Current behavior (stubs)**

- If `COURTVISION_MODEL_PATH` is not set (or the file doesn’t exist), `PredictionService` uses a stub predictor: a small random displacement per player, clipped to court bounds.
- If `COURTVISION_DATASET_PATH` is not set (or missing), `MotionMatcher` uses a small in-memory stub play catalog.

---

## Tech stack

- **Frontend**
  - React + TypeScript + Vite
  - `framer-motion` (animated SVG)
  - `@use-gesture/react` (drag interactions)
- **Backend**
  - FastAPI + Uvicorn
  - Pydantic v2 (request/response schemas + validation)
  - NumPy (stub model + placeholders for future inference)
- **Data/ML utilities**
  - `data-generation/`: pure Python dataset generation + tests
  - `model-training/`: Hugging Face Transformers + PEFT (LoRA) + Datasets + PyTorch (training script)

---

## How to run it (local dev)

### 1) Start the backend API

From the repo root:

```bash
cd backend-api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

API docs (Swagger UI): `http://localhost:8000/docs`

Optional environment variables:

```bash
export COURTVISION_MODEL_PATH="/absolute/path/to/your/model.npy"
export COURTVISION_DATASET_PATH="/absolute/path/to/plays.jsonl"
```

Expected dataset format for `COURTVISION_DATASET_PATH` (JSONL):

```json
{"play_id":"...","description":"...","trajectory":[[1.0,0.5],[2.1,0.4]]}
```

### 2) Start the frontend

In a second terminal:

```bash
cd frontend-web
npm install
npm run dev
```

Open the URL Vite prints (typically `http://localhost:5173`).

Notes:
- The frontend calls `/api/v1/predict` and relies on Vite’s dev proxy to forward `/api/*` → `http://localhost:8000`.

---

## How to use the app

- Click the court to place:
  - 5 offense players
  - 5 defense players
  - 1 ball
- Drag any dot to adjust (after placement is complete).
- Click **Run Predict** to fetch predicted next positions from the backend.
- Use **Reset Court** to start over.

---

## Data generation (optional)

To generate an example JSONL dataset for LLM fine-tuning:

```bash
cd data-generation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m unittest discover -s tests -v
python3 generate_dataset.py -o basketball_tactics_dataset.jsonl -n 100 --seed 42
```

Each JSONL line contains:
- `prompt`: a formatted court state + question
- `completion`: a tactic recommendation (procedural baseline)

---

## Model training (optional)

`model-training/train_lora_llama3.py` trains LoRA adapters for a causal LM on a JSONL dataset.

At minimum, it expects:
- `--model_name_or_path` (a local path or HF model id you have access to)
- `--dataset_path` (a JSONL file)

Example (edit for your environment/GPU setup):

```bash
python3 model-training/train_lora_llama3.py \
  --model_name_or_path <base-model> \
  --dataset_path <dataset.jsonl> \
  --output_dir outputs/lora-llama3
```

---

## Credits

- **CourtVision**: project code and structure in this repository.
- **Open-source libraries**:
  - Frontend: React, Vite, framer-motion, @use-gesture/react
  - Backend: FastAPI, Uvicorn, Pydantic, NumPy
  - Training: PyTorch, Hugging Face Transformers, PEFT, Datasets

