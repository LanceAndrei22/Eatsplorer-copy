# Eatsplorer: Enhancing Food Tourism in Legazpi City Through a Conversational AI Chatbot Using Aspect-Based Sentiment Analysis

**Bicol University College of Science — Computer Science Department**
**Bachelor of Science in Computer Science | Undergraduate Thesis | May 2026**

**Authors:** Manuel James L. Baltasar · Lance Andrei C. Bolaños · Carl Edward L. San Jose

---

## Overview

Eatsplorer is a conversational AI chatbot that addresses fragmented dining discovery in Legazpi City, Philippines by delivering aspect-based restaurant recommendations grounded in real customer sentiment — going beyond generic star ratings. Users can query the chatbot in natural language to get restaurant recommendations filtered by cuisine type, specific dining aspects (food quality, service, ambiance, price/value), or signature dishes, and can retrieve live Google Maps reviews analyzed in real time.

The system integrates two core AI modules:

- **Fine-tuned DeBERTa V3 ABSA Model** — a pre-trained `deberta-v3-base-absa-v1.1` model fine-tuned on over 10,000 curated Legazpi City restaurant reviews using a class-weighted loss function and mixed-precision training. It classifies sentiment across five aspects (Food Quality, Service, Ambiance, Price/Value, Overall) with labels Positive, Neutral, Negative, and N/A. Achieved a peak Macro F1-score of **0.9699**.

- **RASA-Based Conversational Pipeline** — a RASA 3.x NLU pipeline using DIETClassifier with SpaCy featurizers, trained for intent classification (Macro F1: **0.8464**) and entity extraction (Macro F1: **0.8950**) over restaurant-domain conversational queries including temporal filtering.

The system achieved a **SUS score of 79.60/100** and an **87.1% overall satisfaction rate** during User Acceptance Testing.

📄 **Final Manuscript (PDF):** [View on Google Drive](https://drive.google.com/file/d/1ezr_2qf6hIpCt64aGN40nnOp36IX-qsW/view?usp=drive_link)

🌐 **Live Frontend Demo:** [eatsplorerchatbotfrontend.vercel.app](https://eatsplorerchatbotfrontend.vercel.app/)

🤗 **ABSA Model (Hugging Face):** [Ehemjayy/EATSPLORER_REVISED](https://huggingface.co/Ehemjayy/EATSPLORER_REVISED/tree/main)

🤗 **RASAModel (Hugging Face):** [Ehemjayy/Rasa_Chatbot_Model](https://huggingface.co/Ehemjayy/Rasa_Chatbot_Model/tree/main)

🚀 **ABSA Inference Space (Hugging Face Spaces):** [Ehemjayy/EATSPLORER](https://huggingface.co/spaces/Ehemjayy/EATSPLORER)

---

## System Architecture

```
React Frontend (Vercel / Localhost:5173)
        │
        ▼ HTTP
FastAPI Backend (Localhost:8000)  ──── SerpAPI (Live Google Maps Reviews)
        │                                      │
        ▼ HTTP                                 ▼ Gradio API
RASA Server (Localhost:5005)        HuggingFace Spaces (DeBERTa V3 ABSA)
        │
        ▼ HTTP
RASA Actions Server (Localhost:5055)
        │
        ▼ SQL
SQLite Database
```

> For public access during local development, the FastAPI and RASA servers are exposed via **Ngrok** tunnels, and the React frontend is pointed to those tunnel URLs.

---

## Dependencies and Versions

Two separate Python virtual environments are required due to a `pydantic` v1/v2 conflict between RASA and FastAPI.

### Python

| Requirement | Version |
|---|---|
| Python (both venvs) | `3.10.x` |

### `venv_chatbot` — RASA + Actions Server

| Package | Version |
|---|---|
| `rasa` | `3.6.20` |
| `spacy` | `3.8.7` |
| `en_core_web_lg` (spaCy model) | via `python -m spacy download en_core_web_lg` |

### `venv_web` — FastAPI Backend + ABSA Utilities

| Package | Version |
|---|---|
| `fastapi` | `0.121.0` |
| `uvicorn` | latest stable |
| `httpx` | latest stable |
| `transformers` | `5.1.0` |
| `torch` | `2.7.1+cu118` (GPU) or `2.7.1` (CPU) |
| `pandas` | `2.0.3` |
| `scikit-learn` | `1.1.3` |
| `emoji` | `2.15.0` |
| `gradio_client` | latest stable |
| `python-dotenv` | latest stable |

### Frontend

| Requirement | Version |
|---|---|
| Node.js | `18.x` or later |
| React | `18.3.1` |
| Vite | `5.3.4` |

---

## Demo-Only Quick Start (Windows)

The shortest path to running Eatsplorer locally for a live demo or screen recording. It skips everything that is not needed to *run* the system: model training, the dataset, Ngrok, and the local PyTorch/transformers stack (ABSA inference runs on the hosted Hugging Face Space instead). Verified on Windows with VS Code and PowerShell.

> **Folder names:** in this repository the folders are `chatbot_git`, `web_backend_git` and `web_frontend_git`. The full setup further below calls them `chatbot`, `backend` and `frontend`.

### What you need

**Files:** only the three folders above. You can ignore `Training Scripts and Dataset`.

| Requirement | Notes |
|---|---|
| Python **3.10** | For the chatbot. RASA 3.6 does not support 3.11+. Install alongside other versions, leave "Add to PATH" unchecked, and call it with `py -3.10`. Verify with `py -3.10 --version`. |
| Python 3.11 (or 3.10) | For the backend. The demo ran on 3.11 with current package versions. |
| Node.js 18+ | For the frontend. |
| Trained RASA model | Download the `.tar.gz` from the [Rasa_Chatbot_Model](https://huggingface.co/Ehemjayy/Rasa_Chatbot_Model/tree/main) repo. This lets you skip `rasa train`. |
| SerpAPI key | Optional. Only the live-reviews feature needs it. |
| Internet | The backend calls the Hugging Face Space and SerpAPI. |

### 1. Place the trained model

Put the model file, **without extracting it**, in a new `models` folder:

```
chatbot_git/
└── models/
    └── 20260512-143647-corn-segue.tar.gz
```

If you would rather build it yourself, run `rasa train` inside `chatbot_git` (several minutes) instead.

### 2. Edit the code

1. `web_backend_git/hf_absa.py`: replace the placeholder with the Space ID.
   ```python
   SPACE_ID = "Ehemjayy/EATSPLORER"
   ```
2. `web_backend_git/live_reviews.py`: paste your SerpAPI key into `SERPAPI_API_KEY`. In this copy of the code the key is read from this line, **not** from `.env`. Never commit or screen-share this file with a real key in it.
3. *Only if your network intercepts HTTPS* (you see `CERTIFICATE_VERIFY_FAILED`): add these two lines to the top of `web_backend_git/main.py`, after the opening docstring and before `from fastapi import ...`.
   ```python
   import truststore
   truststore.inject_into_ssl()
   ```

No `.env` file is needed for the demo. The backend defaults to RASA at `http://localhost:5005`.

### 3. One-time setup

Run from the project root (the folder containing the three project folders).

**Chatbot environment (Python 3.10):**

```powershell
py -3.10 -m venv venv_chatbot
venv_chatbot\Scripts\activate
pip install "setuptools==65.5.0" wheel
pip install rasa==3.6.20 spacy==3.8.7 pandas requests
pip install "setuptools==80.9.0"
python -m spacy download en_core_web_lg
deactivate
```

The `setuptools` pins matter. A brand-new `setuptools` breaks a RASA dependency build, and one that is too new no longer ships `pkg_resources`, which RASA imports. The install takes 5 to 15 minutes, and the spaCy model is about 400 MB.

**Backend environment:**

```powershell
py -3.11 -m venv venv_web
venv_web\Scripts\activate
pip install fastapi uvicorn httpx python-dotenv pandas gradio_client
deactivate
```

`torch`, `transformers`, `scikit-learn` and `emoji` are not needed here. ABSA runs on the Hugging Face Space.

**Frontend:**

```powershell
cd web_frontend_git
npm install
cd ..
```

The `npm audit` vulnerability warnings can be ignored for a local demo. Do not run `npm audit fix --force`.

### 4. Launch everything with one file

Before launching, open the [Hugging Face Space](https://huggingface.co/spaces/Ehemjayy/EATSPLORER) and wait for it to say "Running". Free Spaces go to sleep, and live reviews fail while it is asleep.

Save this as `start_eatsplorer.bat` in the project root and double-click it. If your model has a different filename, change it in the first line.

```bat
@echo off
set "ROOT=%~dp0"

start "RASA server" /D "%ROOT%chatbot_git" cmd /k ""%ROOT%venv_chatbot\Scripts\rasa.exe" run --enable-api --cors * --port 5005 --model models\20260512-143647-corn-segue.tar.gz"

start "RASA actions" /D "%ROOT%chatbot_git" cmd /k ""%ROOT%venv_chatbot\Scripts\rasa.exe" run actions --port 5055"

start "Backend" /D "%ROOT%web_backend_git" cmd /k ""%ROOT%venv_web\Scripts\python.exe" -m uvicorn main:app --port 8000"

start "Frontend" /D "%ROOT%web_frontend_git" cmd /k "npm run dev"

timeout /t 25 >nul
start http://localhost:5173
```

Four windows open and the browser loads `http://localhost:5173` after about 25 seconds. RASA takes about a minute to finish loading, so wait before the first chat message. To stop everything, close the four windows.

| Window | Port | Ready when it shows |
|---|---|---|
| RASA server | 5005 | `Rasa server is up and running` |
| RASA actions | 5055 | `Action endpoint is up and running on http://0.0.0.0:5055` |
| Backend | 8000 | `connected to space: Ehemjayy/EATSPLORER` |
| Frontend | 5173 | `VITE ... ready` |

Quick check: `http://localhost:8000/api/health` should report 89 restaurants loaded.

### 5. Demo walkthrough

1. **Restaurant Explorer (left panel):** use **Sort by aspect** to re-rank by Service, Food Quality and so on. Switch **All Time** to **By Month** and pick a month.
2. **Restaurant card:** shows the five aspect scores and the number of reviews behind them.
3. **Chatbot (right panel):** click the suggestion chips, for example *"Top 3 restaurants with best food quality"* and *"Budget-friendly café with great ambiance"*.
4. **Live reviews:** click *"Latest review for Starbucks"* or the **🔴 Live Reviews** button on a card. Reviews are fetched from Google Maps through SerpAPI, and each one is scored per aspect by the ABSA model on Hugging Face. This takes a few seconds.

**Data coverage:** the monthly scores run from **January 2024 to March 2026**.

### 6. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `CERTIFICATE_VERIFY_FAILED` during `pip install` | Your network intercepts HTTPS. Pip retries and usually gets through. |
| `CERTIFICATE_VERIFY_FAILED` / `failed to connect to gradio space` in the backend | Add the `truststore` lines from step 2, or try a different network such as a phone hotspot. |
| `canonicalize_version() got an unexpected keyword argument 'strip_trailing_zero'` while installing RASA | `setuptools` is too new. Run the `setuptools==65.5.0` and `wheel` install first, as in step 3. |
| `ModuleNotFoundError: No module named 'pkg_resources'` when starting RASA | In `venv_chatbot`, run `pip install "setuptools==80.9.0"`. |
| Live reviews hang or error | The Hugging Face Space is asleep (open its page and wait), `SPACE_ID` is wrong, or the SerpAPI key is missing or invalid. |
| *"I don't have enough data for \<month\> yet"* | "Right now", "this month" and "currently" route to the best-by-month feature and fall back to the previous month. Ask without those words, e.g. *"Best restaurants in Legazpi City overall"*, or choose a month up to March 2026. |
| Chat error on the first message | RASA is still loading. Wait about a minute and retry. |
| Port already in use | An old server is still running. Close all windows, or find the process with `netstat -ano \| findstr "5005 5055 8000 5173"` and stop it with `taskkill /PID <id> /F`. |
| New VS Code terminals show `(venv_chatbot)` | Harmless. VS Code auto-activates the selected interpreter. The `.bat` file does not depend on it. |

### 7. Cleaning up

Close all windows and VS Code first, then run these in a normal PowerShell window (not inside VS Code).

```powershell
# Delete the project, including both virtual environments and node_modules
cd C:\Users\<you>\Desktop
Remove-Item -Recurse -Force Eatsplorer

# Delete leftover caches
Remove-Item "$env:LOCALAPPDATA\pip\Cache" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "$env:LOCALAPPDATA\npm-cache" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "$env:TEMP\*" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "$env:USERPROFILE\.config\rasa" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "$env:USERPROFILE\.matplotlib" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "$env:USERPROFILE\.cache\huggingface" -Recurse -Force -ErrorAction SilentlyContinue
```

Then delete the model `.tar.gz` and any ZIP from Downloads, empty the Recycle Bin, and uninstall Python 3.10 (and Node.js, if you installed it only for this) from Settings > Apps.

To keep the source but free the space, delete only `venv_chatbot`, `venv_web` and `web_frontend_git\node_modules`. Virtual environments cannot be moved, so rebuild them with step 3 if you need them again.

---

## Local Installation and Deployment

> **Only need to run it for a demo?** Use the [Demo-Only Quick Start (Windows)](#demo-only-quick-start-windows) above. The steps below are the full developer setup, including model training and Ngrok. Replace `chatbot`, `backend` and `frontend` in the commands with the actual folder names `chatbot_git`, `web_backend_git` and `web_frontend_git`.

### Prerequisites

- Python 3.10 installed and available as `python3.10`
- Node.js 18+ and npm installed
- [Ngrok](https://ngrok.com/) installed and authenticated (for public tunneling)
- A [SerpAPI](https://serpapi.com/) API key (for the live reviews feature)

---

### 1. Clone the Repository

```bash
git clone https://github.com/<your-org>/eatsplorer.git
cd eatsplorer
```

---

### 2. Set Up Environment Variables

Create a `.env` file in the backend root directory:

```env
SERPAPI_API_KEY=your_serpapi_key_here
RASA_URL=http://localhost:5005
HF_SPACE_URL=https://ehemjayy-eatsplorer.hf.space
```

---

### 3. Set Up the RASA Virtual Environment (`venv_chatbot`)

```bash
# Create and activate the RASA venv
python3.10 -m venv venv_chatbot
source venv_chatbot/bin/activate        # Windows: venv_chatbot\Scripts\activate

# Install RASA and SpaCy
pip install rasa==3.6.20
pip install spacy==3.8.7
python -m spacy download en_core_web_lg

deactivate
```

---

### 4. Set Up the FastAPI Virtual Environment (`venv_web`)

```bash
# Create and activate the web/backend venv
python3.10 -m venv venv_web
source venv_web/bin/activate            # Windows: venv_web\Scripts\activate

# Install backend dependencies
pip install fastapi==0.121.0 uvicorn httpx python-dotenv
pip install transformers==5.1.0
pip install torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
# For CPU-only machines:
# pip install torch==2.7.1
pip install pandas==2.0.3 scikit-learn==1.1.3 emoji==2.15.0 gradio_client

deactivate
```

---

### 5. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

---

### 6. Running the System Locally

All four components must be running simultaneously. Open four separate terminal windows.

**Terminal 1 — RASA NLU/Dialogue Server**

```bash
cd chatbot
source ../venv_chatbot/bin/activate     # Windows: ..\venv_chatbot\Scripts\activate
rasa run --enable-api --cors "*" --port 5005
```

**Terminal 2 — RASA Custom Actions Server**

```bash
cd chatbot
source ../venv_chatbot/bin/activate
rasa run actions --port 5055
```

**Terminal 3 — FastAPI Backend**

```bash
cd backend
source ../venv_web/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 4 — React Frontend (Development Server)**

```bash
cd frontend
npm run dev
```

The frontend will be accessible at `http://localhost:5173`.

---

### 7. Exposing the Backend via Ngrok (Optional — for Remote Access)

If you need to access the chatbot from outside your local machine (e.g., from the deployed Vercel frontend), expose both the FastAPI and RASA servers via Ngrok tunnels:

```bash
# Expose FastAPI backend
ngrok http 8000

# In a separate terminal, expose the RASA server
ngrok http 5005
```

Update the `.env` file (and Vercel environment variables for the deployed frontend) with the generated Ngrok URLs.

---

### 8. Retraining the RASA Pipeline

After modifying training data in `chatbot/data/` (NLU, stories, rules) or the domain in `chatbot/domain.yml`:

```bash
cd chatbot
source ../venv_chatbot/bin/activate
rasa train
```

The newly trained model will be saved to `chatbot/models/`. Restart the RASA server to load it.

---

## Fine-Tuned DeBERTa V3 ABSA Model

### Model Details

- **Base Model:** `yangheng/deberta-v3-base-absa-v1.1`
- **Task:** Aspect Category Sentiment Analysis (ACSA)
- **Aspects:** Food Quality, Service, Ambiance, Price/Value, Overall
- **Labels:** Positive, Neutral, Negative, N/A
- **Training Configuration:** Class-weighted cross-entropy loss, mixed-precision (fp16), 5-fold cross-validation
- **Peak Macro F1:** 0.9699

### Hugging Face Links

- **Model Repository:** [Ehemjayy/EATSPLORER_REVISED](https://huggingface.co/Ehemjayy/EATSPLORER_REVISED/tree/main)
- **Inference Space (Gradio API):** [Ehemjayy/EATSPLORER](https://huggingface.co/spaces/Ehemjayy/EATSPLORER)

---

### Using the Hosted Inference Space

The easiest way to run ABSA inference is through the Gradio API hosted on Hugging Face Spaces. No local GPU is required.

```python
from gradio_client import Client

client = Client("Ehemjayy/EATSPLORER")

result = client.predict(
    review_text="The food was amazing but the service was a bit slow.",
    rating=4.0,
    api_name="/infer_absa"
)

# result is a dict of aspect -> {"sentiment": str, "score": float}
print(result)
```

---

### Running Inference Locally

To run inference locally using the downloaded model weights:

```bash
source venv_web/bin/activate
python infer_restaurants.py --input data/reviews.csv --output data/scored_reviews.csv
```

Or for a single review:

```bash
python score_reviews.py --text "Great sinigang, very affordable." --rating 5
```

---

### Retraining / Fine-Tuning the Model

To fine-tune the DeBERTa V3 model on new or updated review data:

1. Prepare your dataset in the ACSA format (see `data/eatsplorer_master.csv` for reference structure — columns: `text`, `rating`, `food_quality`, `service`, `ambiance`, `price_value`, `overall`).

2. Activate the web venv and run the training script:

```bash
source venv_web/bin/activate
python train_absa.py \
  --data_path data/eatsplorer_master.csv \
  --model_output models/deberta_absa_finetuned \
  --epochs 5 \
  --batch_size 16 \
  --fp16
```

3. GPU with at least 4 GB VRAM is recommended (NVIDIA GTX 1660 Ti or better). CPU training is supported but significantly slower.

4. After training, update the model on Hugging Face using the `huggingface_hub` CLI:

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload Ehemjayy/EATSPLORER_REVISED ./models/deberta_absa_finetuned
```

---

## Citation

> **Note:** This section will be finalized once the final edited manuscript has been officially uploaded. Please use the placeholder below in the meantime.

If you use Eatsplorer, its dataset, or the fine-tuned ABSA model in academic work, please cite:

**APA:**

> Baltasar, M. J. L., Bolaños, L. A. C., & San Jose, C. E. L. (2026). *Eatsplorer: Enhancing Food Tourism in Legazpi City Through a Conversational AI Chatbot Using Aspect-Based Sentiment Analysis* [Unpublished undergraduate thesis]. Bicol University College of Science.

**BibTeX:**

```bibtex
@thesis{eatsplorer2026,
  author    = {Baltasar, Manuel James L. and Bola{\~n}os, Lance Andrei C. and {San Jose}, Carl Edward L.},
  title     = {Eatsplorer: Enhancing Food Tourism in {Legazpi City} Through a Conversational {AI} Chatbot Using Aspect-Based Sentiment Analysis},
  school    = {Bicol University College of Science},
  year      = {2026},
  type      = {Unpublished Undergraduate Thesis},
  note      = {Manuscript available at: \url{https://drive.google.com/file/d/1ezr_2qf6hIpCt64aGN40nnOp36IX-qsW/view?usp=drive_link}}
}
```

---

## License

This project is released for academic and non-commercial use. For other uses, please contact the authors.
