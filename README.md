<div align="center">
  
# NRL OmniModel
**State-of-the-Art Multi-Modal Transformer for National Rugby League Prediction**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.0+-61dafb?style=for-the-badge&logo=react)](https://reactjs.org/)
[![TailwindCSS](https://img.shields.io/badge/Tailwind-3.0+-38b2ac?style=for-the-badge&logo=tailwind-css)](https://tailwindcss.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

## 📌 Abstract

The **NRL OmniModel** is a research-grade machine learning system designed to predict National Rugby League (NRL) outcomes, generate Ladbrokes-style Same Game Multis (SGMs), and simulate play-by-play match sequences. It evolves previous disjointed multi-layer perceptrons (MLPs) and LSTMs into a unified, Perceiver-inspired Transformer architecture.

By fusing continuous match statistics (Elo, fatigue, weather), categorical embeddings (venues, weather states), and sequential/set-based data (rosters, recent form), the OmniModel learns complex non-linear interactions between modalities. The entire model pipeline is distilled into a single, zero-dependency `NRL_OmniModel_SOTA.pt` TorchScript file for high-performance inference via a FastAPI backend and a modern shadcn-styled web UI.

---

## 🔬 Architecture

The model utilizes a fused Multi-Modal attention trunk, heavily inspired by DeepMind's Perceiver IO. It processes diverse inputs into a unified latent space, projecting it across multiple specialized task heads.

```mermaid
graph TD
    %% Input Modalities
    C[Continuous Features: Elo, Fatigue] --> EmbedC[Linear Projection]
    Cat[Categorical Features: Weather, Venue] --> EmbedCat[Entity Embeddings]
    R[Set Features: Player Rosters] --> EmbedR[Set Transformer]
    S[Sequential Data: Form, Play-by-Play] --> EmbedS[Temporal Attention]

    %% Fusion Trunk
    EmbedC --> Concat((Latent Concat))
    EmbedCat --> Concat
    EmbedR --> Concat
    EmbedS --> Concat

    Concat --> TransTrunk[Multi-Head Self-Attention Trunk<br/>(Transformer Encoder Layers)]

    %% Task Heads
    TransTrunk --> Head1[Match Outcome Head]
    TransTrunk --> Head2[Play-by-Play Sequence Head]
    TransTrunk --> Head3[SGM / Odds Generation Head]

    %% Outputs
    Head1 --> Out1(Win Probability & Margin)
    Head2 --> Out2(Play-by-Play Simulation Loop)
    Head3 --> Out3(Ladbrokes-style SGM Odds)
```

### Key Innovations
1. **Unified Modality Fusion**: Eliminates the need for separate models. Continuous, categorical, and sequential inputs are embedded into a shared latent dimension ($d_{model} = 256$).
2. **Set-Based Roster Processing**: Uses permutation-invariant attention to evaluate team strength regardless of player input order.
3. **Zero-Dependency Distillation**: The entire PyTorch model is traced and scripted via `torch.jit.script`, resulting in a standalone `dist/NRL_OmniModel_SOTA.pt` file requiring only `libtorch` or PyTorch runtime.

---

## 🚀 Quickstart

### 1. Environment Setup

It is recommended to run this project in an isolated virtual environment with Python 3.10+.

```bash
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Training and Distillation

Train the OmniModel on the multimodal dataset and distill it into a TorchScript object.

```bash
# Trains the model using omni_dataset.py logic
python train_omni.py

# Distills the trained weights into a standalone .pt file
python export_omni_model.py
```

### 3. Serving the API & UI

The repository includes a highly-optimized FastAPI backend that serves the `.pt` file, and a modern frontend dashboard.

```bash
# Start the uvicorn server
python api.py
```
**Access the Web UI**: Open your browser to `http://localhost:8000`

---

## 📊 Endpoints Overview

The FastAPI backend exposes the following zero-shot and sequence endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | `POST` | Returns win probability and margin based on continuous and categorical inputs. |
| `/api/simulate` | `POST` | Autoregressively generates a play-by-play text sequence using the Sequence Head. |
| `/api/sgm` | `POST` | Evaluates multimodal context to generate reasonable Same Game Multi bets. |
| `/api/info` | `GET` | Returns model architecture and health metadata. |

---

## 📁 Repository Structure

```text
nrlgpt/
├── api.py                    # FastAPI server & route definitions
├── export_omni_model.py      # TorchScript distillation logic
├── train_omni.py             # Main training loop
├── test_e2e.py               # End-to-end API & model validation
├── nrl_ml/
│   ├── omni_model.py         # PyTorch Multi-Modal Transformer definition
│   └── omni_dataset.py       # Dataloaders for fused continuous/categorical stats
├── static/
│   └── index.html            # Modern shadcn/Tailwind Web Dashboard
├── dist/                     # Target directory for exported .pt models
└── README.md                 # Scientific Documentation
```

---

## 🧪 Testing & Validation

The framework includes an end-to-end testing suite validating the computational graph, distillation integrity, and API response schema.

```bash
pytest test_e2e.py -v
```

---

## 📜 License
This project is licensed under the MIT License. See the `LICENSE` file for details.
