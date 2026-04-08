# Phase 1: Deep Planning & Architecture Review

## Current Limitations (The "Why")
The current stack contains disjointed models (`OutcomeModel` [MLP], `ContextualStackedLSTM` [Sequence], `SGMTransformer` [Set/Graph]). 
- **Flaw 1:** Disconnected representations. Player embeddings learned for outcome prediction don't benefit from the temporal sequence learning in the LSTM.
- **Flaw 2:** Deployment complexity. Loading 3 `.pth` files with 3 separate encoders is not "drag and drop".
- **Flaw 3:** Outdated sequence modeling. LSTMs are eclipsed by Transformers/State-Space Models (SSMs like Mamba) for sequence modeling with long-range dependencies.

## SOTA Academic Target (The "What")
We will implement an **NRLOmniModel** based on DeepMind's Perceiver IO or a Multi-Task Transformer architecture:
- **Unified Heterogeneous Inputs:** A single trunk that accepts categorical scalars (teams, venues), unordered sets (the 34 players), and temporal sequences (play-by-play history) by mapping them all into a shared latent space.
- **Multi-Task Heads:** From the shared latent trunk, separate heads output:
  1. Win/Margin/Points (Scalar regression/classification)
  2. Next Play & Expected Meters (Autoregressive sequence decoding)
  3. SGM Try Scorers (Set-based classification)
- **Single-File Distillation:** We will wrap the model, its tokenizer/encoders, and its inference logic inside a single `torch.jit.ScriptModule` or an exported `.onnx` file. This means the final deliverable is literally ONE `.pt` or `.onnx` file that has no dependencies on the `models.py` source code to run inference.

## Ultra Planner Ralph Loop Milestones
1. **Scaffold NRLResearchLab:** Create a unified `Config` (dataclass) and `NRLOmniModel` stub.
2. **Unify Data Pipeline:** Merge `dataset.py` classes into an `OmniDataset` that yields `(context_tokens, sequence_tokens, targets)`.
3. **Implement OmniModel (Transformer/Perceiver):** Write the shared self-attention trunk and multi-task heads.
4. **Implement Distillation/Export:** Write `export.py` that traces the PyTorch model and bundles the encoders into a standalone TorchScript/ONNX artifact.
5. **API Refactor:** Rewrite `api.py` and `simulate_match.py` to simply load `NRL_OmniModel.pt` and call `model.predict_all()`.
