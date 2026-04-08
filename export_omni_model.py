import torch
import os
from nrl_ml.omni_model import NRLOmniModel, OmniModelConfig

def export_to_single_file():
    print("Setting up SOTA OmniModel Configuration...")
    config = OmniModelConfig(
        num_teams=20,
        num_venues=30,
        num_players=1000,
        vocab_size=20,
        embed_dim=128,
        max_seq_len=200,
        global_cont_dim=4,
        player_cont_dim=3
    )
    
    model = NRLOmniModel(config)
    model.eval()
    
    print("Tracing the model with TorchScript to package Python logic inside the .pt file...")
    scripted_model = torch.jit.script(model)
    
    os.makedirs('dist', exist_ok=True)
    export_path = 'dist/NRL_OmniModel_SOTA.pt'
    
    scripted_model.save(export_path)
    print(f"\n[SUCCESS] Model successfully distilled into a SINGLE drag-and-drop file: {export_path}")
    print("You can now load this anywhere using: model = torch.jit.load('NRL_OmniModel_SOTA.pt')")
    print("Zero source code dependencies required.")

if __name__ == '__main__':
    export_to_single_file()
