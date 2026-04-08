import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from nrl_ml.omni_model import NRLOmniModel, OmniModelConfig


class DummyOmniDataset(Dataset):
    def __init__(
        self,
        num_samples=100,
        seq_len=50,
        num_teams=20,
        num_venues=30,
        num_players=1000,
        vocab_size=20,
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_teams = num_teams
        self.num_venues = num_venues
        self.num_players = num_players
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        cat_x = torch.tensor(
            [
                torch.randint(0, self.num_teams, (1,)).item(),
                torch.randint(0, self.num_teams, (1,)).item(),
                torch.randint(0, self.num_venues, (1,)).item(),
            ],
            dtype=torch.long,
        )

        roster_x = torch.randint(0, self.num_players, (2, 17), dtype=torch.long)
        global_cont = torch.rand(4, dtype=torch.float32)
        player_cont = torch.rand(2, 17, 3, dtype=torch.float32)

        seq_x = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        seq_context = torch.rand(self.seq_len, 3, dtype=torch.float32)

        y_win = torch.rand(1, dtype=torch.float32)
        y_margin = torch.randn(1, dtype=torch.float32) * 10
        y_points = torch.randn(1, dtype=torch.float32) * 40 + 30
        y_try = torch.rand(34, dtype=torch.float32)

        y_next_event = torch.randint(
            0, self.vocab_size, (self.seq_len,), dtype=torch.long
        )
        y_gain = torch.randn(self.seq_len, 1, dtype=torch.float32) * 10

        return (
            cat_x,
            roster_x,
            global_cont,
            player_cont,
            seq_x,
            seq_context,
            y_win,
            y_margin,
            y_points,
            y_try,
            y_next_event,
            y_gain,
        )


def train():
    print("Setting up OmniModel Training Pipeline...")
    config = OmniModelConfig(
        num_teams=20,
        num_venues=30,
        num_players=1000,
        vocab_size=20,
        embed_dim=128,
        n_heads=4,
        num_layers=2,
        max_seq_len=200,
        global_cont_dim=4,
        player_cont_dim=3,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NRLOmniModel(config).to(device)

    dataset = DummyOmniDataset()
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    bce = nn.BCELoss()
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    epochs = 2
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            (
                cat_x,
                roster_x,
                global_cont,
                player_cont,
                seq_x,
                seq_context,
                y_win,
                y_margin,
                y_points,
                y_try,
                y_next_event,
                y_gain,
            ) = [t.to(device) for t in batch]

            optimizer.zero_grad()

            (
                win_prob,
                margin,
                total_points,
                try_probs,
                next_event_logits,
                expected_gain,
            ) = model(cat_x, roster_x, global_cont, player_cont, seq_x, seq_context)

            loss_win = bce(win_prob, y_win)
            loss_margin = mse(margin, y_margin)
            loss_points = mse(total_points, y_points)
            loss_try = bce(try_probs, y_try)

            # Sequence losses
            loss_event = ce(
                next_event_logits.view(-1, config.vocab_size), y_next_event.view(-1)
            )
            loss_gain = mse(expected_gain, y_gain)

            loss = (
                loss_win
                + (0.1 * loss_margin)
                + (0.1 * loss_points)
                + loss_try
                + loss_event
                + (0.1 * loss_gain)
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(loader):.4f}")

    print("\nTraining complete! Distilling to single TorchScript file...")
    model.eval()
    # Move back to CPU for standard export/inference without requiring GPU on load unless specified
    model = model.cpu()
    scripted_model = torch.jit.script(model)
    os.makedirs("dist", exist_ok=True)
    scripted_model.save("dist/NRL_OmniModel_SOTA.pt")
    print("Saved SOTA OmniModel to dist/NRL_OmniModel_SOTA.pt")


if __name__ == "__main__":
    train()
