import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

from nrl_ml.models import SGMTransformer


# Dummy Dataset to mimic Ladbrokes SGM data structure
class DummySGMDataset(Dataset):
    def __init__(self, num_samples=1000, num_teams=20, num_venues=15, num_players=500):
        self.num_samples = num_samples
        self.num_teams = num_teams
        self.num_venues = num_venues
        self.num_players = num_players

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

        # Targets: Win Prob, Margin, Total Points, Try Scorers (17 home, 17 away)
        y_win = torch.rand(1)
        y_margin = torch.randn(1) * 20
        y_points = torch.randn(1) * 40 + 30
        y_try_scorers = torch.rand(34)

        return cat_x, roster_x, y_win, y_margin, y_points, y_try_scorers


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training SGM Transformer on {device}")

    num_teams, num_venues, num_players = 20, 15, 500
    dataset = DummySGMDataset(
        num_teams=num_teams, num_venues=num_venues, num_players=num_players
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = SGMTransformer(
        num_teams=num_teams, num_venues=num_venues, num_players=num_players
    ).to(device)

    criterion_bce = nn.BCELoss()
    criterion_mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    epochs = 10
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for cat_x, roster_x, y_win, y_margin, y_points, y_try_scorers in dataloader:
            cat_x, roster_x = cat_x.to(device), roster_x.to(device)
            y_win, y_margin, y_points, y_try_scorers = (
                y_win.to(device),
                y_margin.to(device),
                y_points.to(device),
                y_try_scorers.to(device),
            )

            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast("cuda"):
                    win_prob, margin, points, try_probs = model(cat_x, roster_x)

                    loss_win = criterion_bce(win_prob, y_win)
                    loss_margin = criterion_mse(margin, y_margin)
                    loss_points = criterion_mse(points, y_points)
                    loss_try = criterion_bce(try_probs, y_try_scorers)

                    loss = (
                        loss_win + (0.1 * loss_margin) + (0.1 * loss_points) + loss_try
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                win_prob, margin, points, try_probs = model(cat_x, roster_x)

                loss_win = criterion_bce(win_prob, y_win)
                loss_margin = criterion_mse(margin, y_margin)
                loss_points = criterion_mse(points, y_points)
                loss_try = criterion_bce(try_probs, y_try_scorers)

                loss = loss_win + (0.1 * loss_margin) + (0.1 * loss_points) + loss_try

                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/sgm_model.pth")
    print("Saved SGM Transformer to models/sgm_model.pth")


if __name__ == "__main__":
    train()
