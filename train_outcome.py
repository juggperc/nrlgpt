import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from nrl_ml.dataset import NRLOutcomeDataset
from nrl_ml.models import OutcomeModel
import os


def train_outcome(epochs, batch_size, lr, device, patience=10):
    print("Loading highly realistic training data...")
    dataset = NRLOutcomeDataset("data/real_processed_matches.csv", is_train=True)

    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_teams = dataset.num_teams
    num_venues = dataset.num_venues
    num_players = dataset.num_players

    model = OutcomeModel(num_teams, num_venues, num_players).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.BCELoss()

    print(f"Training Match Outcome Model on {device}...")
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_cat, batch_roster, batch_cont, batch_y in train_loader:
            batch_cat, batch_roster, batch_cont, batch_y = (
                batch_cat.to(device),
                batch_roster.to(device),
                batch_cont.to(device),
                batch_y.to(device),
            )

            optimizer.zero_grad()
            outputs = model(batch_cat, batch_roster, batch_cont).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_cat, batch_roster, batch_cont, batch_y in val_loader:
                batch_cat, batch_roster, batch_cont, batch_y = (
                    batch_cat.to(device),
                    batch_roster.to(device),
                    batch_cont.to(device),
                    batch_y.to(device),
                )
                outputs = model(batch_cat, batch_roster, batch_cont).squeeze()
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                predicted = (outputs > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        scheduler.step(val_loss)

        print(
            f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {train_loss / len(train_loader):.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {accuracy:.2f}%"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/outcome_model.pth")
            epochs_no_improve = 0
            print(f"[*] New best model saved (Val Loss: {val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs!")
                break

    print("Training finished! Best model is at models/outcome_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_outcome(args.epochs, args.batch_size, args.lr, device, args.patience)
