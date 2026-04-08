import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

from nrl_ml.models import ContextualStackedLSTM


# Dummy Dataset to mimic real sequence data
class DummySequenceDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=50, vocab_size=20):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # x: Sequence of events (integers)
        x = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)

        # context_x: field position, tackle count, score diff
        field_pos = torch.rand(self.seq_len) * 100
        tackle_count = torch.randint(0, 6, (self.seq_len,), dtype=torch.float)
        score_diff = (torch.rand(self.seq_len) * 40) - 20
        context_x = torch.stack([field_pos, tackle_count, score_diff], dim=-1)

        # targets
        y_event = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        y_gain = torch.randn(self.seq_len, 1) * 10

        return x, context_x, y_event, y_gain


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training LSTM on {device}")

    vocab_size = 20
    dataset = DummySequenceDataset(vocab_size=vocab_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # pad_idx must be < vocab_size when creating the embedding. Let's make actual vocab size vocab_size+2
    # to safely include padding and other potential out-of-bounds events.
    actual_vocab_size = vocab_size + 2
    model = ContextualStackedLSTM(
        vocab_size=actual_vocab_size, pad_idx=vocab_size + 1
    ).to(device)

    criterion_event = nn.CrossEntropyLoss()
    criterion_gain = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    epochs = 10
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, context_x, y_event, y_gain in dataloader:
            x, context_x = x.to(device), context_x.to(device)
            y_event, y_gain = y_event.to(device), y_gain.to(device)

            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast("cuda"):
                    event_logits, expected_gain = model(x, context_x)
                    # Reshape for CrossEntropy: (batch * seq_len, vocab_size)
                    loss_event = criterion_event(
                        event_logits.view(-1, actual_vocab_size), y_event.view(-1)
                    )
                    loss_gain = criterion_gain(expected_gain, y_gain)
                    loss = loss_event + (0.1 * loss_gain)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                event_logits, expected_gain = model(x, context_x)
                loss_event = criterion_event(
                    event_logits.view(-1, actual_vocab_size), y_event.view(-1)
                )
                loss_gain = criterion_gain(expected_gain, y_gain)
                loss = loss_event + (0.1 * loss_gain)

                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/lstm_model.pth")
    print("Saved LSTM model to models/lstm_model.pth")


if __name__ == "__main__":
    train()
