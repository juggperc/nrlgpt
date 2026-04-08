import torch
from torch.utils.data import Dataset
import numpy as np


class NRLOmniDataset(Dataset):
    """
    Unified dataset that yields multimodal tokens for the NRLOmniModel.
    Provides: Match Context, Roster context, and Sequence (play-by-play) arrays simultaneously.
    """

    def __init__(self, match_data, sequence_data, max_seq_len=200):
        self.match_data = match_data
        self.sequence_data = sequence_data
        self.max_seq_len = max_seq_len
        self.pad_idx = 20  # Assume vocab size 20 + 1 for padding

    def __len__(self):
        return len(self.match_data)

    def __getitem__(self, idx):
        # 1. Match/Team Context
        # cat_x -> [home_id, away_id, venue_id]
        cat_x = torch.tensor(
            [
                self.match_data[idx]["home_id"],
                self.match_data[idx]["away_id"],
                self.match_data[idx]["venue_id"],
            ],
            dtype=torch.long,
        )

        # 2. Roster Context
        # roster_x -> [17 home, 17 away]
        roster_x = torch.tensor(
            [self.match_data[idx]["home_roster"], self.match_data[idx]["away_roster"]],
            dtype=torch.long,
        )

        # 3. Targets (Global / Set-based)
        win = torch.tensor(self.match_data[idx]["home_win"], dtype=torch.float32)
        margin = torch.tensor(self.match_data[idx]["margin"], dtype=torch.float32)
        points = torch.tensor(self.match_data[idx]["total_points"], dtype=torch.float32)
        try_scorers = torch.tensor(
            self.match_data[idx]["try_scorers"], dtype=torch.float32
        )

        # 4. Sequence Context
        seq_events = self.sequence_data[idx]["events"]
        seq_context = self.sequence_data[idx]["continuous"]  # field_pos, tackle, score
        seq_targets = self.sequence_data[idx]["next_events"]
        seq_gains = self.sequence_data[idx]["gains"]

        # Padding logic for sequences
        if len(seq_events) < self.max_seq_len:
            pad_len = self.max_seq_len - len(seq_events)
            seq_events = np.pad(
                seq_events, (0, pad_len), "constant", constant_values=self.pad_idx
            )
            seq_targets = np.pad(
                seq_targets, (0, pad_len), "constant", constant_values=self.pad_idx
            )

            # Pad context with zeros
            seq_context = np.pad(seq_context, ((0, pad_len), (0, 0)), "constant")
            seq_gains = np.pad(seq_gains, (0, pad_len), "constant")
        else:
            seq_events = seq_events[: self.max_seq_len]
            seq_targets = seq_targets[: self.max_seq_len]
            seq_context = seq_context[: self.max_seq_len]
            seq_gains = seq_gains[: self.max_seq_len]

        return (
            cat_x,
            roster_x,
            torch.tensor(seq_events, dtype=torch.long),
            torch.tensor(seq_context, dtype=torch.float32),
            win,
            margin,
            points,
            try_scorers,
            torch.tensor(seq_targets, dtype=torch.long),
            torch.tensor(seq_gains, dtype=torch.float32),
        )
