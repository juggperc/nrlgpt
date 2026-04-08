import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os


class NRLOutcomeDataset(Dataset):
    def __init__(self, csv_file, is_train=True):
        self.data = pd.read_csv(csv_file)

        # Encoders
        self.team_encoder = LabelEncoder()
        self.venue_encoder = LabelEncoder()
        self.player_encoder = LabelEncoder()

        # Parse rosters
        self.data["home_roster_list"] = (
            self.data.get("home_roster", pd.Series([""] * len(self.data)))
            .fillna("")
            .apply(lambda x: [p for p in str(x).split(",") if p])
        )
        self.data["away_roster_list"] = (
            self.data.get("away_roster", pd.Series([""] * len(self.data)))
            .fillna("")
            .apply(lambda x: [p for p in str(x).split(",") if p])
        )

        # Fit or load
        os.makedirs("models/encoders", exist_ok=True)
        if is_train:
            all_teams = pd.concat(
                [self.data["home_team"], self.data["away_team"]]
            ).unique()
            self.team_encoder.fit(all_teams)
            self.venue_encoder.fit(self.data["venue"])

            # Extract all players
            all_players = set()
            for r in self.data["home_roster_list"]:
                all_players.update(r)
            for r in self.data["away_roster_list"]:
                all_players.update(r)
            self.player_encoder.fit(
                list(all_players) + ["Unknown_H", "Unknown_A", "Unknown"]
            )

            with open("models/encoders/teams.pkl", "wb") as f:
                pickle.dump(self.team_encoder, f)
            with open("models/encoders/venues.pkl", "wb") as f:
                pickle.dump(self.venue_encoder, f)
            with open("models/encoders/players.pkl", "wb") as f:
                pickle.dump(self.player_encoder, f)
        else:
            with open("models/encoders/teams.pkl", "rb") as f:
                self.team_encoder = pickle.load(f)
            with open("models/encoders/venues.pkl", "rb") as f:
                self.venue_encoder = pickle.load(f)
            with open("models/encoders/players.pkl", "rb") as f:
                self.player_encoder = pickle.load(f)

        # Handle unseen labels carefully if testing
        self.data["home_team_idx"] = self.team_encoder.transform(self.data["home_team"])
        self.data["away_team_idx"] = self.team_encoder.transform(self.data["away_team"])
        self.data["venue_idx"] = self.venue_encoder.transform(self.data["venue"])

        # Transform rosters
        def encode_roster(r):
            encoded = []
            for p in r[:17]:
                try:
                    encoded.append(self.player_encoder.transform([p])[0])
                except ValueError:
                    encoded.append(self.player_encoder.transform(["Unknown"])[0])
            while len(encoded) < 17:
                encoded.append(self.player_encoder.transform(["Unknown"])[0])
            return encoded

        self.data["home_roster_idx"] = self.data["home_roster_list"].apply(
            encode_roster
        )
        self.data["away_roster_idx"] = self.data["away_roster_list"].apply(
            encode_roster
        )

        # Continuous Features
        cont_cols = ["home_elo", "away_elo", "home_rest_days", "away_rest_days"]
        self.cont_features = self.data[cont_cols].values

        if is_train:
            self.scaler = StandardScaler()
            self.cont_features = self.scaler.fit_transform(self.cont_features)
            with open("models/encoders/outcome_scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)
        else:
            with open("models/encoders/outcome_scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            self.cont_features = self.scaler.transform(self.cont_features)

        self.labels = self.data["home_win"].values

        self.num_teams = len(self.team_encoder.classes_)
        self.num_venues = len(self.venue_encoder.classes_)
        self.num_players = len(self.player_encoder.classes_)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        cat_x = torch.tensor(
            [
                self.data.iloc[idx]["home_team_idx"],
                self.data.iloc[idx]["away_team_idx"],
                self.data.iloc[idx]["venue_idx"],
            ],
            dtype=torch.long,
        )

        roster_x = torch.tensor(
            [
                self.data.iloc[idx]["home_roster_idx"],
                self.data.iloc[idx]["away_roster_idx"],
            ],
            dtype=torch.long,
        )

        cont_x = torch.tensor(self.cont_features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return cat_x, roster_x, cont_x, y


class NRLSequenceDataset(Dataset):
    def __init__(self, sequence_file, max_len=200, is_train=True):
        self.data = pd.read_csv(sequence_file)
        self.max_len = max_len

        # Event vocab
        os.makedirs("models/encoders", exist_ok=True)
        if is_train:
            self.event_encoder = LabelEncoder()
            self.event_encoder.fit(self.data["event"])
            with open("models/encoders/events.pkl", "wb") as f:
                pickle.dump(self.event_encoder, f)
        else:
            with open("models/encoders/events.pkl", "rb") as f:
                self.event_encoder = pickle.load(f)

        self.vocab_size = len(self.event_encoder.classes_)
        # Add padding token to vocab space, say vocab_size is reserved for padding
        self.pad_idx = self.vocab_size

        self.data["event_idx"] = self.event_encoder.transform(self.data["event"])

        # Group sequences by match
        self.sequences = []
        for match_id, group in self.data.groupby("match_id"):
            group = group.sort_values("minute")
            # Extract sequence of events
            events = group["event_idx"].values
            if len(events) > 1:
                self.sequences.append(events)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        if len(seq) >= self.max_len + 1:
            seq = seq[: self.max_len + 1]
        else:
            seq = np.pad(
                seq,
                (0, self.max_len + 1 - len(seq)),
                "constant",
                constant_values=self.pad_idx,
            )

        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y
