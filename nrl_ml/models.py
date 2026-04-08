import torch
import torch.nn as nn


class OutcomeModel(nn.Module):
    """
    Advanced NRL Match Outcome Predictor.
    Uses embeddings for categorical features (Home/Away/Venue), MLPs for continuous features,
    and EmbeddingBags for 17-man Rosters.
    """

    def __init__(
        self,
        num_teams,
        num_venues,
        num_players=1000,
        embed_dim=16,
        cont_dim=4,
        hidden_dim=256,
    ):
        super(OutcomeModel, self).__init__()

        self.team_emb = nn.Embedding(num_teams, embed_dim)
        self.venue_emb = nn.Embedding(num_venues, embed_dim)

        # Averages the embeddings of the 17 players on the field
        self.roster_emb = nn.EmbeddingBag(num_players, embed_dim, mode="mean")

        input_dim = (
            (embed_dim * 3) + (embed_dim * 2) + cont_dim
        )  # Teams/Venue + 2 Rosters + cont features

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, cat_x, roster_x, cont_x):
        # cat_x: (batch, 3) [home_idx, away_idx, venue_idx]
        # roster_x: (batch, 2, 17) [home_roster, away_roster]
        # cont_x: (batch, 4) [home_elo, away_elo, home_rest, away_rest]

        h_emb = self.team_emb(cat_x[:, 0])
        a_emb = self.team_emb(cat_x[:, 1])
        v_emb = self.venue_emb(cat_x[:, 2])

        # Roster features (takes the 17 IDs for home and 17 IDs for away and averages them into team strength vectors)
        h_roster_emb = self.roster_emb(roster_x[:, 0, :])
        a_roster_emb = self.roster_emb(roster_x[:, 1, :])

        x = torch.cat([h_emb, a_emb, v_emb, h_roster_emb, a_roster_emb, cont_x], dim=1)

        return torch.sigmoid(self.network(x))


class ContextualStackedLSTM(nn.Module):
    """
    Advanced Play-by-Play Contextual Sequence Model.
    Takes the previous Event + Context (Field Position, Tackle Count, Score Diff).
    Outputs a prediction for the Next Event AND the Expected Meters Gained!
    """

    def __init__(
        self,
        vocab_size,
        embed_size=128,
        context_size=3,
        hidden_size=512,
        num_layers=3,
        pad_idx=-1,
    ):
        super(ContextualStackedLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)

        # Project continuous context (field pos, tackle, score) into a dense space
        self.context_proj = nn.Linear(context_size, 32)

        # LSTM input is Event Embedding + Context Projection
        self.lstm = nn.LSTM(
            embed_size + 32,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=False,
        )

        # Multi-task learning heads
        self.fc_event = nn.Linear(hidden_size, vocab_size)  # Predicts next event type
        self.fc_gain = nn.Linear(hidden_size, 1)  # Predicts meters gained on that event

    def forward(self, x, context_x):
        # x: (batch, seq_len)
        # context_x: (batch, seq_len, 3) -> [field_pos, tackle_count, score_diff]

        embedded = self.embedding(x)
        context_proj = torch.relu(self.context_proj(context_x))

        # Combine event info with physical reality of the field
        lstm_input = torch.cat([embedded, context_proj], dim=-1)

        lstm_out, _ = self.lstm(lstm_input)

        event_logits = self.fc_event(lstm_out)
        expected_gain = self.fc_gain(lstm_out)

        return event_logits, expected_gain


class SGMTransformer(nn.Module):
    """
    Reasoning-based Multi-Task Transformer for Same Game Multis (SGMs).
    Ingests Team, Venue, and the exact 34 players on the field.
    Uses Multi-Head Attention to reason about matchups (e.g. Winger vs Winger, Prop vs Prop)
    to output:
    1. Win Probability
    2. Expected Margin
    3. Expected Total Points
    4. Anytime Try Scorer Probabilities for all 34 players
    """

    def __init__(
        self, num_teams, num_venues, num_players, embed_dim=64, n_heads=4, num_layers=2
    ):
        super(SGMTransformer, self).__init__()
        self.team_emb = nn.Embedding(num_teams, embed_dim)
        self.venue_emb = nn.Embedding(num_venues, embed_dim)
        self.player_emb = nn.Embedding(num_players, embed_dim)

        # Transformer Encoder to reason across all entities on the field simultaneously
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, batch_first=True, dropout=0.2
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Prediction Heads based on the pooled reasoning vector
        self.win_head = nn.Linear(embed_dim, 1)
        self.margin_head = nn.Linear(embed_dim, 1)
        self.total_points_head = nn.Linear(embed_dim, 1)

        # Try Scorer head applies to the specific player tokens (indices 3 through 36)
        self.try_scorer_head = nn.Linear(embed_dim, 1)

    def forward(self, cat_x, roster_x):
        # cat_x: (batch, 3) [home_idx, away_idx, venue_idx]
        # roster_x: (batch, 2, 17) [home_roster, away_roster]

        h_emb = self.team_emb(cat_x[:, 0]).unsqueeze(1)  # (B, 1, D)
        a_emb = self.team_emb(cat_x[:, 1]).unsqueeze(1)  # (B, 1, D)
        v_emb = self.venue_emb(cat_x[:, 2]).unsqueeze(1)  # (B, 1, D)

        h_roster = self.player_emb(roster_x[:, 0, :])  # (B, 17, D)
        a_roster = self.player_emb(roster_x[:, 1, :])  # (B, 17, D)

        # Concat the 37 tokens representing the entire match ecosystem
        # [Home Team, Away Team, Venue, 17 Home Players, 17 Away Players]
        seq = torch.cat([h_emb, a_emb, v_emb, h_roster, a_roster], dim=1)  # (B, 37, D)

        # Reason across all relationships (e.g. Cleary's attention on the opponent's weak edge defender)
        reasoned_seq = self.transformer(seq)  # (B, 37, D)

        # Global match context (mean of Team & Venue tokens)
        global_context = reasoned_seq[:, 0:3, :].mean(dim=1)  # (B, D)

        win_prob = torch.sigmoid(self.win_head(global_context))
        expected_margin = self.margin_head(global_context)
        expected_total_points = self.total_points_head(global_context)

        # Try scorer probabilities derived from their specific reasoned tokens
        player_tokens = reasoned_seq[:, 3:, :]  # (B, 34, D)
        try_probs = torch.sigmoid(
            self.try_scorer_head(player_tokens).squeeze(-1)
        )  # (B, 34)

        return win_prob, expected_margin, expected_total_points, try_probs
