import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class OmniModelConfig:
    num_teams: int
    num_venues: int
    num_players: int
    vocab_size: int
    embed_dim: int = 128
    n_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    max_seq_len: int = 200
    global_cont_dim: int = 4
    player_cont_dim: int = 3

class NRLOmniModel(nn.Module):
    def __init__(self, config: OmniModelConfig):
        super(NRLOmniModel, self).__init__()
        self.config = config
        
        # Embeddings
        self.team_emb = nn.Embedding(config.num_teams, config.embed_dim)
        self.venue_emb = nn.Embedding(config.num_venues, config.embed_dim)
        self.player_emb = nn.Embedding(config.num_players, config.embed_dim)
        
        self.global_cont_proj = nn.Linear(config.global_cont_dim, config.embed_dim)
        self.player_cont_proj = nn.Linear(config.player_cont_dim, config.embed_dim)
        
        self.event_emb = nn.Embedding(config.vocab_size + 1, config.embed_dim, padding_idx=config.vocab_size)
        self.context_proj = nn.Linear(3, config.embed_dim)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim, 
            nhead=config.n_heads, 
            dim_feedforward=config.embed_dim * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        self.win_head = nn.Linear(config.embed_dim, 1)
        self.margin_head = nn.Linear(config.embed_dim, 1)
        self.total_points_head = nn.Linear(config.embed_dim, 1)
        self.try_scorer_head = nn.Linear(config.embed_dim, 1)
        
        self.event_head = nn.Linear(config.embed_dim, config.vocab_size)
        self.gain_head = nn.Linear(config.embed_dim, 1)

    def forward(self, cat_x: torch.Tensor, roster_x: torch.Tensor, global_cont: torch.Tensor, player_cont: torch.Tensor, seq_x: Optional[torch.Tensor] = None, seq_context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        B = cat_x.size(0)
        
        h_emb = self.team_emb(cat_x[:, 0]).unsqueeze(1)
        a_emb = self.team_emb(cat_x[:, 1]).unsqueeze(1)
        v_emb = self.venue_emb(cat_x[:, 2]).unsqueeze(1)
        
        g_cont_emb = self.global_cont_proj(global_cont).unsqueeze(1)
        
        h_roster = self.player_emb(roster_x[:, 0, :]) + self.player_cont_proj(player_cont[:, 0, :, :])
        a_roster = self.player_emb(roster_x[:, 1, :]) + self.player_cont_proj(player_cont[:, 1, :, :])
        
        match_tokens = torch.cat([h_emb, a_emb, v_emb, g_cont_emb, h_roster, a_roster], dim=1) # (B, 38, D)
        
        if seq_x is not None and seq_context is not None:
            S = seq_x.size(1)
            positions = torch.arange(0, S, device=seq_x.device).unsqueeze(0).expand(B, S)
            seq_emb = self.event_emb(seq_x) + self.pos_emb(positions) + self.context_proj(seq_context)
            tokens = torch.cat([match_tokens, seq_emb], dim=1)
        else:
            tokens = match_tokens
            
        reasoned_tokens = self.transformer(tokens)
        
        global_context = reasoned_tokens[:, 0:4, :].mean(dim=1)
        player_tokens = reasoned_tokens[:, 4:38, :]
        
        win_prob = torch.sigmoid(self.win_head(global_context))
        margin = self.margin_head(global_context)
        total_points = self.total_points_head(global_context)
        try_probs = torch.sigmoid(self.try_scorer_head(player_tokens).squeeze(-1))
        
        if seq_x is not None and seq_context is not None:
            seq_tokens = reasoned_tokens[:, 38:, :]
            next_event_logits = self.event_head(seq_tokens)
            expected_gain = self.gain_head(seq_tokens)
            return win_prob, margin, total_points, try_probs, next_event_logits, expected_gain
            
        return win_prob, margin, total_points, try_probs, None, None
