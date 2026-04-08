import argparse
import torch
import numpy as np
import pickle
import pandas as pd
from nrl_ml.models import OutcomeModel, ContextualStackedLSTM, SGMTransformer
import os


def load_encoders():
    with open("models/encoders/teams.pkl", "rb") as f:
        team_encoder = pickle.load(f)
    with open("models/encoders/venues.pkl", "rb") as f:
        venue_encoder = pickle.load(f)
    with open("models/encoders/events.pkl", "rb") as f:
        event_encoder = pickle.load(f)
    with open("models/encoders/outcome_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    try:
        with open("models/encoders/players.pkl", "rb") as f:
            player_encoder = pickle.load(f)
    except FileNotFoundError:
        player_encoder = None
    return team_encoder, venue_encoder, player_encoder, event_encoder, scaler


def load_models(device, num_teams, num_venues, num_players, vocab_size):
    outcome_model = OutcomeModel(num_teams, num_venues, num_players).to(device)
    sequence_model = ContextualStackedLSTM(
        vocab_size=vocab_size + 1, pad_idx=vocab_size
    ).to(device)
    sgm_model = SGMTransformer(num_teams, num_venues, num_players).to(device)

    try:
        outcome_model.load_state_dict(
            torch.load("models/outcome_model.pth", map_location=device)
        )
    except:
        pass

    try:
        sequence_model.load_state_dict(
            torch.load("models/sequence_model.pth", map_location=device)
        )
    except:
        pass

    try:
        sgm_model.load_state_dict(
            torch.load("models/sgm_model.pth", map_location=device)
        )
    except:
        pass

    outcome_model.eval()
    sequence_model.eval()
    sgm_model.eval()
    return outcome_model, sequence_model, sgm_model


def simulate_outcome(
    home_team,
    away_team,
    venue,
    model,
    device,
    team_encoder,
    venue_encoder,
    player_encoder,
    scaler,
):
    print(f"\n🏟️  Matchup: {home_team} vs {away_team} at {venue}")

    # Categorical encoding
    try:
        home_idx = team_encoder.transform([home_team])[0]
        away_idx = team_encoder.transform([away_team])[0]
        venue_idx = venue_encoder.transform([venue])[0]
    except ValueError as e:
        print(f"Error: {e}. Available teams: {team_encoder.classes_}")
        return

    cat_x = torch.tensor([[home_idx, away_idx, venue_idx]], dtype=torch.long).to(device)

    # Roster mock for now
    if player_encoder is not None:
        try:
            unknown_idx = player_encoder.transform(["Unknown"])[0]
        except:
            unknown_idx = 0
    else:
        unknown_idx = 0

    roster_x = torch.tensor(
        [[[unknown_idx] * 17, [unknown_idx] * 17]], dtype=torch.long
    ).to(device)

    # Mock realistic continuous features (home_elo, away_elo, home_rest, away_rest)
    cont_raw = np.array([[1550, 1480, 7, 6]])
    cont_scaled = scaler.transform(cont_raw)
    cont_x = torch.tensor(cont_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        home_win_prob = model(cat_x, roster_x, cont_x).item()

    print(f"📊 {home_team} Win Probability: {home_win_prob:.1%}")
    print(f"📊 {away_team} Win Probability: {(1 - home_win_prob):.1%}")
    winner = home_team if home_win_prob > 0.5 else away_team
    print(f"🏆 Predicted Winner: {winner}")


def simulate_sequence(home_team, away_team, model, device, event_encoder, num_plays=20):
    teams = [home_team, away_team]

    print(f"\n🏉 Kickoff! Simulating {num_plays} plays...")
    current_team_idx = 0

    # Kickoff event idx
    try:
        kick_idx = event_encoder.transform(["Kick"])[0]
    except:
        kick_idx = 0

    current_sequence = [kick_idx]

    tackle_count = 0
    field_pos = 50  # 50m line
    score_diff = 0

    for play in range(num_plays):
        input_seq = torch.tensor([current_sequence], dtype=torch.long).to(device)
        context_seq = torch.tensor(
            [[[field_pos, tackle_count, score_diff]]], dtype=torch.float32
        ).to(device)
        # Sequence model needs context length to match sequence length, so we mock context across the whole sequence
        context_seq = context_seq.expand(-1, input_seq.size(1), -1)

        with torch.no_grad():
            logits, gain_pred = model(input_seq, context_seq)
            next_event_logits = logits[0, -1, :]
            gain = int(gain_pred[0, -1, 0].item())

            # Avoid predicting padding token
            pad_idx = len(event_encoder.classes_)
            next_event_logits[pad_idx] = -1e9

            # Apply temperature for sampling
            temperature = 0.8
            probs = torch.softmax(next_event_logits / temperature, dim=-1).cpu().numpy()

            # Sample event
            next_event_id = np.random.choice(len(probs), p=probs)

        current_sequence.append(next_event_id)
        event_name = event_encoder.inverse_transform([next_event_id])[0]

        team_name = teams[current_team_idx]

        # State tracking and rich commentary
        if event_name == "Hit up/Run":
            tackle_count += 1
            field_pos += gain if current_team_idx == 0 else -gain
            print(
                f"Minute {play + 1:02d}: {team_name} takes a hit up. Gained {gain}m. (Tackle {tackle_count}/6)"
            )
        elif event_name == "Tackle":
            tackle_count += 1
            print(
                f"Minute {play + 1:02d}: {team_name} is tackled. (Tackle {tackle_count}/6)"
            )
        elif event_name == "Offload":
            print(
                f"Minute {play + 1:02d}: Brilliant offload by {team_name}, keeping the play alive! (Tackle {tackle_count}/6)"
            )
        elif event_name == "Linebreak":
            gain = np.random.randint(15, 40)
            field_pos += gain if current_team_idx == 0 else -gain
            print(
                f"Minute {play + 1:02d}: ⚡ LINEBREAK by {team_name}! Slicing through the defense for {gain}m!"
            )
        elif event_name == "Error":
            print(
                f"Minute {play + 1:02d}: ❌ Unforced error by {team_name}. Knock on! Turnover to {teams[1 - current_team_idx]}."
            )
            current_team_idx = 1 - current_team_idx
            tackle_count = 0
        elif event_name == "Kick":
            print(
                f"Minute {play + 1:02d}: 🥾 {team_name} kicks downfield. Turnover to {teams[1 - current_team_idx]}."
            )
            current_team_idx = 1 - current_team_idx
            tackle_count = 0
        elif event_name == "Try":
            print(
                f"Minute {play + 1:02d}: 🚨 TRY! {team_name.upper()} CROSSES THE LINE! What a spectacular play!"
            )
            current_team_idx = 1 - current_team_idx
            tackle_count = 0
        elif event_name == "Penalty":
            print(
                f"Minute {play + 1:02d}: 🛑 Penalty against {teams[1 - current_team_idx]}. {team_name} gets a new set of six."
            )
            tackle_count = 0
        elif event_name == "Goal":
            print(f"Minute {play + 1:02d}: 🎯 {team_name} kicks the goal successfully.")
            current_team_idx = 1 - current_team_idx
            tackle_count = 0

        if tackle_count >= 6:
            print(
                f"Minute {play + 1:02d}: 🔄 Handover. {team_name} tackled on the last. Turnover."
            )
            current_team_idx = 1 - current_team_idx
            tackle_count = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=str, default="Storm")
    parser.add_argument("--away", type=str, default="Panthers")
    parser.add_argument("--venue", type=str, default="Suncorp")
    parser.add_argument("--plays", type=int, default=20)
    args = parser.parse_args()

    os.environ["PYTHONIOENCODING"] = "utf-8"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    team_encoder, venue_encoder, player_encoder, event_encoder, scaler = load_encoders()

    num_teams = len(team_encoder.classes_)
    num_venues = len(venue_encoder.classes_)
    num_players = len(player_encoder.classes_) if player_encoder else 1000
    vocab_size = len(event_encoder.classes_)

    outcome_model, sequence_model = load_models(
        device, num_teams, num_venues, num_players, vocab_size
    )

    simulate_outcome(
        args.home,
        args.away,
        args.venue,
        outcome_model,
        device,
        team_encoder,
        venue_encoder,
        player_encoder,
        scaler,
    )
    simulate_sequence(
        args.home,
        args.away,
        sequence_model,
        device,
        event_encoder,
        num_plays=args.plays,
    )
