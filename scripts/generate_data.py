import pandas as pd
import numpy as np
import os
import random


def generate_nrl_data(num_matches=50000):
    teams = [
        "Storm",
        "Panthers",
        "Roosters",
        "Broncos",
        "Rabbitohs",
        "Rabbitohs",
        "Cowboys",
        "Sharks",
        "Sea Sharks",
        "Eels",
        "Knights",
        "Titans",
        "Bulldogs",
        "Dragons",
        "Warriors",
        "Tigers",
    ]
    venues = [
        "Suncorp",
        "AAMI",
        "Allianz",
        "CommBank",
        "BlueBet",
        "Accor",
        "GIO",
        "McDonald Jones",
    ]
    weather = ["Clear", "Rain", "Overcast"]
    events = [
        "Hit up/Run",
        "Tackle",
        "Kick",
        "Try",
        "Error",
        "Penalty",
        "Offload",
        "Linebreak",
        "Goal",
    ]

    print("Generating matches...")
    matches = []
    for match_id in range(1, num_matches + 1):
        home = random.choice(teams)
        away = random.choice([t for t in teams if t != home])
        venue = random.choice(venues)
        w = random.choice(weather)
        home_elo = max(1000, np.random.normal(1500, 150))
        away_elo = max(1000, np.random.normal(1500, 150))
        home_rest = random.randint(5, 10)
        away_rest = random.randint(5, 10)

        elo_diff = home_elo - away_elo + 50  # home advantage
        win_prob = 1 / (1 + 10 ** (-elo_diff / 400))
        home_win = 1 if random.random() < win_prob else 0

        matches.append(
            [
                match_id,
                home,
                away,
                venue,
                w,
                home_elo,
                away_elo,
                home_rest,
                away_rest,
                home_win,
            ]
        )

    df_matches = pd.DataFrame(
        matches,
        columns=[
            "match_id",
            "home_team",
            "away_team",
            "venue",
            "weather",
            "home_elo",
            "away_elo",
            "home_rest_days",
            "away_rest_days",
            "home_win",
        ],
    )

    print("Generating play-by-play...")
    plays = []
    for match_id in range(1, num_matches + 1):
        seq_len = random.randint(180, 220)
        current_pos = random.randint(10, 90)  # 0 to 100 meters
        tackle_count = 0
        possessing = 0  # 0 for home, 1 for away

        for t in range(seq_len):
            # Transition logic (basic Markov-ish)
            if tackle_count >= 5:
                event = "Kick"
            else:
                event = random.choices(events, weights=[40, 30, 2, 2, 5, 5, 10, 5, 1])[
                    0
                ]

            if event == "Hit up/Run":
                current_pos += random.randint(-2, 10)
                tackle_count += 1
            elif event == "Tackle":
                current_pos += random.randint(-1, 3)
                tackle_count += 1
            elif event == "Kick":
                current_pos += random.randint(20, 50)
                tackle_count = 0
                possessing = 1 - possessing
            elif event in ["Error", "Penalty", "Try"]:
                tackle_count = 0
                possessing = 1 - possessing
                if event == "Try":
                    current_pos = 100
                elif event == "Penalty":
                    current_pos += 10

            # keep position in bounds
            current_pos = max(0, min(100, current_pos))
            plays.append([match_id, t, possessing, event, tackle_count, current_pos])

    df_plays = pd.DataFrame(
        plays,
        columns=[
            "match_id",
            "minute",
            "possessing",
            "event",
            "tackle_count",
            "field_position",
        ],
    )

    os.makedirs("data", exist_ok=True)
    df_matches.to_csv("data/nrl_matches.csv", index=False)
    df_plays.to_csv("data/nrl_playbyplay.csv", index=False)
    print("Data generation complete.")


if __name__ == "__main__":
    generate_nrl_data()
