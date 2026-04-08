import json
import os
import pandas as pd


def extract_players(node, player_lookup):
    if isinstance(node, dict):
        for k, v in node.items():
            if (
                isinstance(v, list)
                and len(v) > 0
                and isinstance(v[0], dict)
                and "Name" in v[0]
            ):
                player_lookup[k] = [p["Name"] for p in v if "Name" in p]
            else:
                extract_players(v, player_lookup)
    elif isinstance(node, list):
        for item in node:
            extract_players(item, player_lookup)


def extract_matches(node, year, player_lookup, match_records):
    if isinstance(node, dict):
        if (
            "Home" in node
            and "Away" in node
            and "Home_Score" in node
            and node["Home_Score"] != "null"
        ):
            try:
                h_score = int(node["Home_Score"])
                a_score = int(node["Away_Score"])
                h_team = node["Home"]
                a_team = node["Away"]
                venue = node.get("Venue", "Unknown")
                rnd = str(node.get("Round", "")).replace("Round ", "")

                # Reconstruct key format like "2023-1-Eels-v-Storm"
                # Strip spaces for matching
                search_key = (
                    f"{h_team.replace(' ', '')}-v-{a_team.replace(' ', '')}".lower()
                )

                roster = []
                for pk, pv in player_lookup.items():
                    if search_key in pk.replace(" ", "").lower():
                        roster = pv
                        break

                home_roster = roster[:17]
                away_roster = roster[17:34]

                # Pad if missing
                while len(home_roster) < 17:
                    home_roster.append("Unknown_H")
                while len(away_roster) < 17:
                    away_roster.append("Unknown_A")

                match_records.append(
                    {
                        "year": year,
                        "home_team": h_team,
                        "away_team": a_team,
                        "venue": venue,
                        "home_score": h_score,
                        "away_score": a_score,
                        "home_win": 1 if h_score > a_score else 0,
                        "home_roster": ",".join(home_roster[:17]),
                        "away_roster": ",".join(away_roster[:17]),
                        "home_elo": 1500,  # Default for now
                        "away_elo": 1500,
                        "home_rest_days": 7,
                        "away_rest_days": 7,
                    }
                )
            except ValueError:
                pass
        else:
            for k, v in node.items():
                extract_matches(v, year, player_lookup, match_records)
    elif isinstance(node, list):
        for item in node:
            extract_matches(item, year, player_lookup, match_records)


def process():
    match_records = []
    base_dir = "E:/nrlgpt/data/real_nrl_data/"

    for year in range(2001, 2025):
        match_file = os.path.join(base_dir, f"NRL_data_{year}.json")
        player_file = os.path.join(base_dir, f"NRL_player_statistics_{year}.json")

        if not os.path.exists(match_file) or not os.path.exists(player_file):
            continue

        print(f"Processing {year}...")

        try:
            with open(match_file, "r", encoding="utf-8") as f:
                match_data = json.load(f)
        except:
            continue

        player_lookup = {}
        try:
            with open(player_file, "r", encoding="utf-8") as f:
                player_data = json.load(f)
            extract_players(player_data, player_lookup)
        except:
            pass

        extract_matches(match_data, year, player_lookup, match_records)

    df = pd.DataFrame(match_records)
    df.to_csv("E:/nrlgpt/data/real_processed_matches.csv", index=False)
    print(f"Saved {len(df)} real matches with full 17-man rosters!")


if __name__ == "__main__":
    process()
