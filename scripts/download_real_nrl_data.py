import os
import requests
import json

DATA_WEBSITE = "https://geo145327-staging.s3.ap-southeast-2.amazonaws.com/public/"
COMPETITION = "NRL"
YEARS = list(range(2001, 2025))


def download_all():
    base_dir = "E:/nrlgpt/data/real_nrl_data"
    os.makedirs(base_dir, exist_ok=True)

    file_types = ["data", "detailed_match_data", "player_statistics"]

    for year in YEARS:
        for ftype in file_types:
            filename = f"{COMPETITION}_{ftype}_{year}.json"
            file_url = f"{DATA_WEBSITE}{COMPETITION}/{year}/{filename}"
            file_path = os.path.join(base_dir, filename)

            if os.path.exists(file_path):
                print(f"Exists: {filename}")
                continue

            print(f"Downloading {filename}...")
            response = requests.get(file_url)
            if response.status_code == 200:
                try:
                    data = response.json()
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                except ValueError:
                    print(f"  -> Failed to parse JSON for {filename}")
            else:
                print(f"  -> Not found or error ({response.status_code}): {file_url}")


if __name__ == "__main__":
    download_all()
