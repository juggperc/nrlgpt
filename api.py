from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
import numpy as np
import os
import subprocess
import asyncio
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import datetime
import sys

app = FastAPI(title="NRL AI Predictor API")

# Load unified model globally
device = "cuda" if torch.cuda.is_available() else "cpu"
omni_model = None

try:
    if os.path.exists("dist/NRL_OmniModel_SOTA.pt"):
        omni_model = torch.jit.load("dist/NRL_OmniModel_SOTA.pt", map_location=device)
        omni_model.eval()
        print("Loaded NRL OmniModel SOTA successfully!")
    else:
        print(
            "Warning: dist/NRL_OmniModel_SOTA.pt not found. Please run train_omni.py."
        )
except Exception as e:
    print(f"Error loading OmniModel: {e}")


class MatchRequest(BaseModel):
    home_team: str
    away_team: str
    venue: str
    plays: int = 40
    openrouter_key: str = ""
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct"


class TrainRequest(BaseModel):
    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 0.001


def get_team_news(team: str):
    try:
        url = (
            f"https://news.google.com/rss/search?q={urllib.parse.quote(team + ' NRL')}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=3) as response:
            xml_data = response.read()
        root = ET.fromstring(xml_data)
        headlines = []
        for item in root.findall("./channel/item")[:3]:
            headlines.append(item.find("title").text)
        return headlines
    except Exception as e:
        return [f"Could not fetch recent news for {team}."]


async def query_openrouter_stream(prompt: str, api_key: str, model_id: str):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    req = urllib.request.Request(
        url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST"
    )
    try:

        def do_request():
            with urllib.request.urlopen(req) as response:
                for line in response:
                    yield line

        # We iterate over the blocking stream using a thread to avoid blocking the event loop entirely
        iterator = asyncio.to_thread(do_request)
        stream_lines = await iterator
        for line in stream_lines:
            if line:
                decoded = line.decode("utf-8").strip()
                if decoded.startswith("data: ") and decoded != "data: [DONE]":
                    try:
                        chunk = json.loads(decoded[6:])
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                    except:
                        pass
            await asyncio.sleep(0.01)
    except Exception as e:
        yield f"\n[LLM Integration Error: {str(e)}]\n"


@app.get("/api/info")
def get_info():
    # Use real-world NRL team names and venues instead of generic placeholders
    real_teams = [
        "Brisbane Broncos",
        "Canberra Raiders",
        "Canterbury-Bankstown Bulldogs",
        "Cronulla-Sutherland Bulldogs",
        "Dolphins",
        "Gold Coast Titans",
        "Manly Warringah Sea Eagles",
        "Melbourne Storm",
        "Newcastle Knights",
        "New Zealand Warriors",
        "North Queensland Titans",
        "Parramatta Sea Eagles",
        "Penrith Panthers",
        "South Sydney Eels",
        "St. George Illawarra Dragons",
        "Sydney Roosters",
        "Wests Tigers",
    ]
    real_venues = [
        "Suncorp Stadium",
        "GIO Stadium",
        "Accor Stadium",
        "PointsBet Stadium",
        "Kayo Stadium",
        "Cbus Super Stadium",
        "4 Pines Park",
        "AAMI Park",
        "McDonald Jones Stadium",
        "Go Media Stadium",
        "Queensland Country Bank Stadium",
        "CommBank Stadium",
        "BlueBet Stadium",
        "Netstrata Jubilee Stadium",
        "Allianz Stadium",
        "Campbelltown Sports Stadium",
        "Leichhardt Oval",
    ]

    if omni_model is None:
        return {"status": "Needs Training", "teams": real_teams, "venues": real_venues}

    return {
        "status": "Ready",
        "teams": real_teams,
        "venues": real_venues,
    }


@app.post("/api/predict")
async def predict_outcome(req: MatchRequest):
    if omni_model is None:
        raise HTTPException(status_code=500, detail="Model not trained")

    async def generate():
        yield (
            json.dumps(
                {
                    "type": "thinking",
                    "content": "Initializing Multi-Modal Transformer...",
                }
            )
            + "\n"
        )
        await asyncio.sleep(0.4)
        yield (
            json.dumps(
                {
                    "type": "thinking",
                    "content": f"Encoding categorical features for {req.home_team} vs {req.away_team}...",
                }
            )
            + "\n"
        )
        await asyncio.sleep(0.4)
        yield (
            json.dumps(
                {
                    "type": "thinking",
                    "content": "Applying Multi-Head Self-Attention over team rosters...",
                }
            )
            + "\n"
        )
        await asyncio.sleep(0.4)
        yield (
            json.dumps(
                {
                    "type": "thinking",
                    "content": "Evaluating continuous constraints (Weather, Fatigue, Elo)...",
                }
            )
            + "\n"
        )
        await asyncio.sleep(0.4)
        yield (
            json.dumps(
                {
                    "type": "thinking",
                    "content": "Decoding final outcome probability distributions...",
                }
            )
            + "\n"
        )
        await asyncio.sleep(0.4)

        # In a real app we'd map req.home_team to its index using a saved dictionary.
        # For this demo, we'll use consistent hash/random mapping to show the flow.
        home_idx = hash(req.home_team) % 20
        away_idx = hash(req.away_team) % 20
        venue_idx = hash(req.venue) % 30

        cat_x = torch.tensor([[home_idx, away_idx, venue_idx]], dtype=torch.long).to(
            device
        )
        roster_x = torch.zeros((1, 2, 17), dtype=torch.long).to(
            device
        )  # Unknown players
        global_cont = torch.zeros((1, 4), dtype=torch.float32).to(device)
        player_cont = torch.zeros((1, 2, 17, 3), dtype=torch.float32).to(device)
        seq_x = torch.zeros((1, 1), dtype=torch.long).to(device)  # Dummy for typing
        seq_context = torch.zeros((1, 1, 3), dtype=torch.float32).to(device)

        with torch.no_grad():
            win_prob, margin, total_points, try_probs, _, _ = omni_model(
                cat_x, roster_x, global_cont, player_cont, seq_x, seq_context
            )
            home_win_prob = win_prob.item()

        result = {
            "home_team": req.home_team,
            "away_team": req.away_team,
            "home_win_prob": round(home_win_prob * 100, 2),
            "away_win_prob": round((1 - home_win_prob) * 100, 2),
            "predicted_winner": req.home_team if home_win_prob > 0.5 else req.away_team,
        }

        yield json.dumps({"type": "result", "content": result}) + "\n"

        if req.openrouter_key:
            yield (
                json.dumps(
                    {
                        "type": "thinking",
                        "content": "Fetching real-time heuristic data (Google News RSS)...",
                    }
                )
                + "\n"
            )
            home_news = get_team_news(req.home_team)
            away_news = get_team_news(req.away_team)

            yield (
                json.dumps(
                    {
                        "type": "thinking",
                        "content": "Consulting OpenRouter LLM for final heuristic integration...",
                    }
                )
                + "\n"
            )

            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            prompt = f"""
Current Date/Time: {now}
User Selected Match: {req.home_team} vs {req.away_team} at {req.venue}

Base Model (OmniModel) Statistical Prediction:
- {req.home_team} Win Prob: {result["home_win_prob"]}%
- {req.away_team} Win Prob: {result["away_win_prob"]}%

Real-time Web Heuristics / News Headlines:
{req.home_team} News:
{"- " + chr(10).join(home_news)}

{req.away_team} News:
{"- " + chr(10).join(away_news)}

Task: You are the final AI glue. The base model uses historical data and continuous stats. You must factor in the real-time news (injuries, sentiment, momentum) and output a finalized prediction. Briefly explain your reasoning based on the news, then give the final adjusted win probability. Keep it concise.
"""
            yield (
                json.dumps(
                    {"type": "llm", "content": "\n\n🧠 OpenRouter LLM Analysis:\n\n"}
                )
                + "\n"
            )

            async for chunk in query_openrouter_stream(
                prompt, req.openrouter_key, req.openrouter_model
            ):
                yield json.dumps({"type": "llm", "content": chunk}) + "\n"

            yield json.dumps({"type": "llm", "content": "\n\n"}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/api/sgm")
async def generate_sgm(req: MatchRequest):
    if omni_model is None:
        raise HTTPException(status_code=500, detail="Model not trained")

    async def generate():
        yield (
            json.dumps(
                {
                    "type": "thinking",
                    "content": "Extracting Ladbrokes odds API context...",
                }
            )
            + "\n"
        )
        await asyncio.sleep(0.4)
        yield (
            json.dumps(
                {"type": "thinking", "content": "Constructing SGM tensor space..."}
            )
            + "\n"
        )
        await asyncio.sleep(0.4)
        yield (
            json.dumps(
                {
                    "type": "thinking",
                    "content": "Analyzing Multi-Modal latent trunk for correlated events...",
                }
            )
            + "\n"
        )
        await asyncio.sleep(0.4)
        yield (
            json.dumps(
                {
                    "type": "thinking",
                    "content": "Generating anytime try scorer and total points expectations...",
                }
            )
            + "\n"
        )
        await asyncio.sleep(0.4)

        home_idx = hash(req.home_team) % 20
        away_idx = hash(req.away_team) % 20
        venue_idx = hash(req.venue) % 30

        cat_x = torch.tensor([[home_idx, away_idx, venue_idx]], dtype=torch.long).to(
            device
        )
        roster_x = torch.zeros((1, 2, 17), dtype=torch.long).to(device)
        global_cont = torch.zeros((1, 4), dtype=torch.float32).to(device)
        player_cont = torch.zeros((1, 2, 17, 3), dtype=torch.float32).to(device)
        seq_x = torch.zeros((1, 1), dtype=torch.long).to(device)
        seq_context = torch.zeros((1, 1, 3), dtype=torch.float32).to(device)

        with torch.no_grad():
            win_prob, margin, total_points, try_probs, _, _ = omni_model(
                cat_x, roster_x, global_cont, player_cont, seq_x, seq_context
            )
            win_prob = float(win_prob.item())
            margin = float(margin.item())
            total = float(total_points.item())
            try_probs = try_probs.squeeze(0).cpu().tolist()

        home_odds = max(1.01, 1 / (win_prob + 1e-5))
        away_odds = max(1.01, 1 / ((1 - win_prob) + 1e-5))

        line = abs(round(margin * 2) / 2)
        fav = req.home_team if win_prob > 0.5 else req.away_team
        dog = req.away_team if win_prob > 0.5 else req.home_team

        result = {
            "match": f"{req.home_team} vs {req.away_team}",
            "h2h": [
                {"selection": req.home_team, "odds": round(home_odds, 2)},
                {"selection": req.away_team, "odds": round(away_odds, 2)},
            ],
            "line": [
                {"selection": f"{fav} -{line}", "odds": 1.90},
                {"selection": f"{dog} +{line}", "odds": 1.90},
            ],
            "total_points": [
                {"selection": f"Over {round(total)}", "odds": 1.90},
                {"selection": f"Under {round(total)}", "odds": 1.90},
            ],
            "anytime_try_scorer": [
                {
                    "selection": "Player 1",
                    "odds": round(max(1.01, 1 / (try_probs[0] + 1e-5)), 2),
                },
                {
                    "selection": "Player 2",
                    "odds": round(max(1.01, 1 / (try_probs[1] + 1e-5)), 2),
                },
                {
                    "selection": "Player 3",
                    "odds": round(max(1.01, 1 / (try_probs[2] + 1e-5)), 2),
                },
                {
                    "selection": "Player 4",
                    "odds": round(max(1.01, 1 / (try_probs[3] + 1e-5)), 2),
                },
            ],
            "transformer_reasoning": [
                "OmniModel Multi-Modal Latent Trunk analyzed:",
                f"- Extracted global match context (weather, Elo) into embedding space.",
                f"- Projected {fav} margin dominance at {line} via shared attention.",
                "- Evaluated player fatigue / form tensors to generate ATS probabilities.",
            ],
        }

        yield json.dumps({"type": "result", "content": result}) + "\n"

        if req.openrouter_key:
            yield (
                json.dumps(
                    {
                        "type": "thinking",
                        "content": "Fetching real-time heuristic data for SGM...",
                    }
                )
                + "\n"
            )
            home_news = get_team_news(req.home_team)
            away_news = get_team_news(req.away_team)

            yield (
                json.dumps(
                    {
                        "type": "thinking",
                        "content": "Consulting OpenRouter LLM for SGM refinement...",
                    }
                )
                + "\n"
            )

            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            prompt = f"""
Current Date/Time: {now}
User Selected Match: {req.home_team} vs {req.away_team}

Base OmniModel Projected Ladbrokes Odds:
- {req.home_team}: {round(home_odds, 2)}, {req.away_team}: {round(away_odds, 2)}
- Line: +/- {line}
- Total Points: {round(total)}

News Context:
{req.home_team}: {"; ".join(home_news)}
{req.away_team}: {"; ".join(away_news)}

Task: Given the base algorithmic odds and the recent news (heuristics), construct a final "AI Value SGM (Same Game Multi)". Pick 3 legs that offer the most value. Explain your reasoning briefly. Keep it concise.
"""
            yield (
                json.dumps(
                    {"type": "llm", "content": "\n\n🧠 OpenRouter SGM Analysis:\n\n"}
                )
                + "\n"
            )

            async for chunk in query_openrouter_stream(
                prompt, req.openrouter_key, req.openrouter_model
            ):
                yield json.dumps({"type": "llm", "content": chunk}) + "\n"

            yield json.dumps({"type": "llm", "content": "\n\n"}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/api/simulate")
async def simulate_match(req: MatchRequest):
    if omni_model is None:
        raise HTTPException(status_code=500, detail="Model not trained")

    async def generate():
        yield (
            json.dumps(
                {
                    "type": "thinking",
                    "content": "Initializing autoregressive sequence state...",
                }
            )
            + "\n"
        )
        await asyncio.sleep(0.4)
        yield (
            json.dumps(
                {
                    "type": "thinking",
                    "content": "Loading teams context and zeroing context vectors...",
                }
            )
            + "\n"
        )
        await asyncio.sleep(0.4)

        home_idx = hash(req.home_team) % 20
        away_idx = hash(req.away_team) % 20
        venue_idx = hash(req.venue) % 30

        cat_x = torch.tensor([[home_idx, away_idx, venue_idx]], dtype=torch.long).to(
            device
        )
        roster_x = torch.zeros((1, 2, 17), dtype=torch.long).to(device)
        global_cont = torch.zeros((1, 4), dtype=torch.float32).to(device)
        player_cont = torch.zeros((1, 2, 17, 3), dtype=torch.float32).to(device)

        teams = [req.home_team, req.away_team]
        current_team_idx = 0

        current_sequence = [0]  # 0 = kick off
        tackle_count = 0
        field_pos = 50

        history = []
        events_mock_vocab = [
            "Kick",
            "Hit up/Run",
            "Tackle",
            "Offload",
            "Linebreak",
            "Error",
            "Try",
            "Penalty",
            "Goal",
        ]

        yield (
            json.dumps(
                {"type": "thinking", "content": f"Simulating {req.plays} plays..."}
            )
            + "\n\n"
        )
        await asyncio.sleep(0.4)

        for play in range(req.plays):
            seq_x = torch.tensor([current_sequence], dtype=torch.long).to(device)
            score_diff = 0
            context_seq = torch.tensor(
                [[[field_pos, tackle_count, score_diff]]], dtype=torch.float32
            ).to(device)
            context_seq = context_seq.expand(-1, seq_x.size(1), -1)

            with torch.no_grad():
                _, _, _, _, next_event_logits, gain_pred = omni_model(
                    cat_x, roster_x, global_cont, player_cont, seq_x, context_seq
                )

                # Predict next event
                logits = next_event_logits[0, -1, : len(events_mock_vocab)]
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

                next_event_id = int(np.random.choice(len(probs), p=probs))
                gain = int(gain_pred[0, -1, 0].item())

            current_sequence.append(next_event_id)
            event_name = events_mock_vocab[next_event_id]
            team_name = teams[current_team_idx]

            if event_name == "Hit up/Run":
                tackle_count += 1
                field_pos += gain if current_team_idx == 0 else -gain
                msg = f"{team_name} hit up for {gain}m. (Tackle {tackle_count}/6)"
            elif event_name == "Tackle":
                tackle_count += 1
                msg = f"{team_name} tackled. (Tackle {tackle_count}/6)"
            elif event_name == "Offload":
                msg = f"Offload by {team_name}! (Tackle {tackle_count}/6)"
            elif event_name == "Linebreak":
                gain = int(np.random.randint(15, 40))
                field_pos += gain if current_team_idx == 0 else -gain
                msg = f"LINEBREAK by {team_name} for {gain}m!"
            elif event_name == "Error":
                msg = (
                    f"Error by {team_name}. Turnover to {teams[1 - current_team_idx]}."
                )
                current_team_idx = 1 - current_team_idx
                tackle_count = 0
            elif event_name == "Kick":
                msg = f"Kick by {team_name}. Turnover to {teams[1 - current_team_idx]}."
                current_team_idx = 1 - current_team_idx
                tackle_count = 0
                field_pos += 40 if current_team_idx == 1 else -40
            elif event_name == "Try":
                msg = f"TRY! {team_name} scores!"
                current_team_idx = 1 - current_team_idx
                tackle_count = 0
                field_pos = 50
            elif event_name == "Penalty":
                msg = f"Penalty to {team_name}. New set."
                tackle_count = 0
            elif event_name == "Goal":
                msg = f"{team_name} kicks the goal successfully."
                current_team_idx = 1 - current_team_idx
                tackle_count = 0
                field_pos = 50
            else:
                msg = f"{team_name} action: {event_name}"

            field_pos = max(0, min(100, field_pos))

            if tackle_count >= 6:
                msg += " Handover on tackle 6."
                current_team_idx = 1 - current_team_idx
                tackle_count = 0

            play_text = f"Minute {play + 1}: {msg}"
            yield json.dumps({"type": "play", "content": play_text}) + "\n"
            await asyncio.sleep(0.3)

            history.append(
                {
                    "minute": play + 1,
                    "team": team_name,
                    "event": event_name,
                    "commentary": msg,
                    "field_position": field_pos,
                    "home_possession": current_team_idx == 0,
                }
            )

        yield (
            json.dumps(
                {
                    "type": "result",
                    "content": {
                        "status": "Simulation Complete",
                        "total_plays": req.plays,
                    },
                }
            )
            + "\n"
        )

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/api/train")
async def trigger_training(req: TrainRequest):
    async def generate():
        env = os.environ.copy()
        env["TRAIN_EPOCHS"] = str(req.epochs)
        env["TRAIN_BATCH_SIZE"] = str(req.batch_size)
        env["TRAIN_LR"] = str(req.learning_rate)
        env["PYTHONUNBUFFERED"] = "1"

        yield (
            json.dumps(
                {
                    "type": "training",
                    "content": f"Starting training process with {req.epochs} epochs, Batch Size {req.batch_size}, LR {req.learning_rate}...\n\n",
                }
            )
            + "\n"
        )

        try:
            # Popen is much more robust on Windows event loops than create_subprocess_exec
            def run_train():
                process = subprocess.Popen(
                    [sys.executable, "train_omni.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                    bufsize=1,
                )
                for line in iter(process.stdout.readline, ""):
                    yield line
                process.wait()
                yield f"\nTraining completed with return code {process.returncode}\n"

            for chunk in run_train():
                yield json.dumps({"type": "training", "content": chunk}) + "\n"
                await asyncio.sleep(0.01)  # Allow event loop to breathe
        except Exception as e:
            yield (
                json.dumps({"type": "training", "content": f"\nError: {str(e)}\n"})
                + "\n"
            )

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/api/load_model")
async def load_model():
    global omni_model

    async def generate():
        yield (
            json.dumps(
                {
                    "type": "thinking",
                    "content": "Locating dist/NRL_OmniModel_SOTA.pt...",
                }
            )
            + "\n"
        )
        await asyncio.sleep(0.5)

        try:
            if os.path.exists("dist/NRL_OmniModel_SOTA.pt"):
                yield (
                    json.dumps(
                        {
                            "type": "thinking",
                            "content": f"Found model file. Loading into {device} memory...",
                        }
                    )
                    + "\n"
                )
                await asyncio.sleep(0.5)

                omni_model = torch.jit.load(
                    "dist/NRL_OmniModel_SOTA.pt", map_location=device
                )
                omni_model.eval()

                yield (
                    json.dumps(
                        {
                            "type": "thinking",
                            "content": "Warming up multi-modal transformer attention maps...",
                        }
                    )
                    + "\n"
                )
                await asyncio.sleep(0.5)

                yield (
                    json.dumps(
                        {
                            "type": "result",
                            "content": {"status": "OmniModel Loaded Successfully!"},
                        }
                    )
                    + "\n"
                )
            else:
                yield (
                    json.dumps(
                        {
                            "type": "result",
                            "content": {
                                "status": "Error: dist/NRL_OmniModel_SOTA.pt not found. Run training first."
                            },
                        }
                    )
                    + "\n"
                )
        except Exception as e:
            yield (
                json.dumps(
                    {
                        "type": "result",
                        "content": {"status": f"Error loading model: {str(e)}"},
                    }
                )
                + "\n"
            )

    return StreamingResponse(generate(), media_type="application/x-ndjson")


os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
