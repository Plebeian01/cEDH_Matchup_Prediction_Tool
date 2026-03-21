# Program to automatically convert data from the TopDeck API into csv format and clean missing entries

import os
import requests
import pandas as pd
import time
import json
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("TOPDECK_API_KEY")
if not API_KEY:
    raise EnvironmentError("TOPDECK_API_KEY not set. Copy .env.example to .env and add your key.")

url = "https://topdeck.gg/api/v2/tournaments"
headers = {
    "Authorization": API_KEY
}


def ParseCommanderNames(player_data):
    """Extract commander names from a player entry.

    Prefers the structured deckObj field (available when TopDeck has parsed the
    decklist). Falls back to parsing the raw decklist string, which has the form:
        ~~Commanders~~
        1 Rograkh, Son of Rohgahh
        1 Silas Renn, Seeker Adept

        ~~Mainboard~~
        ...
    Returns names joined alphabetically as a single string, e.g.
    "Rograkh, Son of Rohgahh / Silas Renn, Seeker Adept".
    """
    # Prefer structured deckObj when available
    deck_obj = player_data.get("deckObj")
    if deck_obj and isinstance(deck_obj, dict) and "Commanders" in deck_obj:
        names = sorted(deck_obj["Commanders"].keys())
        return " / ".join(names) if names else "Unknown"

    # Fall back to parsing the decklist string
    decklist = player_data.get("decklist", "")
    if decklist and isinstance(decklist, str):
        lines = decklist.replace("\\n", "\n").split("\n")
        commanders = []
        in_commanders_section = False
        for line in lines:
            stripped = line.strip()
            if stripped == "~~Commanders~~":
                in_commanders_section = True
                continue
            if in_commanders_section:
                if stripped.startswith("~~"):  # hit the next section
                    break
                if stripped:
                    # Lines are in the form "1 Commander Name" — drop the count
                    parts = stripped.split(" ", 1)
                    if len(parts) == 2:
                        commanders.append(parts[1])
        if commanders:
            return " / ".join(sorted(commanders))

    return "Unknown"


def GetTournamentIDs():
    """Fetch a list of tournament IDs from the last 365 days."""
    data = {
        "last": 365,
        "game": "Magic: The Gathering",
        "format": "EDH",
        "participantMin": 16,
        "columns": [],  # Only the TID field is needed from this call
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching tournament IDs: {e}")
        return []

    if response.status_code == 200:
        try:
            tournaments = response.json()
        except json.JSONDecodeError:
            print("Error: Received invalid JSON when fetching tournament IDs.")
            return []
    else:
        print(f"Error {response.status_code}: {response.text}")
        return []

    return [t["TID"] for t in tournaments if "TID" in t]


def GetTournamentData(TID, retries=3):
    """Fetch standings and round data for a specific tournament ID."""
    data = {
        "TID": TID,
        "game": "Magic: The Gathering",
        "format": "EDH",
        "columns": ["decklist", "id"],
        "rounds": ["tables"],
        "tables": ["players", "winner"],
        "players": ["id"],
    }

    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
        except requests.exceptions.Timeout:
            print(f"Timeout on tournament {TID} (attempt {attempt + 1}/{retries}). Retrying...")
            time.sleep(5)
            continue
        except requests.exceptions.RequestException as e:
            print(f"Network error on tournament {TID}: {e}. Retrying...")
            time.sleep(5)
            continue

        if response.status_code == 200:
            try:
                tournament_data = response.json()
                if not tournament_data:
                    print(f"Warning: Tournament {TID} returned an empty response.")
                    return None
                return tournament_data
            except json.JSONDecodeError:
                print(f"Error: Tournament {TID} returned invalid JSON. Retrying...")

        elif response.status_code == 429:
            print("Rate limit hit. Waiting 60s before retrying.")
            time.sleep(60)
        else:
            print(f"Error {response.status_code} for tournament {TID}: {response.text}")

    print(f"Failed to retrieve tournament {TID} after {retries} attempts.")
    return None


def BuildCommanderIDDict(tournament_data):
    """Link player ID to commander name string for a specific tournament."""
    cmdr_id_dict = {}
    for t in tournament_data:
        for player in t.get("standings", []):
            player_id = player.get("id")
            if player_id:
                cmdr_id_dict[player_id] = ParseCommanderNames(player)
    return cmdr_id_dict


def ProcessRoundsData(tournament_data, player_commander_dict, TID):
    """Replace player IDs with commander names and structure the dataset."""
    structured_data = []

    for round_num, round_data in enumerate(tournament_data.get("rounds", []), start=1):
        tables_data = round_data["tables"] if isinstance(round_data, dict) else []

        for table_num, table in enumerate(tables_data, start=1):
            players = table["players"]
            winner_id = table.get("winner_id", "Unknown")

            for seat, player in enumerate(players, start=1):
                player_id = player["id"]
                commander = player_commander_dict.get(player_id, "Unknown")

                opponents = [
                    player_commander_dict.get(opp["id"], "Unknown")
                    for opp in players if opp["id"] != player_id
                ]

                if winner_id == "Draw":
                    winner_value = 2
                elif player_id == winner_id:
                    winner_value = 1
                else:
                    winner_value = 0

                structured_data.append({
                    "Tournament ID": TID,
                    "Round": round_num,
                    "Table #": table_num,
                    "Player Commander": commander,
                    "Seat": seat,
                    "Opponent 1": opponents[0] if len(opponents) > 0 else "Unknown",
                    "Opponent 2": opponents[1] if len(opponents) > 1 else "Unknown",
                    "Opponent 3": opponents[2] if len(opponents) > 2 else "Unknown",
                    "Winner": winner_value,
                })

    return structured_data


# Fetch tournament IDs
tournamentIDs = GetTournamentIDs()

dataset = []
for count, TID in enumerate(tournamentIDs, start=1):
    tournament_data = GetTournamentData(TID)
    if tournament_data:
        player_commander_dict = BuildCommanderIDDict(tournament_data)
        structured_matches = ProcessRoundsData(tournament_data[0], player_commander_dict, TID)
        dataset.extend(structured_matches)

    print(f"Tournament {TID} processed. {len(tournamentIDs) - count} remaining.")
    time.sleep(3)

# Convert to DataFrame & save to CSV
df_structured = pd.DataFrame(dataset)

df_filtered = df_structured[
    (df_structured["Player Commander"] != "Unknown") &
    (df_structured["Opponent 1"] != "Unknown") &
    (df_structured["Opponent 2"] != "Unknown") &
    (df_structured["Opponent 3"] != "Unknown")
]

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "511_tournament_data.csv")
df_filtered.to_csv(output_path, index=False)
print(f"Successfully saved structured tournament data to {output_path}")
