# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:35:22 2024

@author: freez
"""

#Program to automatically convert data from the TopDeck API into csv format and clean missing entries

import requests
import pandas as pd
import time
import json
import re

API_KEY = "472bc6b8-8b76-4b60-9d44-3db14ce71b3f"

url = "https://topdeck.gg/api/v2/tournaments"
headers = {
    "Authorization": API_KEY
}

def StandardizeCmdrNames(names): #Joins commander entry as a single string, with alphabetical sorting for partner commanders
    if names:
        rename = " / ".join(sorted(names))
        return rename
    else:
        return "Unknown"
        

def GetTournamentIDs(): #Fetch a list of tournament IDs from the API
    data = {
        "last": 90,
        "game": "Magic: The Gathering",
        "format": "EDH",
        "participantMin": 16,
        "columns": [None],
    }
    
    response = requests.post(url, headers=headers, json=data)
    
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

def GetTournamentData(TID, retries=3): #Fetch data for a specific tournament ID
    data = {
        "TID": TID,
        "game": "Magic: The Gathering",
        "format": "EDH",
        "columns": ["commanders", "id"],
        "rounds": ["tables"],
        "tables": ["players", "winner"],
        "players": ["id"],
    }
    
    for attempt in range(retries):  #retry in case of error
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            try:
                tournament_data = response.json()
                if not tournament_data:
                    print(f"Warning: Tournament {TID} returned an empty response.")
                    return None
                return tournament_data
            except json.JSONDecodeError:
                print(f"Error: Tournament {TID} returned invalid JSON. Retrying...")
        
        elif response.status_code == 429:  # Rate limit exceeded
            print("Rate limit hit. Waiting 60s before retrying.")
            time.sleep(60)  # Wait 60 seconds before retrying
        else:
            print(f"Error {response.status_code} for tournament {TID}: {response.text}")

    print(f"Failed to retrieve tournament {TID} after {retries} attempts.")
    return None

def BuildCommanderIDDict(tournament_data):  #link player id to commander for a specific tournament
    cmdr_id_pairs = []
    for t in tournament_data:
        for pair in t["standings"]:
            cmdr_id_pairs.append([pair["id"], StandardizeCmdrNames(pair["commanders"])])
    return dict(cmdr_id_pairs)
        

def ProcessRoundsData(tournament_data, player_commander_dict, TID):
    """ Replaces Player IDs with Commander names in the rounds data and structures the dataset. """
    structured_data = []
    
    for round_num, round_data in enumerate(tournament_data.get("rounds", []), start=1):
        # Ensure round_data is a dictionary before accessing "tables"
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
                    winner_value = 2  # ✅ Encode Draw as 2
                elif player_id == winner_id:
                    winner_value = 1  # ✅ Win = 1
                else:
                    winner_value = 0  # ✅ Loss = 0

                structured_data.append({
                    "Tournament ID": TID,
                    "Round": round_num,
                    "Table #": table_num,
                    "Player Commander": commander,
                    "Seat": seat,
                    "Opponent 1": opponents[0] if len(opponents) > 0 else "Unknown", #0=loss
                    "Opponent 2": opponents[1] if len(opponents) > 1 else "Unknown", #1=win
                    "Opponent 3": opponents[2] if len(opponents) > 2 else "Unknown", #2=draw
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

# Convert to DataFrame & Save to CSV
df_structured = pd.DataFrame(dataset)

df_filtered = df_structured[
    (df_structured["Player Commander"] != 'Unknown') &
    (df_structured["Opponent 1"] != 'Unknown') &
    (df_structured["Opponent 2"] != 'Unknown') &
    (df_structured["Opponent 3"] != 'Unknown')
]

df_filtered.to_csv("processed_tournament_data_test.csv", index=False)
print("Successfully saved structured tournament data to processed_tournament_data.csv")
    