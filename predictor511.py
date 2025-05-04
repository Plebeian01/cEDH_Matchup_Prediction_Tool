# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 17:07:54 2025

@author: freez
"""
import pandas as pd
import torch
import torch.nn.functional as F
import pickle
import numpy as np
from ML2 import DrawModel, WinLossModel, build_matchup_stats, apply_matchup_features  # Ensure these are imported correctly

# --- SETTINGS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LOAD MODELS ---
with open("deck_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

NUM_DECKS = len(encoder.classes_)

draw_model = DrawModel(num_decks=NUM_DECKS).to(DEVICE)
draw_model.load_state_dict(torch.load("best_draw_model.pth", map_location=DEVICE))
draw_model.eval()

seat_model = WinLossModel(num_decks=NUM_DECKS).to(DEVICE)
seat_model.load_state_dict(torch.load("best_seat_model.pth", map_location=DEVICE))
seat_model.eval()

# --- LOAD HISTORICAL DATA FOR SYNERGY ---
df_history = pd.read_csv("processed_tournament_data.csv")
matchup_stats = build_matchup_stats(df_history)

# --- LOAD AND PREPROCESS NEW TOURNAMENT DATA ---
df_new = pd.read_csv("new_tournament_data.csv")

# Rename columns if needed
if "Commander 1" in df_new.columns:
    df_new = df_new.rename(columns={
        "Commander 1": "deck1",
        "Commander 2": "deck2",
        "Commander 3": "deck3",
        "Commander 4": "deck4"
    })

# Handle unknown decks + encode
for col in ["deck1", "deck2", "deck3", "deck4"]:
    df_new[col] = df_new[col].apply(lambda x: x if x in encoder.classes_ else "Unknown")

#if "Unknown" not in encoder.classes_:
#    encoder.classes_ = np.append(encoder.classes_, "Unknown")
    
print("Number of decks:", len(encoder.classes_))
print("Is 'Unknown' in encoder?", "Unknown" in encoder.classes_)

for col in ["deck1", "deck2", "deck3", "deck4"]:
    df_new[col] = encoder.transform(df_new[col])

# Apply synergy features
df_new = apply_matchup_features(df_new, matchup_stats)

print("Max deck index in df_new:", df_new[["deck1","deck2","deck3","deck4"]].max().max())
print("NUM_DECKS in model:", NUM_DECKS)

# --- PREDICT ---
predictions = []
for idx, row in df_new.iterrows():
    decks = torch.tensor([[row["deck1"], row["deck2"], row["deck3"], row["deck4"]]], dtype=torch.long, device=DEVICE)
    synergy = torch.tensor([[row["matchup_12"], row["matchup_13"], row["matchup_14"],
                             row["matchup_23"], row["matchup_24"], row["matchup_34"]]],
                           dtype=torch.float32, device=DEVICE)

    # Step 1: Predict Draw Probability
    with torch.no_grad():
        draw_logits = draw_model(decks, synergy)
        draw_probs = F.softmax(draw_logits, dim=1)
        p_draw = draw_probs[0,1].item()
        p_notdraw = draw_probs[0,0].item()

    # Step 2: Predict Seat Probabilities
    with torch.no_grad():
        seat_logits = seat_model(decks, synergy)
        seat_probs = F.softmax(seat_logits, dim=1).cpu().numpy()[0]

    # Final Distribution
    final_output = {
        "p_draw": p_draw,
        "p_seat0": seat_probs[0],
        "p_seat1": seat_probs[1],
        "p_seat2": seat_probs[2],
        "p_seat3": seat_probs[3],
    }

    predictions.append(final_output)

    print(f"Game {idx}: Seat1={final_output['p_seat0']:.3f}, Seat2={final_output['p_seat1']:.3f}, Seat3={final_output['p_seat2']:.3f}, Seat4={final_output['p_seat3']:.3f}, Draw={final_output['p_draw']:.3f}")

# --- SAVE TO CSV ---
df_new["p_seat1"] = [p["p_seat0"] for p in predictions]
df_new["p_seat2"] = [p["p_seat1"] for p in predictions]
df_new["p_seat3"] = [p["p_seat2"] for p in predictions]
df_new["p_seat4"] = [p["p_seat3"] for p in predictions]
df_new["p_draw"]  = [p["p_draw"] for p in predictions]

df_new.to_csv("predicted_new_tournament.csv", index=False)
print("Predictions saved to predicted_new_tournament.csv")
