import pandas as pd
import torch
import torch.nn.functional as F
import pickle
from ML2 import DrawModel, WinLossModel, build_matchup_stats, apply_matchup_features

# --- SETTINGS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LOAD MODELS ---
with open("deck_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

NUM_DECKS = len(encoder.classes_)

draw_model = DrawModel(num_decks=NUM_DECKS).to(DEVICE)
draw_model.load_state_dict(torch.load("best_draw_model.pth", map_location=DEVICE, weights_only=True))
draw_model.eval()

seat_model = WinLossModel(num_decks=NUM_DECKS).to(DEVICE)
seat_model.load_state_dict(torch.load("best_seat_model.pth", map_location=DEVICE, weights_only=True))
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

for col in ["deck1", "deck2", "deck3", "deck4"]:
    df_new[col] = encoder.transform(df_new[col])

# Apply synergy features
df_new = apply_matchup_features(df_new, matchup_stats)

# --- PREDICT ---
deck_arr = df_new[["deck1", "deck2", "deck3", "deck4"]].values
synergy_arr = df_new[["matchup_12", "matchup_13", "matchup_14",
                       "matchup_23", "matchup_24", "matchup_34"]].values

predictions = []
for i in range(len(df_new)):
    decks = torch.tensor([deck_arr[i]], dtype=torch.long, device=DEVICE)
    synergy = torch.tensor([synergy_arr[i]], dtype=torch.float32, device=DEVICE)

    # Step 1: Predict Draw Probability
    with torch.no_grad():
        draw_logits = draw_model(decks, synergy)
        draw_probs = F.softmax(draw_logits, dim=1)
        p_draw = draw_probs[0, 1].item()

    # Step 2: Predict Seat Probabilities
    with torch.no_grad():
        seat_logits = seat_model(decks, synergy)
        seat_probs = F.softmax(seat_logits, dim=1).cpu().numpy()[0]

    final_output = {
        "p_draw": p_draw,
        "p_seat0": seat_probs[0],
        "p_seat1": seat_probs[1],
        "p_seat2": seat_probs[2],
        "p_seat3": seat_probs[3],
    }

    predictions.append(final_output)

    print(f"Game {i}: Seat1={final_output['p_seat0']:.3f}, Seat2={final_output['p_seat1']:.3f}, "
          f"Seat3={final_output['p_seat2']:.3f}, Seat4={final_output['p_seat3']:.3f}, Draw={final_output['p_draw']:.3f}")

# --- SAVE TO CSV ---
df_new["p_seat1"] = [p["p_seat0"] for p in predictions]
df_new["p_seat2"] = [p["p_seat1"] for p in predictions]
df_new["p_seat3"] = [p["p_seat2"] for p in predictions]
df_new["p_seat4"] = [p["p_seat3"] for p in predictions]
df_new["p_draw"]  = [p["p_draw"] for p in predictions]

df_new.to_csv("predicted_new_tournament.csv", index=False)
print("Predictions saved to predicted_new_tournament.csv")
