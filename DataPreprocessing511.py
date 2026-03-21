import os

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

### PREPROCESSING ###
# Load data
script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(script_dir, "511_tournament_data.csv"))
print(f"Raw rows loaded:              {len(df)}")

# Filter out unknown commander entries
df = df[
    (df["Player Commander"] != "Unknown") &
    (df["Opponent 1"] != "Unknown") &
    (df["Opponent 2"] != "Unknown") &
    (df["Opponent 3"] != "Unknown")
]
print(f"After Unknown filter:         {len(df)} ({len(df)/171000*100:.1f}% of raw)")

# Get commander frequency, then compute logarithmic cutoff
commander_counts = df["Player Commander"].value_counts()
log_cutoff = np.log1p(commander_counts).mean() * 10
print(f"Unique commanders:            {len(commander_counts)}")
print(f"Log-Based Cutoff:             {int(log_cutoff)} entries")
valid_commanders = commander_counts[commander_counts >= log_cutoff].index
print(f"Commanders passing cutoff:    {len(valid_commanders)} / {len(commander_counts)}")

# Apply the cutoff
min_appearance = log_cutoff
df_filtered = df[df["Player Commander"].isin(valid_commanders)]
df_filtered = df_filtered[df_filtered["Opponent 1"].isin(valid_commanders)]
df_filtered = df_filtered[df_filtered["Opponent 2"].isin(valid_commanders)]
df_filtered = df_filtered[df_filtered["Opponent 3"].isin(valid_commanders)]
print(f"After frequency cutoff:       {len(df_filtered)}")

# Create one row per game using seat/winner perspective
player_cmdr = df_filtered["Player Commander"].values
opp1 = df_filtered["Opponent 1"].values
opp2 = df_filtered["Opponent 2"].values
opp3 = df_filtered["Opponent 3"].values
winner = df_filtered["Winner"].values
seat = df_filtered["Seat"].values

tables = []
for pc, o1, o2, o3, w, s in zip(player_cmdr, opp1, opp2, opp3, winner, seat):
    if w == 1:
        if s == 1:
            tables.append([pc, o1, o2, o3, 0])
        elif s == 2:
            tables.append([o1, pc, o2, o3, 1])
        elif s == 3:
            tables.append([o1, o2, pc, o3, 2])
        elif s == 4:
            tables.append([o1, o2, o3, pc, 3])
    elif w == 2 and s == 1:
        tables.append([pc, o1, o2, o3, 4])

labels = ["deck1", "deck2", "deck3", "deck4", "outcome"]
df_singled_tables = pd.DataFrame(tables, columns=labels)


### Label ENCODING ###
# Initialize Label Encoder and fit on all commanders
encoder = LabelEncoder()
encoder.fit(valid_commanders)

# Warn about any unseen commanders before transforming
for col in ["deck1", "deck2", "deck3", "deck4"]:
    unseen = set(df_singled_tables[col].unique()) - set(encoder.classes_)
    if unseen:
        print(f"WARNING: Unseen commanders in {col} before transforming:", unseen)

# Replace unseen commanders with "Unknown" before transforming
for col in ["deck1", "deck2", "deck3", "deck4"]:
    df_singled_tables[col] = df_singled_tables[col].apply(lambda x: x if x in encoder.classes_ else "Unknown")

# Encode commanders
for col in ["deck1", "deck2", "deck3", "deck4"]:
    df_singled_tables[col] = encoder.transform(df_singled_tables[col])

df_encoded = df_singled_tables[["deck1", "deck2", "deck3", "deck4", "outcome"]]

# Save the encoded dataset
output_path = os.path.join(script_dir, "processed_tournament_data.csv")
df_encoded.to_csv(output_path, index=False)
print(f"Filtered dataset saved! Kept {len(df_encoded)} rows ({len(df_filtered)} before table deduplication).")

# Save encoder to pickle for use in matchup predictor
encoder_path = os.path.join(script_dir, "deck_encoder.pkl")
with open(encoder_path, "wb") as f:
    pickle.dump(encoder, f)
