# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 18:18:05 2025

@author: freez
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import os

### PREPROCESSING ###
# load Data
df = pd.read_csv("511_tournament_data.csv")

# filter out unknown commander entries
df = df[
    (df["Player Commander"] != "Unknown") &
    (df["Opponent 1"] != "Unknown") &
    (df["Opponent 2"] != "Unknown") &
    (df["Opponent 3"] != "Unknown")
]

# get commander frequency, then compute logarithmic cutoff
commander_counts = df["Player Commander"].value_counts()
log_cutoff = np.log1p(commander_counts).mean() * 10
print(f"Log-Based Cutoff: {int(log_cutoff)} entries")

# Apply the cutoff
min_appearance = log_cutoff
valid_commanders = commander_counts[commander_counts >= min_appearance].index
df_filtered = df[df["Player Commander"].isin(valid_commanders)]
df_filtered = df_filtered[df_filtered["Opponent 1"].isin(valid_commanders)]
df_filtered = df_filtered[df_filtered["Opponent 2"].isin(valid_commanders)]
df_filtered = df_filtered[df_filtered["Opponent 3"].isin(valid_commanders)]

#Save Legacy Dataset (Testing Purposes Only)
df_filtered.to_csv("processed_tournament_data_LEGACY.csv")

# Create one row for each game 
tables = []
for index, row in df_filtered.iterrows():
    if row["Winner"] == 1:
        if row["Seat"] == 1:
            tables.append([row["Player Commander"], row["Opponent 1"], row["Opponent 2"], row["Opponent 3"], 0])
        elif row["Seat"] == 2:
            tables.append([row["Opponent 1"], row["Player Commander"], row["Opponent 2"], row["Opponent 3"], 1])
        elif row["Seat"] == 3:
            tables.append([row["Opponent 1"], row["Opponent 2"], row["Player Commander"], row["Opponent 3"], 2])
        elif row["Seat"] == 4:
            tables.append([row["Opponent 1"], row["Opponent 2"], row["Opponent 3"], row["Player Commander"], 3])
    elif row["Winner"] == 2 and row["Seat"] == 1:
        tables.append([row["Player Commander"], row["Opponent 1"], row["Opponent 2"], row["Opponent 3"], 4])

labels = ["deck1", "deck2", "deck3", "deck4", "outcome"]
df_singled_tables = pd.DataFrame(tables, columns=labels)


### Label ENCODING ###
#Initialize Label Encoder and fit on all commanders
encoder = LabelEncoder()
encoder.fit(valid_commanders) 

#Ensure "Unknown" is always included in the encoder
#if "Unknown" not in encoder.classes_:
#    encoder.classes_ = np.append(encoder.classes_, "Unknown")
    
#print("Number of decks:", len(encoder.classes_))
#print("Is 'Unknown' in encoder?", "Unknown" in encoder.classes_)

#Debug: Print unseen commanders before transforming
for col in ["deck1", "deck2", "deck3", "deck4"]:
    unseen = set(df_singled_tables[col].unique()) - set(encoder.classes_)
    if unseen:
        print(f"WARNING: Unseen commanders in {col} before transforming:", unseen)

#Replace unseen commanders with "Unknown" before transforming
for col in ["deck1", "deck2", "deck3", "deck4"]:
    df_singled_tables[col] = df_singled_tables[col].apply(lambda x: x if x in encoder.classes_ else "Unknown")

#Encode Commanders
for col in ["deck1", "deck2", "deck3", "deck4"]:
    df_singled_tables[f"{col}"] = encoder.transform(df_singled_tables[col])
    
df_encoded = df_singled_tables[["deck1", "deck2", "deck3", "deck4", "outcome"]] 

# Save the filtered dataset
df_encoded.to_csv("processed_tournament_data.csv", index=False)
print(f"Filtered dataset saved! Kept {len(df_filtered)} rows.")

#Save encoder to pickle for use in matchup predictor
with open("deck_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)