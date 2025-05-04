# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 19:28:34 2025

@author: freez
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from torch.optim.lr_scheduler import ReduceLROnPlateau

###########################################################
#Matchup Stat Functions
###########################################################
#Tracks A vs. B winrates across the dataset
def build_matchup_stats(table_df):
    wins = defaultdict(lambda: defaultdict(int))
    appearances = defaultdict(lambda: defaultdict(int))

    for _, row in table_df.iterrows():
        decks = [row["deck1"], row["deck2"], row["deck3"], row["deck4"]]
        outcome = row["outcome"]
        if 0 <= outcome <= 3:  # one seat is winner
            winning_deck = decks[outcome]
        else:
            winning_deck = None  # draw

        for i in range(4):
            for j in range(i + 1, 4):
                A = decks[i]
                B = decks[j]
                # record appearance
                appearances[A][B] += 1
                appearances[B][A] += 1

                # increment A->B or B->A if there's a winner
                if winning_deck is not None:
                    if A == winning_deck and B != winning_deck:
                        wins[A][B] += 1
                    elif B == winning_deck and A != winning_deck:
                        wins[B][A] += 1

    # compute fraction
    matchup = defaultdict(lambda: defaultdict(float))
    for A in appearances:
        for B in appearances[A]:
            if appearances[A][B] > 0:
                matchup[A][B] = wins[A][B] / appearances[A][B]
            else:
                matchup[A][B] = 0.0
    return matchup

#Adds matchup features to dataset
def apply_matchup_features(table_df, matchup):
    new_cols = {
        "matchup_12": [], "matchup_13": [], "matchup_14": [],
        "matchup_23": [], "matchup_24": [], "matchup_34": []
    }
    for _, row in table_df.iterrows():
        d1, d2, d3, d4 = row["deck1"], row["deck2"], row["deck3"], row["deck4"]
        new_cols["matchup_12"].append(matchup[d1][d2])
        new_cols["matchup_13"].append(matchup[d1][d3])
        new_cols["matchup_14"].append(matchup[d1][d4])
        new_cols["matchup_23"].append(matchup[d2][d3])
        new_cols["matchup_24"].append(matchup[d2][d4])
        new_cols["matchup_34"].append(matchup[d3][d4])

    matchup_df = pd.DataFrame(new_cols)
    return pd.concat([table_df.reset_index(drop=True), matchup_df], axis=1)

###########################################################
# Datasets
###########################################################
class DrawVsNotDrawDataset(Dataset):
    def __init__(self, df):
        self.deck_data = df[["deck1","deck2","deck3","deck4"]].values.astype(np.int64)
        self.matchup_data = df[["matchup_12","matchup_13","matchup_14",
                                "matchup_23","matchup_24","matchup_34"]].values.astype(np.float32)
        self.labels = (df["outcome"] == 4).astype(np.int64).values  # binary label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.deck_data[idx], self.matchup_data[idx], self.labels[idx]


class WinLossDataset(Dataset):
    """
    Only rows with outcome in [0..3], i.e. seat0..3
    """
    def __init__(self, df):
        self.deck_data = df[["deck1","deck2","deck3","deck4"]].values.astype(np.int64)
        self.matchup_data = df[["matchup_12","matchup_13","matchup_14",
                                "matchup_23","matchup_24","matchup_34"]].values.astype(np.float32)
        self.labels = df["outcome"].values.astype(np.int64)  # seat0..3

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.deck_data[idx], self.matchup_data[idx], self.labels[idx]

###########################################################
# Models
###########################################################
class DrawModel(nn.Module):
    """
    Binary classifier: 0=not-draw, 1=draw
    """
    def __init__(self, num_decks, embedding_dim=32, num_heads=4):
        super().__init__()
        self.deck_embedding = nn.Embedding(num_decks, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim*4 + 6, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # 2-class: not-draw, draw
        )

    def forward(self, deck_batch, synergy_batch):
        emb = self.deck_embedding(deck_batch)
        emb = self.dropout(emb)
        attn_out, _ = self.attention(emb, emb, emb)
        attn_flat = attn_out.reshape(attn_out.size(0), -1)
        x = torch.cat([attn_flat, synergy_batch], dim=1)
        return self.fc(x)


class WinLossModel(nn.Module):
    """
    4-class seat predictor: seat0..3
    """
    def __init__(self, num_decks, embedding_dim=32, num_heads=8):
        super().__init__()
        self.deck_embedding = nn.Embedding(num_decks, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim*4 + 6, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 4)  # seat0..3
        )

    def forward(self, deck_batch, synergy_batch):
        emb = self.deck_embedding(deck_batch)
        emb = self.dropout(emb)
        attn_out, _ = self.attention(emb, emb, emb)
        attn_flat = attn_out.reshape(attn_out.size(0), -1)
        x = torch.cat([attn_flat, synergy_batch], dim=1)
        return self.fc(x)

###########################################################
# Train & Evaluate
###########################################################
def train_draw_model(df_all, num_decks, device):
    draw_dataset = DrawVsNotDrawDataset(df_all)
    train_size = int(0.8 * len(draw_dataset))
    val_size = len(draw_dataset) - train_size
    train_ds, val_ds = random_split(draw_dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    draw_model = DrawModel(num_decks=num_decks).to(device)

    draw_counts = Counter((df_all["outcome"] == 4).astype(int))
    total = sum(draw_counts.values())
    class_weights = [ total / draw_counts[i] for i in [0,1] ]
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(draw_model.parameters(), lr=0.0005)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    best_state = None
    patience = 2
    patience_counter = 0

    draw_train_losses = []
    draw_val_losses = []

    for epoch in range(50):
        draw_model.train()
        total_loss = 0
        for deck_batch, synergy_batch, labels in train_loader:
            deck_batch, synergy_batch, labels = deck_batch.to(device), synergy_batch.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = draw_model(deck_batch, synergy_batch)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        draw_model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for deck_batch, synergy_batch, labels in val_loader:
                deck_batch, synergy_batch, labels = deck_batch.to(device), synergy_batch.to(device), labels.to(device)
                logits = draw_model(deck_batch, synergy_batch)
                batch_loss = criterion(logits, labels).item()
                val_loss += batch_loss
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        draw_train_losses.append(avg_train_loss)
        draw_val_losses.append(avg_val_loss)
        print(f"[DrawModel] Epoch {epoch+1}: TrainLoss={avg_train_loss:.4f}, ValLoss={avg_val_loss:.4f}, ValAcc={val_acc:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = draw_model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    if best_state is not None:
        draw_model.load_state_dict(best_state)

    plt.figure(figsize=(8,5))
    plt.plot(draw_train_losses, label='DrawModel Train Loss')
    plt.plot(draw_val_losses, label='DrawModel Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DrawModel Training vs Validation Loss')
    plt.legend()
    plt.show()

    return draw_model


def train_seat_model(df_all, num_decks, device):
    seat_dataset = WinLossDataset(df_all)
    train_size = int(0.8 * len(seat_dataset))
    val_size = len(seat_dataset) - train_size
    train_ds, val_ds = random_split(seat_dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    seat_model = WinLossModel(num_decks=num_decks).to(device)

    seat_counts = Counter(df_all["outcome"])
    total = sum(seat_counts.values())
    class_weights = [total / seat_counts[i] for i in [0,1,2,3]]
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(seat_model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    best_state = None
    patience = 4
    patience_counter = 0

    seat_train_losses = []
    seat_val_losses = []

    for epoch in range(50):
        seat_model.train()
        total_loss = 0
        for deck_batch, synergy_batch, labels in train_loader:
            deck_batch, synergy_batch, labels = deck_batch.to(device), synergy_batch.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = seat_model(deck_batch, synergy_batch)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        seat_model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for deck_batch, synergy_batch, labels in val_loader:
                deck_batch, synergy_batch, labels = deck_batch.to(device), synergy_batch.to(device), labels.to(device)
                logits = seat_model(deck_batch, synergy_batch)
                batch_loss = criterion(logits, labels).item()
                val_loss += batch_loss
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        seat_train_losses.append(avg_train_loss)
        seat_val_losses.append(avg_val_loss)
        print(f"[SeatModel] Epoch {epoch+1}: TrainLoss={avg_train_loss:.4f}, ValLoss={avg_val_loss:.4f}, ValAcc={val_acc:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = seat_model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    if best_state is not None:
        seat_model.load_state_dict(best_state)

    plt.figure(figsize=(8,5))
    plt.plot(seat_train_losses, label='SeatModel Train Loss')
    plt.plot(seat_val_losses, label='SeatModel Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('SeatModel Training vs Validation Loss')
    plt.legend()
    plt.show()

    return seat_model


###########################################################
# Main logic
###########################################################
def main():
    # 1) Load data and generate synergy features
    df = pd.read_csv("processed_tournament_data.csv")
    matchup_stats = build_matchup_stats(df)
    df = apply_matchup_features(df, matchup_stats)

    # 2) Global train/test split
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # 3) Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_decks = df[["deck1", "deck2", "deck3", "deck4"]].max().max() + 1

    # 4) Train models using shared df_train
    draw_model = train_draw_model(df_train, num_decks, device)
    
    df_winloss = df_train[df_train["outcome"].isin([0,1,2,3])].copy()
    seat_model = train_seat_model(df_winloss, num_decks, device)

    # 5) Save models
    torch.save(draw_model.state_dict(), "best_draw_model.pth")
    torch.save(seat_model.state_dict(), "best_seat_model.pth")

    # 6) Evaluate on df_test
    draw_model.eval()
    seat_model.eval()

    final_preds = []
    final_labels = []

    for _, row in df_test.iterrows():
        decks = torch.tensor([[row["deck1"], row["deck2"], row["deck3"], row["deck4"]]], dtype=torch.long, device=device)
        synergy = torch.tensor([[row["matchup_12"], row["matchup_13"], row["matchup_14"],
                                 row["matchup_23"], row["matchup_24"], row["matchup_34"]]], dtype=torch.float32, device=device)

        with torch.no_grad():
            draw_logits = draw_model(decks, synergy)
            draw_probs = F.softmax(draw_logits, dim=1)
            p_draw = draw_probs[0, 1].item()
            draw_pred = 1 if p_draw > 0.575 else 0

        if draw_pred == 1:
            final_preds.append(4)
        else:
            with torch.no_grad():
                seat_logits = seat_model(decks, synergy)
                seat_probs = F.softmax(seat_logits, dim=1)
                seat_pred = seat_probs.argmax(dim=1).item()
                final_preds.append(seat_pred)

        final_labels.append(row["outcome"])

    # 7) Print evaluation
    print("Two-step final classification report (0–3 seats, 4=draw):")
    print(classification_report(final_labels, final_preds, labels=[0, 1, 2, 3, 4], digits=3))

    cm = confusion_matrix(final_labels, final_preds, labels=[0, 1, 2, 3, 4])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Seat1", "Seat2", "Seat3", "Seat4", "Draw"],
                yticklabels=["Seat1", "Seat2", "Seat3", "Seat4", "Draw"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Draw vs. Win Prediction)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
  
