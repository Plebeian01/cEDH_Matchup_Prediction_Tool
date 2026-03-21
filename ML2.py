import json
import os
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
# Matchup Stat Functions
###########################################################
# Tracks A vs. B winrates across the dataset
def build_matchup_stats(table_df):
    deck_arr = table_df[["deck1", "deck2", "deck3", "deck4"]].values
    outcomes = table_df["outcome"].values

    # Precompute winning deck index per row (-1 for draws)
    winning_deck = np.full(len(outcomes), -1, dtype=deck_arr.dtype)
    is_win = (outcomes >= 0) & (outcomes <= 3)
    winning_deck[is_win] = deck_arr[np.arange(len(deck_arr))[is_win], outcomes[is_win]]

    wins = defaultdict(lambda: defaultdict(int))
    appearances = defaultdict(lambda: defaultdict(int))

    for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
        A_col = deck_arr[:, i]
        B_col = deck_arr[:, j]
        for a, b, w in zip(A_col, B_col, winning_deck):
            appearances[a][b] += 1
            appearances[b][a] += 1
            if w != -1:
                if a == w:
                    wins[a][b] += 1
                elif b == w:
                    wins[b][a] += 1

    matchup = defaultdict(lambda: defaultdict(lambda: 0.25))
    for A in appearances:
        for B in appearances[A]:
            if appearances[A][B] > 0:
                matchup[A][B] = wins[A][B] / appearances[A][B]
    return matchup


# Adds matchup features to dataset
def apply_matchup_features(table_df, matchup):
    d1 = table_df["deck1"].values
    d2 = table_df["deck2"].values
    d3 = table_df["deck3"].values
    d4 = table_df["deck4"].values

    df = table_df.copy()
    df["matchup_12"] = [matchup[a][b] for a, b in zip(d1, d2)]
    df["matchup_13"] = [matchup[a][b] for a, b in zip(d1, d3)]
    df["matchup_14"] = [matchup[a][b] for a, b in zip(d1, d4)]
    df["matchup_23"] = [matchup[a][b] for a, b in zip(d2, d3)]
    df["matchup_24"] = [matchup[a][b] for a, b in zip(d2, d4)]
    df["matchup_34"] = [matchup[a][b] for a, b in zip(d3, d4)]
    return df


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
    class_weights = [total / draw_counts[i] for i in [0, 1]]
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

    plt.figure(figsize=(8, 5))
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
    class_weights = [total / seat_counts[i] for i in [0, 1, 2, 3]]
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

    plt.figure(figsize=(8, 5))
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, "processed_tournament_data.csv"))
    matchup_stats = build_matchup_stats(df)
    df = apply_matchup_features(df, matchup_stats)

    # 2) Global train/test split
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # 3) Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_decks = df[["deck1", "deck2", "deck3", "deck4"]].values.max() + 1

    # 4) Train models using shared df_train
    draw_model = train_draw_model(df_train, num_decks, device)

    df_winloss = df_train[df_train["outcome"].isin([0, 1, 2, 3])].copy()
    seat_model = train_seat_model(df_winloss, num_decks, device)

    # 5) Save models
    torch.save(draw_model.state_dict(), os.path.join(script_dir, "best_draw_model.pth"))
    torch.save(seat_model.state_dict(), os.path.join(script_dir, "best_seat_model.pth"))

    # 6) Evaluate on df_test
    draw_model.eval()
    seat_model.eval()

    deck_arr = df_test[["deck1", "deck2", "deck3", "deck4"]].values
    synergy_arr = df_test[["matchup_12", "matchup_13", "matchup_14",
                            "matchup_23", "matchup_24", "matchup_34"]].values
    label_arr = df_test["outcome"].values

    # First pass: collect raw scores for both models
    raw_draw_probs = []
    raw_seat_preds = []

    for i in range(len(df_test)):
        decks = torch.tensor([deck_arr[i]], dtype=torch.long, device=device)
        synergy = torch.tensor([synergy_arr[i]], dtype=torch.float32, device=device)

        with torch.no_grad():
            draw_logits = draw_model(decks, synergy)
            raw_draw_probs.append(F.softmax(draw_logits, dim=1)[0, 1].item())

            seat_logits = seat_model(decks, synergy)
            raw_seat_preds.append(F.softmax(seat_logits, dim=1).argmax(dim=1).item())

    raw_draw_probs = np.array(raw_draw_probs)

    # Calibrate threshold: find the p_draw cutoff where predicted draw rate == actual draw rate
    actual_draw_rate = (label_arr == 4).mean()
    draw_threshold = float(np.percentile(raw_draw_probs, (1 - actual_draw_rate) * 100))
    print(f"Actual draw rate: {actual_draw_rate:.3f} → calibrated threshold: {draw_threshold:.4f}")

    # Save threshold so predictor511.py uses the same cutoff
    with open(os.path.join(script_dir, "draw_threshold.json"), "w") as f:
        json.dump({"draw_threshold": draw_threshold}, f)

    # Second pass: apply threshold
    final_preds = [4 if p >= draw_threshold else s
                   for p, s in zip(raw_draw_probs, raw_seat_preds)]
    final_labels = list(label_arr)

    # 7) Print evaluation
    print("Two-step final classification report (0-3 seats, 4=draw):")
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
