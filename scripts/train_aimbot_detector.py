# -*- coding: utf-8 -*-
"""
Whole‑match sequence classification for aimbot detection in AssaultCube.
Uses full player sequences (position events only), adds aim‑correction features,
and performs stratified match‑level cross‑validation.
"""

import ast
import json
import random
import math
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from collections import defaultdict
from utils import *
# -----------------------------------------------------------------------------
# Configuration – your paths and labels
# -----------------------------------------------------------------------------
DEMO_FILES = [
    # Clean matches
    ("demo20260306_2225_local_ac_desert_10min_DM.json", {0: False, 1: False, 2: False, 3: False, 4: False,}),
    ("demo20260306_2245_local_ac_desert3_10min_DM.json", {0: False, 1: False, 2: False, 3: False, 4: False, 5: False}),
    ("demo20260306_2316_local_ac_lainio_10min_DM.json", {0: False, 1: False, 2: False, 3: False, 4: False, 5: False}),
    ("demo20260306_2331_local_ac_scaffold_10min_DM.json", {0: False, 1: False, 2: False, 3: False, 4: False, 5: False}),
    ("demo20260306_2341_local_ac_scaffold_10min_DM.json", {0: False, 1: False, 2: False, 3: False, 4: False, 5: False}),


    ("demo20260428_1947_local_ac_desert3_10min_DM.json", {0: False, 1: False, 2: False}),
    ("demo20260428_2007_local_ac_desert_10min_DM.json", {0: False, 1: False, 2: False, 3: False}),
    ("demo20260428_2028_local_ac_elevation_10min_DM.json", {0: False, 1: False, 2: False, 3: False}),
    ("demo20260428_2038_local_ac_desert3_10min_DM.json", {0: False, 1: False, 2: False, 3: False}),
    #("demo20260428_2048_local_ac_scaffold_10min_DM.json", {0: False, 1: False, 2: False, 3: False}),
    # ("demo20260428_2058_local_ac_desert_10min_DM.json", {0: False, 1: False, 2: False, 3: False}),
    #---------------------------------
    # Full cheating matches
    ("demo20260322_2024_local_ac_scaffold_10min_DM.json", {0: True, 1: True, 2: True}),
    ("demo20260322_2042_local_ac_desert3_10min_DM.json", {0: True, 1: True, 2: True}),
    ("demo20260322_2052_local_ac_scaffold_10min_DM.json", {0: True, 1: True, 2: True}),
    ("demo20260322_2102_local_ac_desert_10min_DM.json", {0: True, 1: True, 2: True}),
    ("demo20260322_2114_local_ac_elevation_10min_DM.json", {0: True, 1: True, 2: True}),
    ("demo20260322_2126_local_ac_desert3_10min_DM.json", {0: True, 1: True, 2: True}),
    ("demo20260322_2136_local_ac_scaffold_10min_DM.json", {0: True, 1: True, 2: True}),
    ("demo20260322_2146_local_ac_desert_10min_DM.json", {0: True, 1: True, 2: True}),

    ("demo20260423_2125_local_ac_desert3_10min_DM.json", {0: True, 1: True, 2: True}),
    ("demo20260423_2146_local_ac_desert_10min_DM.json", {0: True, 1: True, 2: True}),
    ("demo20260423_2156_local_ac_lainio_10min_DM.json", {0: True, 1: True, 2: True}),
    ("demo20260423_2206_local_ac_elevation_10min_DM.json", {0: True, 1: True, 2: True}),

    ("demo20260424_1233_local_ac_desert3_10min_DM.json", {0: True, 1: True, 2: True}),
    ("demo20260424_1243_local_ac_scaffold_10min_DM.json", {0: True, 1: True, 2: True}),
    ("demo20260424_1253_local_ac_desert_10min_DM.json", {0: True, 1: True, 2: True}),
    ("demo20260424_1303_local_ac_lainio_10min_DM.json", {0: True, 1: True, 2: True}),
    #---------------------------------
    # Mixed matches (one cheater, rest clean)
    ("demo20260325_1513_local_ac_desert3_10min_DM.json", {0: True, 1: False, 2: False, 3: False}),
    ("demo20260325_1523_local_ac_scaffold_10min_DM.json", {0: False, 1: False, 2: False, 3: True}),
    ("demo20260325_1533_local_ac_desert_10min_DM.json", {0: False, 1: False, 2: True, 3: False}),
    ("demo20260325_1544_local_ac_lainio_10min_DM.json", {0: False, 1: False, 2: False, 3: True}),
]

# Features to use
FEATURE_NAMES = [
    'x', 'y', 'z', 'yaw', 'pitch',
    'dx', 'dy', 'dz', 'dyaw', 'dpitch', 'shooting',
    'min_angle_to_enemy',   # smallest angle to any other player
    'aim_correction_delta'  # angular difference to closest enemy
]

# Hyperparameters
BATCH_SIZE = 4              # small because whole sequences can be long
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
K_FOLDS = 5
RANDOM_SEED = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -----------------------------------------------------------------------------
# Data loading and feature engineering
# -----------------------------------------------------------------------------
def load_events(json_path):
    """Load file with Python dict literals (one per line) into a list of dicts."""
    events = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(ast.literal_eval(line))
    return events

def build_player_timelines(events):
    """
    Group only position events by client number; sort by gametime.
    """
    players = defaultdict(list)
    for ev in events:
        if ev.get('type') == 'position' and 'cn' in ev:
            players[ev['cn']].append(ev)
    for cn in players:
        players[cn].sort(key=lambda e: e.get('gametime', 0))
    return players

def add_derived_features(events):
    """Add dx, dy, dz, dyaw, dpitch. Modifies list in‑place."""
    prev = None
    for ev in events:
        if prev is None:
            ev['dx'] = ev['dy'] = ev['dz'] = ev['dyaw'] = ev['dpitch'] = 0.0
        else:
            ev['dx'] = ev['x'] - prev['x']
            ev['dy'] = ev['y'] - prev['y']
            ev['dz'] = ev['z'] - prev['z']
            dyaw = ev['yaw'] - prev['yaw']
            if dyaw > 180:
                dyaw -= 360
            elif dyaw < -180:
                dyaw += 360
            ev['dyaw'] = dyaw
            ev['dpitch'] = ev['pitch'] - prev['pitch']
        # ensure shooting exists (position events already have it)
        if 'shooting' not in ev:
            ev['shooting'] = 0
        prev = ev
    return events

def compute_aim_correction_features(player_events, all_players_events):
    """
    For each position event of the subject player, compute:
      - min_angle_to_enemy : smallest angle (deg) between current view direction
        and the vector to any other player (assumed enemy).
      - aim_correction_delta : angular difference to the closest enemy.
    Modifies player_events in‑place.
    """
    if not player_events:
        return player_events

    subject_cn = player_events[0]['cn']
    # Build lookup of positions for other players (only position events)
    other_players = {}
    for cn, evs in all_players_events.items():
        if cn == subject_cn:
            continue
        pos_evs = [e for e in evs if e.get('type') == 'position']
        if not pos_evs:
            continue
        times = np.array([e['gametime'] for e in pos_evs])
        coords = np.array([[e['x'], e['y'], e['z']] for e in pos_evs])
        other_players[cn] = (times, coords)

    if not other_players:
        # No other players? Fill with zeros.
        for ev in player_events:
            ev['min_angle_to_enemy'] = 0.0
            ev['aim_correction_delta'] = 0.0
        return player_events

    for ev in player_events:
        t = ev['gametime']
        my_pos = np.array([ev['x'], ev['y'], ev['z']])
        my_yaw = ev['yaw']
        my_pitch = ev['pitch']

        min_angle = 180.0
        aim_delta = 180.0

        for cn, (times, coords) in other_players.items():
            idx = np.searchsorted(times, t)
            if idx == 0:
                nearest_idx = 0
            elif idx == len(times):
                nearest_idx = len(times) - 1
            else:
                if abs(times[idx] - t) < abs(times[idx-1] - t):
                    nearest_idx = idx
                else:
                    nearest_idx = idx - 1

            other_pos = coords[nearest_idx]
            to_enemy = other_pos - my_pos
            dist = np.linalg.norm(to_enemy)
            if dist < 1e-6:
                continue

            req_yaw = math.degrees(math.atan2(to_enemy[1], to_enemy[0]))
            req_yaw = req_yaw % 360
            req_pitch = math.degrees(math.asin(np.clip(to_enemy[2]/dist, -1.0, 1.0)))

            yaw_diff = abs(req_yaw - my_yaw)
            if yaw_diff > 180:
                yaw_diff = 360 - yaw_diff
            pitch_diff = abs(req_pitch - my_pitch)

            total_angle = math.sqrt(yaw_diff**2 + pitch_diff**2)

            if total_angle < min_angle:
                min_angle = total_angle
                aim_delta = total_angle

        ev['min_angle_to_enemy'] = min_angle if min_angle != 180.0 else 0.0
        ev['aim_correction_delta'] = aim_delta if aim_delta != 180.0 else 0.0

    return player_events

def build_full_sequence(player_events, all_players_events, is_cheater):
    """
    Convert a player's position events into a 2D numpy array of features.
    Returns (sequence_array, label) or (None, None) if too short.
    """
    if len(player_events) < 10:
        return None, None

    player_events = add_derived_features(player_events)
    player_events = compute_aim_correction_features(player_events, all_players_events)

    seq = []
    for ev in player_events:
        vec = [float(ev.get(feat, 0.0)) for feat in FEATURE_NAMES]
        seq.append(vec)

    return np.array(seq, dtype=np.float32), 1 if is_cheater else 0

# -----------------------------------------------------------------------------
# Dataset and collate function
# -----------------------------------------------------------------------------
class CheatSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.float32)
        lbl = torch.tensor(self.labels[idx], dtype=torch.float32)
        return seq, lbl

def collate_fn(batch):
    seqs, lbls = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    seqs_pad = pad_sequence(seqs, batch_first=True)
    max_len = seqs_pad.size(1)
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    return seqs_pad, mask, torch.stack(lbls)

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class AimDetector(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask):
        batch_size, seq_len, _ = x.shape
        lengths = mask.sum(dim=1).cpu()
        x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(x_packed)
        last_hidden = hidden[-1]                     # shape: (batch, hidden_size)
        out = self.fc(last_hidden)                   # shape: (batch, 1)
        out = self.sigmoid(out).squeeze(-1)          # shape: (batch,)   (keeps batch dimension)
        return out

# -----------------------------------------------------------------------------
# Training function for one cross‑validation fold
# -----------------------------------------------------------------------------
def train_one_fold(train_loader, val_loader, model, optimizer, criterion, epochs, fold):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for x, mask, y in train_loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x, mask)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            pred_labels = (preds > 0.5).float()
            train_correct += (pred_labels == y).sum().item()
            train_total += y.size(0)
        train_loss /= train_total
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for x, mask, y in val_loader:
                x, mask, y = x.to(device), mask.to(device), y.to(device)
                preds = model(x, mask)
                loss = criterion(preds, y)
                val_loss += loss.item() * x.size(0)
                pred_labels = (preds > 0.5).float()
                val_correct += (pred_labels == y).sum().item()
                val_total += y.size(0)
                val_preds.extend(pred_labels.cpu().numpy())
                val_labels.extend(y.cpu().numpy())
        val_loss /= val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if (epoch + 1) % 5 == 0:
            print(f"Fold {fold+1} | Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    # Compute final metrics
    val_acc = val_accs[-1] if val_accs else 0.0
    precision = precision_score(val_labels, val_preds, zero_division=0)
    recall = recall_score(val_labels, val_preds, zero_division=0)
    f1 = f1_score(val_labels, val_preds, zero_division=0)
    return val_acc, precision, recall, f1, train_losses, val_losses, train_accs, val_accs

# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    print("Loading and processing demos...")
    all_sequences = []
    all_labels = []
    match_ids = []

    for match_idx, (json_path, cheat_dict) in enumerate(DEMO_FILES):
        if not os.path.exists(json_path):
            print(f"Warning: file not found: {json_path}")
            continue
        events = load_events(json_path)
        players_events = build_player_timelines(events)
        # For each player with a label
        for cn, is_cheater in cheat_dict.items():
            if cn not in players_events:
                print(f"Warning: cn {cn} not in demo {json_path}")
                continue
            seq, lbl = build_full_sequence(players_events[cn], players_events, is_cheater)
            if seq is not None:
                all_sequences.append(seq)
                all_labels.append(lbl)
                match_ids.append(match_idx)

    if len(all_sequences) == 0:
        print("No valid sequences found.")
        return

    print(f"Total sequences: {len(all_sequences)}")
    print(f"Cheater sequences: {sum(all_labels)}, Clean: {len(all_labels) - sum(all_labels)}")

    # Normalize features across all timesteps
    all_timesteps = np.vstack(all_sequences)
    mean = all_timesteps.mean(axis=0, keepdims=True)
    std = all_timesteps.std(axis=0, keepdims=True) + 1e-8
    norm_sequences = [(seq - mean) / std for seq in all_sequences]

    # Prepare cross‑validation at match level
    unique_matches = sorted(set(match_ids))
    match_cheat_label = []
    for mid in unique_matches:
        lbls = [all_labels[i] for i, m in enumerate(match_ids) if m == mid]
        match_cheat_label.append(1 if any(l == 1 for l in lbls) else 0)

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_results = []

    for fold, (train_match_idx, val_match_idx) in enumerate(skf.split(unique_matches, match_cheat_label)):
        print(f"\n{'='*50}\nFold {fold+1}/{K_FOLDS}\n{'='*50}")
        train_matches = set(unique_matches[i] for i in train_match_idx)
        val_matches   = set(unique_matches[i] for i in val_match_idx)

        train_idx = [i for i, m in enumerate(match_ids) if m in train_matches]
        val_idx   = [i for i, m in enumerate(match_ids) if m in val_matches]

        X_train = [norm_sequences[i] for i in train_idx]
        y_train = [all_labels[i] for i in train_idx]
        X_val   = [norm_sequences[i] for i in val_idx]
        y_val   = [all_labels[i] for i in val_idx]

        print(f"Train: {len(X_train)} seqs, Val: {len(X_val)} seqs")
        print(f"Train cheat ratio: {sum(y_train)/len(y_train):.2f}, Val cheat ratio: {sum(y_val)/len(y_val):.2f}")

        train_dataset = CheatSequenceDataset(X_train, y_train)
        val_dataset   = CheatSequenceDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        model = AimDetector(input_size=len(FEATURE_NAMES)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCELoss()

        val_acc, prec, rec, f1, tloss, vloss, tacc, vacc = train_one_fold(
            train_loader, val_loader, model, optimizer, criterion, NUM_EPOCHS, fold
        )

        fold_results.append({
            'fold': fold+1,
            'val_acc': val_acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        })

    # Summary
    print("\n" + "="*50)
    print("Cross‑Validation Summary")
    print("="*50)
    accs = [r['val_acc'] for r in fold_results]
    precs = [r['precision'] for r in fold_results]
    recs = [r['recall'] for r in fold_results]
    f1s = [r['f1'] for r in fold_results]

    print(f"Average Accuracy : {np.mean(accs):.4f} (+/- {np.std(accs):.4f})")
    print(f"Average Precision: {np.mean(precs):.4f} (+/- {np.std(precs):.4f})")
    print(f"Average Recall   : {np.mean(recs):.4f} (+/- {np.std(recs):.4f})")
    print(f"Average F1-score : {np.mean(f1s):.4f} (+/- {np.std(f1s):.4f})")
    for r in fold_results:
        print(f"Fold {r['fold']}: Acc={r['val_acc']:.4f}, Prec={r['precision']:.4f}, Rec={r['recall']:.4f}, F1={r['f1']:.4f}")

if __name__ == "__main__":
    main()