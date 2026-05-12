# -*- coding: utf-8 -*-
"""
Unified model comparison for aimbot detection.
Trains and evaluates:
  - Classical: Logistic Regression, Random Forest, XGBoost, Simple Average
  - Deep: 1D-CNN, LSTM (Dunham-style)
On the same match-level 5-fold cross-validation.
Saves final models and scaler for later inference.
"""

import ast
import json
import random
import math
import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from collections import defaultdict
import joblib
import xgboost as xgb

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Configuration – DEMO_FILES and FEATURE_NAMES
# -----------------------------------------------------------------------------
DEMO_FILES = [
    # Clean matches
    ("demo20260306_2225_local_ac_desert_10min_DM.json", {0: False, 1: False, 2: False, 3: False, 4: False}),
    ("demo20260306_2245_local_ac_desert3_10min_DM.json", {0: False, 1: False, 2: False, 3: False, 4: False, 5: False}),
    ("demo20260306_2316_local_ac_lainio_10min_DM.json", {0: False, 1: False, 2: False, 3: False, 4: False, 5: False}),
    ("demo20260306_2331_local_ac_scaffold_10min_DM.json", {0: False, 1: False, 2: False, 3: False, 4: False, 5: False}),
    ("demo20260306_2341_local_ac_scaffold_10min_DM.json", {0: False, 1: False, 2: False, 3: False, 4: False, 5: False}),
    ("demo20260428_1947_local_ac_desert3_10min_DM.json", {0: False, 1: False, 2: False}),
    ("demo20260428_2007_local_ac_desert_10min_DM.json", {0: False, 1: False, 2: False, 3: False}),
    ("demo20260428_2028_local_ac_elevation_10min_DM.json", {0: False, 1: False, 2: False, 3: False}),
    ("demo20260428_2038_local_ac_desert3_10min_DM.json", {0: False, 1: False, 2: False, 3: False}),
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
    # Mixed matches (one cheater, rest clean)
    ("demo20260325_1513_local_ac_desert3_10min_DM.json", {0: True, 1: False, 2: False, 3: False}),
    ("demo20260325_1523_local_ac_scaffold_10min_DM.json", {0: False, 1: False, 2: False, 3: True}),
    ("demo20260325_1533_local_ac_desert_10min_DM.json", {0: False, 1: False, 2: True, 3: False}),
    ("demo20260325_1544_local_ac_lainio_10min_DM.json", {0: False, 1: False, 2: False, 3: True}),
]

FEATURE_NAMES = [
    # Removed x, y, z to avoid map bias – keep their deltas (dx, dy, dz)
    'yaw', 'pitch',
    'dx', 'dy', 'dz',
    'dyaw', 'dpitch',
    'shooting',
    'min_angle_to_enemy',   # smallest angle to any other player
    'aim_correction_delta'  # angular difference to closest enemy
]


RANDOM_SEED = 42
K_FOLDS = 5
FIXED_LEN = 2000               # for 1D-CNN and LSTM (LSTM will still use variable length via packing)
BATCH_SIZE = 4                 # small batch for deep models
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 0.001
NUM_EPOCHS_DEEP = 30           # epochs for LSTM and CNN
NUM_EPOCHS_CNN = 20            # CNN can use fewer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------------------------------------------------------
# Helper functions (shared with utils.py)
# -----------------------------------------------------------------------------
def load_events(json_path):
    events = []
    for enc in ('utf-8', 'utf-16', 'utf-16-le', 'latin-1'):
        try:
            with open(json_path, 'r', encoding=enc) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    events.append(ast.literal_eval(line))
            return events
        except (UnicodeDecodeError, ValueError, SyntaxError):
            events = []
    raise ValueError(f"Could not decode {json_path}")

def build_player_timelines(events):
    players = defaultdict(list)
    for ev in events:
        if ev.get('type') == 'position' and 'cn' in ev:
            players[ev['cn']].append(ev)
    for cn in players:
        players[cn].sort(key=lambda e: e.get('gametime', 0))
    return players

def add_derived_features(events):
    prev = None
    for ev in events:
        if prev is None:
            ev['dx'] = ev['dy'] = ev['dz'] = ev['dyaw'] = ev['dpitch'] = 0.0
        else:
            ev['dx'] = ev['x'] - prev['x']
            ev['dy'] = ev['y'] - prev['y']
            ev['dz'] = ev['z'] - prev['z']
            dyaw = ev['yaw'] - prev['yaw']
            if dyaw > 180: dyaw -= 360
            elif dyaw < -180: dyaw += 360
            ev['dyaw'] = dyaw
            ev['dpitch'] = ev['pitch'] - prev['pitch']
        if 'shooting' not in ev: ev['shooting'] = 0
        prev = ev
    return events

def compute_aim_correction_features(player_events, all_players_events):
    if not player_events: return player_events
    subject_cn = player_events[0]['cn']
    other_players = {}
    for cn, evs in all_players_events.items():
        if cn == subject_cn: continue
        pos_evs = [e for e in evs if e.get('type') == 'position']
        if not pos_evs: continue
        times = np.array([e['gametime'] for e in pos_evs])
        coords = np.array([[e['x'], e['y'], e['z']] for e in pos_evs])
        other_players[cn] = (times, coords)
    if not other_players:
        for ev in player_events:
            ev['min_angle_to_enemy'] = 0.0
            ev['aim_correction_delta'] = 0.0
        return player_events
    for ev in player_events:
        t = ev['gametime']
        my_pos = np.array([ev['x'], ev['y'], ev['z']])
        my_yaw, my_pitch = ev['yaw'], ev['pitch']
        min_angle = 180.0
        aim_delta = 180.0
        for cn, (times, coords) in other_players.items():
            idx = np.searchsorted(times, t)
            if idx == 0: nearest_idx = 0
            elif idx == len(times): nearest_idx = len(times)-1
            else:
                nearest_idx = idx-1 if abs(times[idx-1]-t) < abs(times[idx]-t) else idx
            other_pos = coords[nearest_idx]
            to_enemy = other_pos - my_pos
            dist = np.linalg.norm(to_enemy)
            if dist < 1e-6: continue
            req_yaw = math.degrees(math.atan2(to_enemy[1], to_enemy[0])) % 360
            req_pitch = math.degrees(math.asin(np.clip(to_enemy[2]/dist, -1.0, 1.0)))
            yaw_diff = abs(req_yaw - my_yaw)
            if yaw_diff > 180: yaw_diff = 360 - yaw_diff
            pitch_diff = abs(req_pitch - my_pitch)
            total_angle = math.sqrt(yaw_diff**2 + pitch_diff**2)
            if total_angle < min_angle:
                min_angle = total_angle
                aim_delta = total_angle
        ev['min_angle_to_enemy'] = min_angle if min_angle != 180.0 else 0.0
        ev['aim_correction_delta'] = aim_delta if aim_delta != 180.0 else 0.0
    return player_events

def build_full_sequence(player_events, all_players_events, is_cheater):
    if len(player_events) < 10: return None, None
    player_events = add_derived_features(player_events)
    player_events = compute_aim_correction_features(player_events, all_players_events)
    seq = []
    for ev in player_events:
        seq.append([float(ev.get(f, 0.0)) for f in FEATURE_NAMES])
    return np.array(seq, dtype=np.float32), 1 if is_cheater else 0

def extract_statistical_features(seq):
    """
    seq: np.array of shape (T, F) – variable length, F = len(FEATURE_NAMES).
    Returns a fixed-length feature vector (1D array).
    """
    # Basic statistics per feature
    mean = np.mean(seq, axis=0)
    std = np.std(seq, axis=0)
    min_val = np.min(seq, axis=0)
    max_val = np.max(seq, axis=0)
    median = np.median(seq, axis=0)
    p5 = np.percentile(seq, 5, axis=0)
    p95 = np.percentile(seq, 95, axis=0)

    # Temporal features (autocorrelation of dyaw, dpitch)
    if seq.shape[0] >= 2:
        # Indices in new FEATURE_NAMES:
        # dyaw = index 5, dpitch = index 6
        dyaw_series = seq[1:, 5]   # shift by 1
        dyaw_lag = seq[:-1, 5]
        ac_dyaw = np.corrcoef(dyaw_lag, dyaw_series)[0, 1] if np.std(dyaw_lag) > 0 else 0.0

        dpitch_series = seq[1:, 6]
        dpitch_lag = seq[:-1, 6]
        ac_dpitch = np.corrcoef(dpitch_lag, dpitch_series)[0, 1] if np.std(dpitch_lag) > 0 else 0.0
    else:
        ac_dyaw = ac_dpitch = 0.0

    # Shooting behaviour – index 7
    shooting = seq[:, 7]
    frac_shooting = np.mean(shooting)
    shooting_events = np.sum(np.diff(shooting) > 0.5)

    # Average speed from dx,dy,dz – indices 2,3,4
    total_dist = np.sum(np.sqrt(seq[:, 2]**2 + seq[:, 3]**2 + seq[:, 4]**2))
    duration = len(seq) * 0.04   # approx 40ms per tick
    avg_speed = total_dist / duration if duration > 0 else 0.0

    # Aim‑correction delta – index 9
    max_aim_corr = np.max(seq[:, 9])
    aim_on_target = np.mean(seq[:, 9] < 10.0)

    # Concatenate all features
    features = np.concatenate([
        mean, std, min_val, max_val, median, p5, p95,
        [ac_dyaw, ac_dpitch, frac_shooting, shooting_events,
         avg_speed, max_aim_corr, aim_on_target]
    ])
    return features

# -----------------------------------------------------------------------------
# 1D-CNN Model
# -----------------------------------------------------------------------------
class Simple1DCNN(nn.Module):
    def __init__(self, input_channels, seq_len):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, 5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, 5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * (seq_len // 4), 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.sigmoid(self.fc2(x)).squeeze(-1)

# -----------------------------------------------------------------------------
# LSTM Model (Dunham style)
# -----------------------------------------------------------------------------
class AimDetectorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers>1 else 0, bidirectional=False)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, mask):
        lengths = mask.sum(dim=1).cpu()
        x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(x_packed)
        return self.sigmoid(self.fc(hidden[-1])).squeeze(-1)

# Dataset for LSTM (variable-length)
class LSTMSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

def lstm_collate_fn(batch):
    seqs, lbls = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    padded = pad_sequence(seqs, batch_first=True)
    max_len = padded.size(1)
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    return padded, mask, torch.stack(lbls)

# -----------------------------------------------------------------------------
# Training helper for LSTM/CNN (can be reused)
# -----------------------------------------------------------------------------
def train_deep_model(train_loader, val_loader, model, optimizer, criterion, epochs, fold, model_name):
    best_val_f1 = -1.0
    best_state = model.state_dict().copy()
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0; train_correct = 0; train_total = 0
        for batch in train_loader:
            if model_name == 'LSTM':
                x, mask, y = batch
                x, mask, y = x.to(device), mask.to(device), y.to(device)
                preds = model(x, mask)
            else:  # CNN
                x, y = batch
                x, y = x.to(device), y.to(device)
                preds = model(x)
            loss = criterion(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            pred_labels = (preds > 0.5).float()
            train_correct += (pred_labels == y).sum().item()
            train_total += y.size(0)
        # Validation
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                if model_name == 'LSTM':
                    x, mask, y = batch
                    x, mask, y = x.to(device), mask.to(device), y.to(device)
                    preds = model(x, mask)
                else:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    preds = model(x)
                pred_labels = (preds > 0.5).int().cpu().numpy().tolist()
                val_preds.extend(pred_labels)
                val_true.extend(y.cpu().numpy().tolist())
        val_f1 = f1_score(val_true, val_preds, zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
        if (epoch+1) % 5 == 0:
            print(f"Fold {fold} | {model_name} Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss/train_total:.4f} Acc: {train_correct/train_total:.4f} | "
                  f"Val F1: {val_f1:.4f}")
    model.load_state_dict(best_state)
    model.eval()
    final_preds, final_true = [], []
    with torch.no_grad():
        for batch in val_loader:
            if model_name == 'LSTM':
                x, mask, y = batch
                x, mask, y = x.to(device), mask.to(device), y.to(device)
                preds = model(x, mask)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                preds = model(x)
            final_preds.extend((preds > 0.5).int().cpu().numpy().tolist())
            final_true.extend(y.cpu().numpy().tolist())
    return (accuracy_score(final_true, final_preds),
            precision_score(final_true, final_preds, zero_division=0),
            recall_score(final_true, final_preds, zero_division=0),
            f1_score(final_true, final_preds, zero_division=0))

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(RANDOM_SEED)

    print("Loading and processing demos...")
    all_sequences = []; all_labels = []; match_ids = []
    for match_idx, (json_path, cheat_dict) in enumerate(DEMO_FILES):
        if not os.path.exists(json_path):
            print(f"Warning: file not found: {json_path}")
            continue
        events = load_events(json_path)
        players_events = build_player_timelines(events)
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
        print("No valid sequences found."); return

    print(f"Total sequences: {len(all_sequences)}")
    print(f"Cheater: {sum(all_labels)}, Clean: {len(all_labels)-sum(all_labels)}")

    # Statistical features for classical + CNN input
    X_stat = np.array([extract_statistical_features(s) for s in all_sequences])
    y = np.array(all_labels)
    print(f"Statistical feature vector shape: {X_stat.shape}")

    # Raw sequences for LSTM (normalise per sequence later)
    X_raw = all_sequences  # list of numpy arrays

    # Prepare fixed-length sequences for CNN
    print("Preparing fixed-length sequences for CNN...")
    X_cnn = np.zeros((len(all_sequences), len(FEATURE_NAMES), FIXED_LEN), dtype=np.float32)
    for i, seq in enumerate(all_sequences):
        T = seq.shape[0]
        if T <= FIXED_LEN:
            X_cnn[i, :, :T] = seq.T
        else:
            X_cnn[i, :, :] = seq[-FIXED_LEN:, :].T

    # Match-level CV splits
    unique_matches = sorted(set(match_ids))
    match_cheat_label = []
    for mid in unique_matches:
        lbls = [all_labels[i] for i, m in enumerate(match_ids) if m == mid]
        match_cheat_label.append(1 if any(l == 1 for l in lbls) else 0)
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    models_results = {
        'Logistic Regression': {'acc':[], 'prec':[], 'rec':[], 'f1':[], 'time':[]},
        'Random Forest':       {'acc':[], 'prec':[], 'rec':[], 'f1':[], 'time':[]},
        'XGBoost':             {'acc':[], 'prec':[], 'rec':[], 'f1':[], 'time':[]},
        '1D-CNN':              {'acc':[], 'prec':[], 'rec':[], 'f1':[], 'time':[]},
        'LSTM':                {'acc':[], 'prec':[], 'rec':[], 'f1':[], 'time':[]},
        'Simple Average':      {'acc':[], 'prec':[], 'rec':[], 'f1':[], 'time':[]}
    }
    rf_importances = []; xgb_importances = []

    fold_num = 0
    for train_match_idx, val_match_idx in skf.split(unique_matches, match_cheat_label):
        fold_num += 1
        print(f"\n{'='*50}\nFold {fold_num}/{K_FOLDS}\n{'='*50}")
        train_m = set(unique_matches[i] for i in train_match_idx)
        val_m   = set(unique_matches[i] for i in val_match_idx)

        train_idx = [i for i, m in enumerate(match_ids) if m in train_m]
        val_idx   = [i for i, m in enumerate(match_ids) if m in val_m]

        y_train = y[train_idx]; y_val = y[val_idx]

        # --- Classical models (statistical features) ---
        X_train_stat = X_stat[train_idx]; X_val_stat = X_stat[val_idx]
        scaler = StandardScaler().fit(X_train_stat)
        X_train_stat_s = scaler.transform(X_train_stat)
        X_val_stat_s   = scaler.transform(X_val_stat)

        # Logistic Regression
        start = time.time()
        lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED).fit(X_train_stat_s, y_train)
        lr_proba = lr.predict_proba(X_val_stat_s)[:,1]
        lr_preds = (lr_proba >= 0.5).astype(int)
        models_results['Logistic Regression']['acc'].append(accuracy_score(y_val, lr_preds))
        models_results['Logistic Regression']['prec'].append(precision_score(y_val, lr_preds, zero_division=0))
        models_results['Logistic Regression']['rec'].append(recall_score(y_val, lr_preds, zero_division=0))
        models_results['Logistic Regression']['f1'].append(f1_score(y_val, lr_preds, zero_division=0))
        models_results['Logistic Regression']['time'].append(time.time() - start)

        # Random Forest
        start = time.time()
        rf = RandomForestClassifier(random_state=RANDOM_SEED)
        param_grid = {'n_estimators': [50,100], 'max_depth': [3,5,None]}
        best_rf = GridSearchCV(rf, param_grid, cv=3, scoring='f1').fit(X_train_stat_s, y_train).best_estimator_
        rf_proba = best_rf.predict_proba(X_val_stat_s)[:,1]
        rf_preds = (rf_proba >= 0.5).astype(int)
        models_results['Random Forest']['acc'].append(accuracy_score(y_val, rf_preds))
        models_results['Random Forest']['prec'].append(precision_score(y_val, rf_preds, zero_division=0))
        models_results['Random Forest']['rec'].append(recall_score(y_val, rf_preds, zero_division=0))
        models_results['Random Forest']['f1'].append(f1_score(y_val, rf_preds, zero_division=0))
        models_results['Random Forest']['time'].append(time.time() - start)
        rf_importances.append(best_rf.feature_importances_)

        # XGBoost
        start = time.time()
        xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=RANDOM_SEED)
        param_grid_xgb = {'n_estimators': [50,100], 'max_depth': [3,5]}
        best_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=3, scoring='f1').fit(X_train_stat_s, y_train).best_estimator_
        xgb_proba = best_xgb.predict_proba(X_val_stat_s)[:,1]
        xgb_preds = (xgb_proba >= 0.5).astype(int)
        models_results['XGBoost']['acc'].append(accuracy_score(y_val, xgb_preds))
        models_results['XGBoost']['prec'].append(precision_score(y_val, xgb_preds, zero_division=0))
        models_results['XGBoost']['rec'].append(recall_score(y_val, xgb_preds, zero_division=0))
        models_results['XGBoost']['f1'].append(f1_score(y_val, xgb_preds, zero_division=0))
        models_results['XGBoost']['time'].append(time.time() - start)
        xgb_importances.append(best_xgb.feature_importances_)

        # Simple Average
        avg_proba = (lr_proba + rf_proba + xgb_proba) / 3.0
        avg_preds = (avg_proba >= 0.5).astype(int)
        models_results['Simple Average']['acc'].append(accuracy_score(y_val, avg_preds))
        models_results['Simple Average']['prec'].append(precision_score(y_val, avg_preds, zero_division=0))
        models_results['Simple Average']['rec'].append(recall_score(y_val, avg_preds, zero_division=0))
        models_results['Simple Average']['f1'].append(f1_score(y_val, avg_preds, zero_division=0))
        models_results['Simple Average']['time'].append(0.0)

        # --- 1D-CNN ---
        X_train_cnn = X_cnn[train_idx]; X_val_cnn = X_cnn[val_idx]
        train_cnn_set = TensorDataset(torch.tensor(X_train_cnn, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        val_cnn_set   = TensorDataset(torch.tensor(X_val_cnn, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
        cnn_train_loader = DataLoader(train_cnn_set, batch_size=BATCH_SIZE, shuffle=True)
        cnn_val_loader   = DataLoader(val_cnn_set, batch_size=BATCH_SIZE, shuffle=False)

        cnn_model = Simple1DCNN(len(FEATURE_NAMES), FIXED_LEN).to(device)
        cnn_optim = torch.optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        cnn_crit = nn.BCELoss()
        cnn_acc, cnn_prec, cnn_rec, cnn_f1 = train_deep_model(
            cnn_train_loader, cnn_val_loader, cnn_model, cnn_optim, cnn_crit, NUM_EPOCHS_CNN, fold_num, model_name='CNN')
        models_results['1D-CNN']['acc'].append(cnn_acc)
        models_results['1D-CNN']['prec'].append(cnn_prec)
        models_results['1D-CNN']['rec'].append(cnn_rec)
        models_results['1D-CNN']['f1'].append(cnn_f1)
        models_results['1D-CNN']['time'].append(0.0)

        # --- LSTM ---
        # Normalize raw sequences per fold (using training stats)
        raw_mean = np.mean(np.vstack([X_raw[i] for i in train_idx]), axis=0)
        raw_std  = np.std(np.vstack([X_raw[i] for i in train_idx]), axis=0) + 1e-8
        X_train_lstm = [(X_raw[i] - raw_mean) / raw_std for i in train_idx]
        X_val_lstm   = [(X_raw[i] - raw_mean) / raw_std for i in val_idx]

        lstm_train_set = LSTMSequenceDataset(X_train_lstm, y_train.tolist())
        lstm_val_set   = LSTMSequenceDataset(X_val_lstm, y_val.tolist())
        lstm_train_loader = DataLoader(lstm_train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lstm_collate_fn)
        lstm_val_loader   = DataLoader(lstm_val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lstm_collate_fn)

        lstm_model = AimDetectorLSTM(len(FEATURE_NAMES)).to(device)
        lstm_optim = torch.optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)
        lstm_crit = nn.BCELoss()
        lstm_acc, lstm_prec, lstm_rec, lstm_f1 = train_deep_model(
            lstm_train_loader, lstm_val_loader, lstm_model, lstm_optim, lstm_crit, NUM_EPOCHS_DEEP, fold_num, model_name='LSTM')
        models_results['LSTM']['acc'].append(lstm_acc)
        models_results['LSTM']['prec'].append(lstm_prec)
        models_results['LSTM']['rec'].append(lstm_rec)
        models_results['LSTM']['f1'].append(lstm_f1)
        models_results['LSTM']['time'].append(0.0)

        print(f"Fold {fold_num} results:")
        for name, m in models_results.items():
            if len(m['f1'])>0:
                print(f"  {name:20s}: Acc={m['acc'][-1]:.3f}, F1={m['f1'][-1]:.3f}")

    # --- Final summary ---
    print("\n" + "="*60)
    print("Final Comparison (averaged over folds)")
    print("="*60)
    for name, m in models_results.items():
        f1s = m['f1']
        if not f1s: continue
        accs = m['acc']; precs = m['prec']; recs = m['rec']
        print(f"\n{name}:")
        print(f"  Accuracy : {np.mean(accs):.3f} (+/- {np.std(accs):.3f})")
        print(f"  Precision: {np.mean(precs):.3f} (+/- {np.std(precs):.3f})")
        print(f"  Recall   : {np.mean(recs):.3f} (+/- {np.std(recs):.3f})")
        print(f"  F1-score : {np.mean(f1s):.3f} (+/- {np.std(f1s):.3f})")

    # Boxplot
    model_names = list(models_results.keys())
    f1_data = [models_results[n]['f1'] for n in model_names]
    plt.figure(figsize=(10,6))
    plt.boxplot(f1_data, tick_labels=model_names)
    plt.title('Model Comparison – F1 Score Across 5 Folds')
    plt.ylabel('F1 Score'); plt.xticks(rotation=45, ha='right'); plt.grid(axis='y',alpha=0.5)
    plt.tight_layout(); plt.savefig('model_boxplot_f1.png'); plt.show()

    # LaTeX table
    print("\n% LaTeX table code")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Cross-validation results (mean $\\pm$ standard deviation across 5 folds).}")
    print("\\label{tab:model_comparison}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("Model & Accuracy & Precision & Recall & F1-score \\\\")
    print("\\midrule")
    for n in model_names:
        m = models_results[n]
        print(f"{n} & {np.mean(m['acc']):.3f} $\\pm$ {np.std(m['acc']):.3f} & {np.mean(m['prec']):.3f} $\\pm$ {np.std(m['prec']):.3f} & {np.mean(m['rec']):.3f} $\\pm$ {np.std(m['rec']):.3f} & {np.mean(m['f1']):.3f} $\\pm$ {np.std(m['f1']):.3f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    # Feature importance
    base_names = []
    for f in FEATURE_NAMES:
        base_names.extend([f'{f}_mean', f'{f}_std', f'{f}_min', f'{f}_max', f'{f}_med', f'{f}_p5', f'{f}_p95'])
    extra_names = ['ac_dyaw','ac_dpitch','frac_shooting','shooting_events','avg_speed','max_aim_corr','aim_on_target']
    all_feat_names = base_names + extra_names

    if rf_importances:
        avg_rf = np.mean(rf_importances, axis=0)
        idx = np.argsort(avg_rf)[::-1][:20]
        plt.figure(figsize=(10,6)); plt.barh(range(len(idx)), avg_rf[idx]); plt.yticks(range(len(idx)), [all_feat_names[i] for i in idx])
        plt.xlabel('Importance'); plt.title('Random Forest Feature Importances (avg over folds)'); plt.tight_layout()
        plt.savefig('rf_feature_importance.png'); plt.show()

    if xgb_importances:
        avg_xgb = np.mean(xgb_importances, axis=0)
        idx = np.argsort(avg_xgb)[::-1][:20]
        plt.figure(figsize=(10,6)); plt.barh(range(len(idx)), avg_xgb[idx]); plt.yticks(range(len(idx)), [all_feat_names[i] for i in idx])
        plt.xlabel('Importance'); plt.title('XGBoost Feature Importances (avg over folds)'); plt.tight_layout()
        plt.savefig('xgb_feature_importance.png'); plt.show()

    # -------------------------------------------------------------------------
    # Save final models on ALL data
    # -------------------------------------------------------------------------
    print("\nTraining final models on all data...")
    os.makedirs('models', exist_ok=True)
    final_scaler = StandardScaler().fit(X_stat)
    X_stat_all_s = final_scaler.transform(X_stat)

    # Logistic Regression
    final_lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED).fit(X_stat_all_s, y)
    joblib.dump(final_lr, 'models/lr_model.pkl')

    # Random Forest
    final_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_SEED).fit(X_stat_all_s, y)
    joblib.dump(final_rf, 'models/rf_model.pkl')
    joblib.dump(final_scaler, 'models/scaler.pkl')

    # XGBoost
    final_xgb = xgb.XGBClassifier(eval_metric='logloss', random_state=RANDOM_SEED).fit(X_stat_all_s, y)
    joblib.dump(final_xgb, 'models/xgb_model.pkl')

    # CNN final (retrain on all raw padded)
    cnn_final = Simple1DCNN(len(FEATURE_NAMES), FIXED_LEN).to(device)
    cnn_optim = torch.optim.Adam(cnn_final.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    cnn_crit = nn.BCELoss()
    cnn_all_set = TensorDataset(torch.tensor(X_cnn, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    cnn_all_loader = DataLoader(cnn_all_set, batch_size=BATCH_SIZE, shuffle=True)
    for epoch in range(30):
        cnn_final.train()
        for xb, yb in cnn_all_loader:
            xb, yb = xb.to(device), yb.to(device)
            cnn_optim.zero_grad()
            loss = cnn_crit(cnn_final(xb), yb)
            loss.backward(); cnn_optim.step()
    torch.save(cnn_final.state_dict(), 'models/cnn_final.pth')

    # LSTM final (retrain on all raw sequences, normalised globally)
    raw_mean_all = np.mean(np.vstack(X_raw), axis=0)
    raw_std_all = np.std(np.vstack(X_raw), axis=0) + 1e-8
    X_raw_all_norm = [(X_raw[i] - raw_mean_all) / raw_std_all for i in range(len(X_raw))]
    lstm_all_set = LSTMSequenceDataset(X_raw_all_norm, y.tolist())
    lstm_all_loader = DataLoader(lstm_all_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lstm_collate_fn)
    lstm_final = AimDetectorLSTM(len(FEATURE_NAMES)).to(device)
    lstm_optim = torch.optim.Adam(lstm_final.parameters(), lr=LEARNING_RATE)
    lstm_crit = nn.BCELoss()
    for epoch in range(30):
        lstm_final.train()
        for x, mask, yb in lstm_all_loader:
            x, mask, yb = x.to(device), mask.to(device), yb.to(device)
            lstm_optim.zero_grad()
            loss = lstm_crit(lstm_final(x, mask), yb)
            loss.backward(); lstm_optim.step()
    torch.save(lstm_final.state_dict(), 'models/lstm_final.pth')
    # Also save normalization stats for LSTM
    np.savez('models/lstm_norm.npz', mean=raw_mean_all, std=raw_std_all)

    print("All models and scaler saved to 'models/' directory.")

if __name__ == "__main__":
    main()