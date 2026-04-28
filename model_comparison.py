# -*- coding: utf-8 -*-
"""
Model comparison for aimbot detection:
Extracts statistical features from whole-match sequences and trains
Logistic Regression, Random Forest, XGBoost, and a 1D-CNN (PyTorch).

Uses the same match-level cross-validation as the LSTM script.
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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from collections import defaultdict

# XGBoost may need installation: pip install xgboost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not installed, skipping XGBoost.")

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Configuration – reuse your DEMO_FILES list
# -----------------------------------------------------------------------------
DEMO_FILES = [
    # Clean matches
    ("demo20260306_2225_local_ac_desert_10min_DM.json", {0: False, 1: False, 2: False, 3: False, 4: False,}),
    ("demo20260306_2245_local_ac_desert3_10min_DM.json", {0: False, 1: False, 2: False, 3: False, 4: False, 5: False}),
    ("demo20260306_2316_local_ac_lainio_10min_DM.json", {0: False, 1: False, 2: False, 3: False, 4: False, 5: False}),
    ("demo20260306_2331_local_ac_scaffold_10min_DM.json", {0: False, 1: False, 2: False, 3: False, 4: False, 5: False}),
    ("demo20260306_2341_local_ac_scaffold_10min_DM.json", {0: False, 1: False, 2: False, 3: False, 4: False, 5: False}),
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
    #---------------------------------

    # Mixed matches (one cheater, rest clean)
    ("demo20260325_1513_local_ac_desert3_10min_DM.json", {0: True, 1: False, 2: False, 3: False}),
    ("demo20260325_1523_local_ac_scaffold_10min_DM.json", {0: False, 1: False, 2: False, 3: True}),
    ("demo20260325_1533_local_ac_desert_10min_DM.json", {0: False, 1: False, 2: True, 3: False}),
    ("demo20260325_1544_local_ac_lainio_10min_DM.json", {0: False, 1: False, 2: False, 3: True}),
]

FEATURE_NAMES = [
    'x', 'y', 'z', 'yaw', 'pitch',
    'dx', 'dy', 'dz', 'dyaw', 'dpitch', 'shooting',
    'min_angle_to_enemy', 'aim_correction_delta'
]

RANDOM_SEED = 42
K_FOLDS = 5  # for cross-validation (adjust as needed)
FIXED_LEN = 2000  # for CNN: pad/truncate sequences to this length

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------------------------------------------------------
# Reused data loading functions (identical to your LSTM script)
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
            if dyaw > 180:
                dyaw -= 360
            elif dyaw < -180:
                dyaw += 360
            ev['dyaw'] = dyaw
            ev['dpitch'] = ev['pitch'] - prev['pitch']
        if 'shooting' not in ev:
            ev['shooting'] = 0
        prev = ev
    return events

def compute_aim_correction_features(player_events, all_players_events):
    if not player_events:
        return player_events
    subject_cn = player_events[0]['cn']
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
                nearest_idx = len(times)-1
            else:
                nearest_idx = idx-1 if abs(times[idx-1]-t) < abs(times[idx]-t) else idx
            other_pos = coords[nearest_idx]
            to_enemy = other_pos - my_pos
            dist = np.linalg.norm(to_enemy)
            if dist < 1e-6:
                continue
            req_yaw = math.degrees(math.atan2(to_enemy[1], to_enemy[0])) % 360
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
# Feature extraction from a whole-match sequence
# -----------------------------------------------------------------------------
def extract_statistical_features(seq):
    """
    seq: np.array of shape (T, F) - variable length.
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

    # Temporal features
    # Autocorrelation of dyaw and dpitch (lag 1) if seq length >= 2
    if seq.shape[0] >= 2:
        # indices: see order in FEATURE_NAMES: 'dyaw' idx=8, 'dpitch' idx=9
        dyaw_series = seq[1:, 8]  # shift by 1
        dyaw_lag = seq[:-1, 8]
        ac_dyaw = np.corrcoef(dyaw_lag, dyaw_series)[0, 1] if np.std(dyaw_lag) > 0 else 0.0

        dpitch_series = seq[1:, 9]
        dpitch_lag = seq[:-1, 9]
        ac_dpitch = np.corrcoef(dpitch_lag, dpitch_series)[0, 1] if np.std(dpitch_lag) > 0 else 0.0
    else:
        ac_dyaw = ac_dpitch = 0.0

    # Fraction of time shooting
    shooting_idx = 10  # 'shooting' index
    frac_shooting = np.mean(seq[:, shooting_idx])

    # Number of distinct shooting bursts? (simple: count transitions 0->1)
    shooting_events = np.sum(np.diff(seq[:, shooting_idx]) > 0.5)

    # Average distance travelled per second (dt between ticks is about 40ms, but we can approximate by total displacement)
    # Total path length: sum of instantaneous speed * dt (we don't have dt exactly, but we can sum absolute dx,dy,dz)
    total_dist = np.sum(np.sqrt(seq[:, 5]**2 + seq[:, 6]**2 + seq[:, 7]**2))  # dx,dy,dz at indices 5,6,7
    duration = len(seq) * 0.04  # assuming 40ms per tick? In AssaultCube, position events are roughly every 40ms. Adjust if needed.
    avg_speed = total_dist / duration if duration > 0 else 0.0

    # Maximum aim correction delta
    max_aim_corr = np.max(seq[:, 12])  # aim_correction_delta is index 12

    # Number of times aim_correction_delta < 10 degrees (aiming directly at enemy)
    aim_on_target = np.mean(seq[:, 12] < 10.0)

    # Concatenate all features
    features = np.concatenate([
        mean, std, min_val, max_val, median, p5, p95,
        [ac_dyaw, ac_dpitch, frac_shooting, shooting_events, avg_speed, max_aim_corr, aim_on_target]
    ])

    return features

# -----------------------------------------------------------------------------
# 1D-CNN Model (PyTorch)
# -----------------------------------------------------------------------------
class Simple1DCNN(nn.Module):
    def __init__(self, input_channels, seq_len, num_classes=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        # After two pooling layers, seq_len is reduced by factor 4
        self.fc1 = nn.Linear(64 * (seq_len // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        x = self.sigmoid(x).squeeze()
        return x

# -----------------------------------------------------------------------------
# Main comparison loop
# -----------------------------------------------------------------------------
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print("Loading and processing demos...")
    all_sequences = []   # list of numpy arrays (variable length)
    all_labels = []
    match_ids = []

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
        print("No valid sequences found.")
        return

    print(f"Total sequences: {len(all_sequences)}")
    print(f"Cheater: {sum(all_labels)}, Clean: {len(all_labels)-sum(all_labels)}")

    # Extract statistical features for each sequence
    print("Extracting statistical features...")
    X_stat = np.array([extract_statistical_features(seq) for seq in all_sequences])
    y = np.array(all_labels)
    print(f"Statistical feature vector shape: {X_stat.shape}")

    # For CNN: pad/truncate all sequences to FIXED_LEN
    print("Preparing fixed-length sequences for CNN...")
    X_cnn = np.zeros((len(all_sequences), len(FEATURE_NAMES), FIXED_LEN), dtype=np.float32)
    for i, seq in enumerate(all_sequences):
        T = seq.shape[0]
        if T <= FIXED_LEN:
            X_cnn[i, :, :T] = seq.T
        else:
            # Truncate to last FIXED_LEN timesteps (or random crop; we'll take last)
            X_cnn[i, :, :] = seq[-FIXED_LEN:, :].T

    # Prepare cross-validation splits (same match-level stratified)
    unique_matches = sorted(set(match_ids))
    match_cheat_label = []
    for mid in unique_matches:
        lbls = [all_labels[i] for i, m in enumerate(match_ids) if m == mid]
        match_cheat_label.append(1 if any(l == 1 for l in lbls) else 0)

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # Store results for each model
    models_results = {
        'Logistic Regression': {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'time': []},
        'Random Forest': {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'time': []},
        'XGBoost': {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'time': []},
        '1D-CNN': {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'time': []},
    }
    # For feature importance (RF and XGB)
    rf_importances = []
    xgb_importances = []

    fold_num = 0
    for train_match_idx, val_match_idx in skf.split(unique_matches, match_cheat_label):
        fold_num += 1
        print(f"\n{'='*50}\nFold {fold_num}/{K_FOLDS}\n{'='*50}")
        train_matches = set(unique_matches[i] for i in train_match_idx)
        val_matches   = set(unique_matches[i] for i in val_match_idx)

        train_idx = [i for i, m in enumerate(match_ids) if m in train_matches]
        val_idx   = [i for i, m in enumerate(match_ids) if m in val_matches]

        # Statistical features
        X_train_stat = X_stat[train_idx]
        y_train = y[train_idx]
        X_val_stat = X_stat[val_idx]
        y_val = y[val_idx]

        # CNN data
        X_train_cnn = X_cnn[train_idx]
        X_val_cnn = X_cnn[val_idx]

        # Normalize statistical features per fold (fit on train)
        scaler = StandardScaler()
        X_train_stat_scaled = scaler.fit_transform(X_train_stat)
        X_val_stat_scaled = scaler.transform(X_val_stat)

        # 1. Logistic Regression
        start = time.time()
        lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        lr.fit(X_train_stat_scaled, y_train)
        preds = lr.predict(X_val_stat_scaled)
        models_results['Logistic Regression']['acc'].append(accuracy_score(y_val, preds))
        models_results['Logistic Regression']['prec'].append(precision_score(y_val, preds, zero_division=0))
        models_results['Logistic Regression']['rec'].append(recall_score(y_val, preds, zero_division=0))
        models_results['Logistic Regression']['f1'].append(f1_score(y_val, preds, zero_division=0))
        models_results['Logistic Regression']['time'].append(time.time() - start)

        # 2. Random Forest (with simple hyperparameter tuning within fold)
        start = time.time()
        rf = RandomForestClassifier(random_state=RANDOM_SEED)
        # Quick grid search on small values
        param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5, None]}
        rf_gs = GridSearchCV(rf, param_grid, cv=3, scoring='f1')
        rf_gs.fit(X_train_stat_scaled, y_train)
        best_rf = rf_gs.best_estimator_
        preds = best_rf.predict(X_val_stat_scaled)
        models_results['Random Forest']['acc'].append(accuracy_score(y_val, preds))
        models_results['Random Forest']['prec'].append(precision_score(y_val, preds, zero_division=0))
        models_results['Random Forest']['rec'].append(recall_score(y_val, preds, zero_division=0))
        models_results['Random Forest']['f1'].append(f1_score(y_val, preds, zero_division=0))
        models_results['Random Forest']['time'].append(time.time() - start)
        rf_importances.append(best_rf.feature_importances_)

        # 3. XGBoost (if available)
        if XGB_AVAILABLE:
            start = time.time()
            xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=RANDOM_SEED)
            param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5]}
            xgb_gs = GridSearchCV(xgb_model, param_grid, cv=3, scoring='f1')
            xgb_gs.fit(X_train_stat_scaled, y_train)
            best_xgb = xgb_gs.best_estimator_
            preds = best_xgb.predict(X_val_stat_scaled)
            models_results['XGBoost']['acc'].append(accuracy_score(y_val, preds))
            models_results['XGBoost']['prec'].append(precision_score(y_val, preds, zero_division=0))
            models_results['XGBoost']['rec'].append(recall_score(y_val, preds, zero_division=0))
            models_results['XGBoost']['f1'].append(f1_score(y_val, preds, zero_division=0))
            models_results['XGBoost']['time'].append(time.time() - start)
            xgb_importances.append(best_xgb.feature_importances_)

        # 4. 1D-CNN (train from scratch each fold)
        start = time.time()
        X_train_cnn_t = torch.tensor(X_train_cnn, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_val_cnn_t = torch.tensor(X_val_cnn, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

        train_dataset = TensorDataset(X_train_cnn_t, y_train_t)
        val_dataset = TensorDataset(X_val_cnn_t, y_val_t)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

        cnn_model = Simple1DCNN(input_channels=len(FEATURE_NAMES), seq_len=FIXED_LEN).to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001, weight_decay=1e-4)

        best_val_f1 = -1.0          # changed from 0.0 to -1.0 so that any real F1 will be an improvement
        best_state = cnn_model.state_dict().copy()   # initialise with current (random) state

        # Train for max 20 epochs, early stopping
        for epoch in range(20):
            cnn_model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = cnn_model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

            # Validation
            cnn_model.eval()
            val_preds = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = cnn_model(xb)
                    # Ensure output is at least 1D, convert to list of ints
                    preds = (out > 0.5).int().view(-1).cpu().numpy().tolist()
                    val_preds.extend(preds)
            val_f1 = f1_score(y_val, val_preds, zero_division=0)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = cnn_model.state_dict().copy()

        # Load the best (or final if never improved) model
        cnn_model.load_state_dict(best_state)
        cnn_model.eval()
        final_preds = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = cnn_model(xb)
                preds = (out > 0.5).int().view(-1).cpu().numpy().tolist()
                final_preds.extend(preds)

        models_results['1D-CNN']['acc'].append(accuracy_score(y_val, final_preds))
        models_results['1D-CNN']['prec'].append(precision_score(y_val, final_preds, zero_division=0))
        models_results['1D-CNN']['rec'].append(recall_score(y_val, final_preds, zero_division=0))
        models_results['1D-CNN']['f1'].append(f1_score(y_val, final_preds, zero_division=0))
        models_results['1D-CNN']['time'].append(time.time() - start)

        print(f"Fold {fold_num} results:")
        for model_name, metrics in models_results.items():
            if len(metrics['f1']) > 0:
                print(f"  {model_name:20s}: Acc={metrics['acc'][-1]:.3f}, F1={metrics['f1'][-1]:.3f}")

    # -------------------------------------------------------------------------
    # Print final summary
    print("\n" + "="*60)
    print("Final Comparison (averaged over folds)")
    print("="*60)
    for model_name, metrics in models_results.items():
        if len(metrics['f1']) == 0:
            continue
        print(f"\n{model_name}:")
        print(f"  Accuracy : {np.mean(metrics['acc']):.3f} (+/- {np.std(metrics['acc']):.3f})")
        print(f"  Precision: {np.mean(metrics['prec']):.3f} (+/- {np.std(metrics['prec']):.3f})")
        print(f"  Recall   : {np.mean(metrics['rec']):.3f} (+/- {np.std(metrics['rec']):.3f})")
        print(f"  F1-score : {np.mean(metrics['f1']):.3f} (+/- {np.std(metrics['f1']):.3f})")
        print(f"  Avg train time: {np.mean(metrics['time']):.3f}s")

    # -------------------------------------------------------------------------
    # Feature importance plot for Random Forest and XGBoost
    # Generate feature names for the statistical vector
    base_names = []
    for f in FEATURE_NAMES:
        base_names.extend([f'{f}_mean', f'{f}_std', f'{f}_min', f'{f}_max', f'{f}_med', f'{f}_p5', f'{f}_p95'])
    extra_names = ['ac_dyaw', 'ac_dpitch', 'frac_shooting', 'shooting_events', 'avg_speed', 'max_aim_corr', 'aim_on_target']
    all_feature_names = base_names + extra_names

    if len(rf_importances) > 0:
        # Average importance across folds
        avg_rf_imp = np.mean(rf_importances, axis=0)
        plt.figure(figsize=(10, 6))
        indices = np.argsort(avg_rf_imp)[::-1][:20]  # top 20
        plt.barh(range(len(indices)), avg_rf_imp[indices], align='center')
        plt.yticks(range(len(indices)), [all_feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title('Random Forest Feature Importances (avg over folds)')
        plt.tight_layout()
        plt.savefig('rf_feature_importance.png')
        plt.show()

    if XGB_AVAILABLE and len(xgb_importances) > 0:
        avg_xgb_imp = np.mean(xgb_importances, axis=0)
        plt.figure(figsize=(10, 6))
        indices = np.argsort(avg_xgb_imp)[::-1][:20]
        plt.barh(range(len(indices)), avg_xgb_imp[indices], align='center')
        plt.yticks(range(len(indices)), [all_feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title('XGBoost Feature Importances (avg over folds)')
        plt.tight_layout()
        plt.savefig('xgb_feature_importance.png')
        plt.show()

if __name__ == "__main__":
    main()