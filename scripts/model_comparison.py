# -*- coding: utf-8 -*-
"""
Model comparison for aimbot detection:
Extracts statistical features from whole-match sequences and trains
Logistic Regression, Random Forest, XGBoost, a 1D-CNN, and a simple
average ensemble. Produces a LaTeX comparison table and a boxplot
of F1 scores across folds.
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
from utils import *

# XGBoost may need installation: pip install xgboost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not installed, skipping XGBoost.")

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Configuration
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

FEATURE_NAMES = [
    'x', 'y', 'z', 'yaw', 'pitch',
    'dx', 'dy', 'dz', 'dyaw', 'dpitch', 'shooting',
    'min_angle_to_enemy', 'aim_correction_delta'
]

RANDOM_SEED = 42
K_FOLDS = 5
FIXED_LEN = 2000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------
def extract_statistical_features(seq):
    """
    seq: np.array of shape (T, F) - variable length.
    Returns a fixed-length feature vector (1D array).
    """
    mean = np.mean(seq, axis=0)
    std = np.std(seq, axis=0)
    min_val = np.min(seq, axis=0)
    max_val = np.max(seq, axis=0)
    median = np.median(seq, axis=0)
    p5 = np.percentile(seq, 5, axis=0)
    p95 = np.percentile(seq, 95, axis=0)

    if seq.shape[0] >= 2:
        dyaw_series = seq[1:, 8]
        dyaw_lag = seq[:-1, 8]
        ac_dyaw = np.corrcoef(dyaw_lag, dyaw_series)[0, 1] if np.std(dyaw_lag) > 0 else 0.0

        dpitch_series = seq[1:, 9]
        dpitch_lag = seq[:-1, 9]
        ac_dpitch = np.corrcoef(dpitch_lag, dpitch_series)[0, 1] if np.std(dpitch_lag) > 0 else 0.0
    else:
        ac_dyaw = ac_dpitch = 0.0

    shooting_idx = 10
    frac_shooting = np.mean(seq[:, shooting_idx])
    shooting_events = np.sum(np.diff(seq[:, shooting_idx]) > 0.5)

    total_dist = np.sum(np.sqrt(seq[:, 5]**2 + seq[:, 6]**2 + seq[:, 7]**2))
    duration = len(seq) * 0.04
    avg_speed = total_dist / duration if duration > 0 else 0.0

    max_aim_corr = np.max(seq[:, 12])
    aim_on_target = np.mean(seq[:, 12] < 10.0)

    features = np.concatenate([
        mean, std, min_val, max_val, median, p5, p95,
        [ac_dyaw, ac_dpitch, frac_shooting, shooting_events, avg_speed, max_aim_corr, aim_on_target]
    ])
    return features

# -----------------------------------------------------------------------------
# 1D-CNN Model
# -----------------------------------------------------------------------------
class Simple1DCNN(nn.Module):
    def __init__(self, input_channels, seq_len, num_classes=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * (seq_len // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)                    # shape: (batch, 1)
        x = self.sigmoid(x).squeeze(-1)    # shape: (batch,)  even for batch=1 -> (1,)
        return x

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

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

    # Extract statistical features
    print("Extracting statistical features...")
    X_stat = np.array([extract_statistical_features(seq) for seq in all_sequences])
    y = np.array(all_labels)
    print(f"Statistical feature vector shape: {X_stat.shape}")

    # Prepare fixed-length sequences for CNN
    print("Preparing fixed-length sequences for CNN...")
    X_cnn = np.zeros((len(all_sequences), len(FEATURE_NAMES), FIXED_LEN), dtype=np.float32)
    for i, seq in enumerate(all_sequences):
        T = seq.shape[0]
        if T <= FIXED_LEN:
            X_cnn[i, :, :T] = seq.T
        else:
            X_cnn[i, :, :] = seq[-FIXED_LEN:, :].T

    # Cross-validation splits
    unique_matches = sorted(set(match_ids))
    match_cheat_label = []
    for mid in unique_matches:
        lbls = [all_labels[i] for i, m in enumerate(match_ids) if m == mid]
        match_cheat_label.append(1 if any(l == 1 for l in lbls) else 0)

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # Store results for each model (add Simple Average)
    models_results = {
        'Logistic Regression': {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'time': []},
        'Random Forest': {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'time': []},
        'XGBoost': {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'time': []},
        '1D-CNN': {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'time': []},
        'Simple Average': {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'time': []}
    }
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

        X_train_stat = X_stat[train_idx]
        y_train = y[train_idx]
        X_val_stat = X_stat[val_idx]
        y_val = y[val_idx]

        X_train_cnn = X_cnn[train_idx]
        X_val_cnn = X_cnn[val_idx]

        # Standardisation
        scaler = StandardScaler()
        X_train_stat_scaled = scaler.fit_transform(X_train_stat)
        X_val_stat_scaled = scaler.transform(X_val_stat)

        # --- 1. Logistic Regression ---
        start = time.time()
        lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        lr.fit(X_train_stat_scaled, y_train)
        lr_proba = lr.predict_proba(X_val_stat_scaled)[:, 1]
        lr_preds = (lr_proba >= 0.5).astype(int)
        models_results['Logistic Regression']['acc'].append(accuracy_score(y_val, lr_preds))
        models_results['Logistic Regression']['prec'].append(precision_score(y_val, lr_preds, zero_division=0))
        models_results['Logistic Regression']['rec'].append(recall_score(y_val, lr_preds, zero_division=0))
        models_results['Logistic Regression']['f1'].append(f1_score(y_val, lr_preds, zero_division=0))
        models_results['Logistic Regression']['time'].append(time.time() - start)

        # --- 2. Random Forest ---
        start = time.time()
        rf = RandomForestClassifier(random_state=RANDOM_SEED)
        param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5, None]}
        rf_gs = GridSearchCV(rf, param_grid, cv=3, scoring='f1')
        rf_gs.fit(X_train_stat_scaled, y_train)
        best_rf = rf_gs.best_estimator_
        rf_proba = best_rf.predict_proba(X_val_stat_scaled)[:, 1]
        rf_preds = (rf_proba >= 0.5).astype(int)
        models_results['Random Forest']['acc'].append(accuracy_score(y_val, rf_preds))
        models_results['Random Forest']['prec'].append(precision_score(y_val, rf_preds, zero_division=0))
        models_results['Random Forest']['rec'].append(recall_score(y_val, rf_preds, zero_division=0))
        models_results['Random Forest']['f1'].append(f1_score(y_val, rf_preds, zero_division=0))
        models_results['Random Forest']['time'].append(time.time() - start)
        rf_importances.append(best_rf.feature_importances_)

        # --- 3. XGBoost ---
        xgb_proba = None
        if XGB_AVAILABLE:
            start = time.time()
            xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=RANDOM_SEED)
            param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5]}
            xgb_gs = GridSearchCV(xgb_model, param_grid, cv=3, scoring='f1')
            xgb_gs.fit(X_train_stat_scaled, y_train)
            best_xgb = xgb_gs.best_estimator_
            xgb_proba = best_xgb.predict_proba(X_val_stat_scaled)[:, 1]
            xgb_preds = (xgb_proba >= 0.5).astype(int)
            models_results['XGBoost']['acc'].append(accuracy_score(y_val, xgb_preds))
            models_results['XGBoost']['prec'].append(precision_score(y_val, xgb_preds, zero_division=0))
            models_results['XGBoost']['rec'].append(recall_score(y_val, xgb_preds, zero_division=0))
            models_results['XGBoost']['f1'].append(f1_score(y_val, xgb_preds, zero_division=0))
            models_results['XGBoost']['time'].append(time.time() - start)
            xgb_importances.append(best_xgb.feature_importances_)
        else:
            # fill with NaN if XGB not available
            models_results['XGBoost']['acc'].append(float('nan'))
            models_results['XGBoost']['prec'].append(float('nan'))
            models_results['XGBoost']['rec'].append(float('nan'))
            models_results['XGBoost']['f1'].append(float('nan'))
            models_results['XGBoost']['time'].append(float('nan'))

        # --- 4. Simple Average Ensemble (works even if XGBoost missing) ---
        if xgb_proba is not None:
            avg_proba = (lr_proba + rf_proba + xgb_proba) / 3.0
        else:
            avg_proba = (lr_proba + rf_proba) / 2.0
        avg_preds = (avg_proba >= 0.5).astype(int)
        models_results['Simple Average']['acc'].append(accuracy_score(y_val, avg_preds))
        models_results['Simple Average']['prec'].append(precision_score(y_val, avg_preds, zero_division=0))
        models_results['Simple Average']['rec'].append(recall_score(y_val, avg_preds, zero_division=0))
        models_results['Simple Average']['f1'].append(f1_score(y_val, avg_preds, zero_division=0))
        models_results['Simple Average']['time'].append(0.0)  # negligible

        # --- 5. 1D-CNN ---
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

        best_val_f1 = -1.0
        best_state = cnn_model.state_dict().copy()

        for epoch in range(20):
            cnn_model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = cnn_model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

            cnn_model.eval()
            val_preds = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = cnn_model(xb)
                    preds = (out > 0.5).int().view(-1).cpu().numpy().tolist()
                    val_preds.extend(preds)
            val_f1 = f1_score(y_val, val_preds, zero_division=0)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = cnn_model.state_dict().copy()

        cnn_model.load_state_dict(best_state)
        cnn_model.eval()
        final_preds = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = cnn_model(xb)
                final_preds.extend((out > 0.5).int().view(-1).cpu().numpy().tolist())

        models_results['1D-CNN']['acc'].append(accuracy_score(y_val, final_preds))
        models_results['1D-CNN']['prec'].append(precision_score(y_val, final_preds, zero_division=0))
        models_results['1D-CNN']['rec'].append(recall_score(y_val, final_preds, zero_division=0))
        models_results['1D-CNN']['f1'].append(f1_score(y_val, final_preds, zero_division=0))
        models_results['1D-CNN']['time'].append(time.time() - start)

        # Fold summary
        print(f"Fold {fold_num} results:")
        for model_name, metrics in models_results.items():
            if len(metrics['f1']) > 0 and not np.isnan(metrics['f1'][-1]):
                print(f"  {model_name:20s}: Acc={metrics['acc'][-1]:.3f}, F1={metrics['f1'][-1]:.3f}")

    # -------------------------------------------------------------------------
    # Final summary (ignoring NaN)
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("Final Comparison (averaged over folds)")
    print("="*60)
    for model_name, metrics in models_results.items():
        f1_scores = [v for v in metrics['f1'] if not np.isnan(v)]
        if len(f1_scores) == 0:
            continue
        acc_scores = [v for v in metrics['acc'] if not np.isnan(v)]
        prec_scores = [v for v in metrics['prec'] if not np.isnan(v)]
        rec_scores = [v for v in metrics['rec'] if not np.isnan(v)]
        print(f"\n{model_name}:")
        print(f"  Accuracy : {np.mean(acc_scores):.3f} (+/- {np.std(acc_scores):.3f})")
        print(f"  Precision: {np.mean(prec_scores):.3f} (+/- {np.std(prec_scores):.3f})")
        print(f"  Recall   : {np.mean(rec_scores):.3f} (+/- {np.std(rec_scores):.3f})")
        print(f"  F1-score : {np.mean(f1_scores):.3f} (+/- {np.std(f1_scores):.3f})")
        if 'time' in metrics:
            times = [v for v in metrics['time'] if not np.isnan(v)]
            print(f"  Avg train time: {np.mean(times):.3f}s")

    # -------------------------------------------------------------------------
    # Boxplot of F1 scores
    # -------------------------------------------------------------------------
    model_names_for_plot = [m for m in models_results.keys() if any(not np.isnan(v) for v in models_results[m]['f1'])]
    f1_data = [models_results[m]['f1'] for m in model_names_for_plot]

    plt.figure(figsize=(10, 6))
    plt.boxplot(f1_data, tick_labels=model_names_for_plot)
    plt.title('Model Comparison – F1 Score Across 5 Folds')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig('model_boxplot_f1.png')
    plt.show()

    # -------------------------------------------------------------------------
    # LaTeX table
    # -------------------------------------------------------------------------
    print("\n% LaTeX table code")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Cross-validation results (mean $\\pm$ standard deviation across 5 folds).}")
    print("\\label{tab:model_comparison}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("Model & Accuracy & Precision & Recall & F1-score \\\\")
    print("\\midrule")
    for name in model_names_for_plot:
        m = models_results[name]
        acc = f"{np.mean(m['acc']):.3f} $\\pm$ {np.std(m['acc']):.3f}"
        prec = f"{np.mean(m['prec']):.3f} $\\pm$ {np.std(m['prec']):.3f}"
        rec = f"{np.mean(m['rec']):.3f} $\\pm$ {np.std(m['rec']):.3f}"
        f1 = f"{np.mean(m['f1']):.3f} $\\pm$ {np.std(m['f1']):.3f}"
        print(f"{name} & {acc} & {prec} & {rec} & {f1} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    # -------------------------------------------------------------------------
    # Feature importance plots
    # -------------------------------------------------------------------------
    base_names = []
    for f in FEATURE_NAMES:
        base_names.extend([f'{f}_mean', f'{f}_std', f'{f}_min', f'{f}_max', f'{f}_med', f'{f}_p5', f'{f}_p95'])
    extra_names = ['ac_dyaw', 'ac_dpitch', 'frac_shooting', 'shooting_events', 'avg_speed', 'max_aim_corr', 'aim_on_target']
    all_feature_names = base_names + extra_names

    if len(rf_importances) > 0:
        avg_rf_imp = np.mean(rf_importances, axis=0)
        plt.figure(figsize=(10, 6))
        indices = np.argsort(avg_rf_imp)[::-1][:20]
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

    # -------------------------------------------------------------------------
    # Save final Random Forest model and scaler
    # -------------------------------------------------------------------------
    print("\nTraining final Random Forest on all data...")
    final_scaler = StandardScaler()
    X_stat_scaled_all = final_scaler.fit_transform(X_stat)
    final_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_SEED)
    final_rf.fit(X_stat_scaled_all, y)

    import joblib
    os.makedirs('models', exist_ok=True)
    joblib.dump(final_rf, 'models/rf_model.pkl')
    joblib.dump(final_scaler, 'models/scaler.pkl')
    # Save final LR and XGBoost as well
    final_lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    final_lr.fit(X_stat_scaled_all, y)
    joblib.dump(final_lr, 'models/lr_model.pkl')

    final_xgb = xgb.XGBClassifier(eval_metric='logloss', random_state=RANDOM_SEED)
    final_xgb.fit(X_stat_scaled_all, y)
    joblib.dump(final_xgb, 'models/xgb_model.pkl')
    print("Saved final model and scaler to 'models/' directory.")


if __name__ == "__main__":
    main()