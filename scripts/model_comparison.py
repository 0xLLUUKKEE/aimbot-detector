"""Unified model comparison for aimbot detection.

Trains and evaluates classical models (Logistic Regression, Random Forest,
XGBoost, plus a simple-average ensemble) and deep models (1D-CNN, LSTM in the
Dunham style) under the same match-level 5-fold cross-validation, and saves
the final models and scaler for later inference.
"""

import os
import random
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset, TensorDataset

from utils import (
    FEATURE_NAMES,
    build_full_sequence,
    build_player_timelines,
    extract_statistical_features,
    load_events,
)

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Data manifest
# -----------------------------------------------------------------------------
# Each entry: (json_path, {client_number: is_cheater}).
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

# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
RANDOM_SEED = 42
K_FOLDS = 5
FIXED_LEN = 2000           # Tick budget for the 1D-CNN (LSTM uses packed variable lengths).
BATCH_SIZE = 4
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 0.001
NUM_EPOCHS_DEEP = 30       # LSTM training epochs.
NUM_EPOCHS_CNN = 20
NUM_EPOCHS_FINAL = 30      # Used when retraining both deep nets on all data.
DECISION_THRESHOLD = 0.5
MODELS_DIR = 'models'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -----------------------------------------------------------------------------
# Deep model definitions
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


class AimDetectorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask):
        lengths = mask.sum(dim=1).cpu()
        x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(x_packed)
        return self.sigmoid(self.fc(hidden[-1])).squeeze(-1)


class LSTMSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


def lstm_collate_fn(batch):
    seqs, lbls = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    padded = pad_sequence(seqs, batch_first=True)
    max_len = padded.size(1)
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    return padded, mask, torch.stack(lbls)


# -----------------------------------------------------------------------------
# Training / evaluation helpers
# -----------------------------------------------------------------------------
def _forward_batch(model, batch, model_name):
    if model_name == 'LSTM':
        x, mask, y = batch
        x, mask, y = x.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
        return model(x, mask), y, x.size(0)
    x, y = batch
    x, y = x.to(DEVICE), y.to(DEVICE)
    return model(x), y, x.size(0)


def _predict_loader(model, loader, model_name):
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for batch in loader:
            out, y, _ = _forward_batch(model, batch, model_name)
            preds.extend((out > DECISION_THRESHOLD).int().cpu().numpy().tolist())
            true.extend(y.cpu().numpy().tolist())
    return preds, true


def train_deep_model(train_loader, val_loader, model, optimizer, criterion, epochs, fold, model_name):
    best_val_f1 = -1.0
    best_state = model.state_dict().copy()
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch in train_loader:
            preds, y, batch_size = _forward_batch(model, batch, model_name)
            loss = criterion(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_size
            train_correct += ((preds > DECISION_THRESHOLD).float() == y).sum().item()
            train_total += batch_size

        val_preds, val_true = _predict_loader(model, val_loader, model_name)
        val_f1 = f1_score(val_true, val_preds, zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()

        if (epoch + 1) % 5 == 0:
            print(
                f"Fold {fold} | {model_name} Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss / train_total:.4f} "
                f"Acc: {train_correct / train_total:.4f} | "
                f"Val F1: {val_f1:.4f}"
            )

    model.load_state_dict(best_state)
    final_preds, final_true = _predict_loader(model, val_loader, model_name)
    return (
        accuracy_score(final_true, final_preds),
        precision_score(final_true, final_preds, zero_division=0),
        recall_score(final_true, final_preds, zero_division=0),
        f1_score(final_true, final_preds, zero_division=0),
    )


def _append_classical_metrics(results_dict, y_true, y_pred, elapsed):
    results_dict['acc'].append(accuracy_score(y_true, y_pred))
    results_dict['prec'].append(precision_score(y_true, y_pred, zero_division=0))
    results_dict['rec'].append(recall_score(y_true, y_pred, zero_division=0))
    results_dict['f1'].append(f1_score(y_true, y_pred, zero_division=0))
    results_dict['time'].append(elapsed)


def _statistical_feature_names():
    base_names = []
    for f in FEATURE_NAMES:
        base_names.extend([f'{f}_mean', f'{f}_std', f'{f}_min', f'{f}_max',
                           f'{f}_med', f'{f}_p5', f'{f}_p95'])
    extra_names = ['ac_dyaw', 'ac_dpitch', 'frac_shooting', 'shooting_events',
                   'avg_speed', 'max_aim_corr', 'aim_on_target']
    return base_names + extra_names


def _plot_feature_importance(importances, all_feat_names, title, out_path):
    avg_imp = np.mean(importances, axis=0)
    idx = np.argsort(avg_imp)[::-1][:20]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(idx)), avg_imp[idx])
    plt.yticks(range(len(idx)), [all_feat_names[i] for i in idx])
    plt.xlabel('Importance')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    print("Loading and processing demos...")
    all_sequences, all_labels, match_ids = [], [], []
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

    if not all_sequences:
        print("No valid sequences found.")
        return

    print(f"Total sequences: {len(all_sequences)}")
    print(f"Cheater: {sum(all_labels)}, Clean: {len(all_labels) - sum(all_labels)}")

    # Statistical features for the classical models.
    X_stat = np.array([extract_statistical_features(s) for s in all_sequences])
    y = np.array(all_labels)
    print(f"Statistical feature vector shape: {X_stat.shape}")

    # Raw variable-length sequences for the LSTM (normalised per fold below).
    X_raw = all_sequences

    # Fixed-length, channels-first tensors for the CNN.
    print("Preparing fixed-length sequences for CNN...")
    X_cnn = np.zeros((len(all_sequences), len(FEATURE_NAMES), FIXED_LEN), dtype=np.float32)
    for i, seq in enumerate(all_sequences):
        T = seq.shape[0]
        if T <= FIXED_LEN:
            X_cnn[i, :, :T] = seq.T
        else:
            X_cnn[i, :, :] = seq[-FIXED_LEN:, :].T

    # Stratify folds at the match level so no player straddles train/val.
    unique_matches = sorted(set(match_ids))
    match_cheat_label = []
    for mid in unique_matches:
        lbls = [all_labels[i] for i, m in enumerate(match_ids) if m == mid]
        match_cheat_label.append(1 if any(l == 1 for l in lbls) else 0)
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    def empty_metrics():
        return {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'time': []}

    models_results = {
        'Logistic Regression': empty_metrics(),
        'Random Forest':       empty_metrics(),
        'XGBoost':             empty_metrics(),
        '1D-CNN':              empty_metrics(),
        'LSTM':                empty_metrics(),
        'Simple Average':      empty_metrics(),
    }
    rf_importances, xgb_importances = [], []

    for fold_num, (train_match_idx, val_match_idx) in enumerate(
            skf.split(unique_matches, match_cheat_label), start=1):
        print(f"\n{'=' * 50}\nFold {fold_num}/{K_FOLDS}\n{'=' * 50}")
        train_m = {unique_matches[i] for i in train_match_idx}
        val_m = {unique_matches[i] for i in val_match_idx}

        train_idx = [i for i, m in enumerate(match_ids) if m in train_m]
        val_idx = [i for i, m in enumerate(match_ids) if m in val_m]

        y_train, y_val = y[train_idx], y[val_idx]

        # ---- Classical models on statistical features ----
        X_train_stat, X_val_stat = X_stat[train_idx], X_stat[val_idx]
        scaler = StandardScaler().fit(X_train_stat)
        X_train_stat_s = scaler.transform(X_train_stat)
        X_val_stat_s = scaler.transform(X_val_stat)

        # Logistic Regression
        start = time.time()
        lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED).fit(X_train_stat_s, y_train)
        lr_proba = lr.predict_proba(X_val_stat_s)[:, 1]
        lr_preds = (lr_proba >= DECISION_THRESHOLD).astype(int)
        _append_classical_metrics(models_results['Logistic Regression'], y_val, lr_preds, time.time() - start)

        # Random Forest with small grid search.
        start = time.time()
        rf_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5, None]}
        best_rf = GridSearchCV(
            RandomForestClassifier(random_state=RANDOM_SEED),
            rf_grid, cv=3, scoring='f1',
        ).fit(X_train_stat_s, y_train).best_estimator_
        rf_proba = best_rf.predict_proba(X_val_stat_s)[:, 1]
        rf_preds = (rf_proba >= DECISION_THRESHOLD).astype(int)
        _append_classical_metrics(models_results['Random Forest'], y_val, rf_preds, time.time() - start)
        rf_importances.append(best_rf.feature_importances_)

        # XGBoost with small grid search.
        start = time.time()
        xgb_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5]}
        best_xgb = GridSearchCV(
            xgb.XGBClassifier(eval_metric='logloss', random_state=RANDOM_SEED),
            xgb_grid, cv=3, scoring='f1',
        ).fit(X_train_stat_s, y_train).best_estimator_
        xgb_proba = best_xgb.predict_proba(X_val_stat_s)[:, 1]
        xgb_preds = (xgb_proba >= DECISION_THRESHOLD).astype(int)
        _append_classical_metrics(models_results['XGBoost'], y_val, xgb_preds, time.time() - start)
        xgb_importances.append(best_xgb.feature_importances_)

        # Simple Average ensemble of the three classical probabilities.
        avg_proba = (lr_proba + rf_proba + xgb_proba) / 3.0
        avg_preds = (avg_proba >= DECISION_THRESHOLD).astype(int)
        _append_classical_metrics(models_results['Simple Average'], y_val, avg_preds, 0.0)

        # ---- 1D-CNN ----
        X_train_cnn, X_val_cnn = X_cnn[train_idx], X_cnn[val_idx]
        cnn_train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train_cnn), torch.tensor(y_train, dtype=torch.float32)),
            batch_size=BATCH_SIZE, shuffle=True,
        )
        cnn_val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val_cnn), torch.tensor(y_val, dtype=torch.float32)),
            batch_size=BATCH_SIZE, shuffle=False,
        )

        cnn_model = Simple1DCNN(len(FEATURE_NAMES), FIXED_LEN).to(DEVICE)
        cnn_optim = torch.optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        cnn_metrics = train_deep_model(
            cnn_train_loader, cnn_val_loader, cnn_model, cnn_optim, nn.BCELoss(),
            NUM_EPOCHS_CNN, fold_num, model_name='CNN',
        )
        for key, value in zip(('acc', 'prec', 'rec', 'f1'), cnn_metrics):
            models_results['1D-CNN'][key].append(value)
        models_results['1D-CNN']['time'].append(0.0)

        # ---- LSTM (variable-length, fold-local normalisation) ----
        raw_mean = np.mean(np.vstack([X_raw[i] for i in train_idx]), axis=0)
        raw_std = np.std(np.vstack([X_raw[i] for i in train_idx]), axis=0) + 1e-8
        X_train_lstm = [(X_raw[i] - raw_mean) / raw_std for i in train_idx]
        X_val_lstm = [(X_raw[i] - raw_mean) / raw_std for i in val_idx]

        lstm_train_loader = DataLoader(
            LSTMSequenceDataset(X_train_lstm, y_train.tolist()),
            batch_size=BATCH_SIZE, shuffle=True, collate_fn=lstm_collate_fn,
        )
        lstm_val_loader = DataLoader(
            LSTMSequenceDataset(X_val_lstm, y_val.tolist()),
            batch_size=BATCH_SIZE, shuffle=False, collate_fn=lstm_collate_fn,
        )

        lstm_model = AimDetectorLSTM(len(FEATURE_NAMES)).to(DEVICE)
        lstm_optim = torch.optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)
        lstm_metrics = train_deep_model(
            lstm_train_loader, lstm_val_loader, lstm_model, lstm_optim, nn.BCELoss(),
            NUM_EPOCHS_DEEP, fold_num, model_name='LSTM',
        )
        for key, value in zip(('acc', 'prec', 'rec', 'f1'), lstm_metrics):
            models_results['LSTM'][key].append(value)
        models_results['LSTM']['time'].append(0.0)

        print(f"Fold {fold_num} results:")
        for name, m in models_results.items():
            if m['f1']:
                print(f"  {name:20s}: Acc={m['acc'][-1]:.3f}, F1={m['f1'][-1]:.3f}")

    # --- Final summary across folds ---
    print("\n" + "=" * 60)
    print("Final Comparison (averaged over folds)")
    print("=" * 60)
    for name, m in models_results.items():
        if not m['f1']:
            continue
        print(f"\n{name}:")
        print(f"  Accuracy : {np.mean(m['acc']):.3f} (+/- {np.std(m['acc']):.3f})")
        print(f"  Precision: {np.mean(m['prec']):.3f} (+/- {np.std(m['prec']):.3f})")
        print(f"  Recall   : {np.mean(m['rec']):.3f} (+/- {np.std(m['rec']):.3f})")
        print(f"  F1-score : {np.mean(m['f1']):.3f} (+/- {np.std(m['f1']):.3f})")

    # Boxplot of fold-level F1 scores.
    model_names = list(models_results.keys())
    f1_data = [models_results[n]['f1'] for n in model_names]
    plt.figure(figsize=(10, 6))
    plt.boxplot(f1_data, tick_labels=model_names)
    plt.title('Model Comparison – F1 Score Across 5 Folds')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig('model_boxplot_f1.png')
    plt.show()

    # LaTeX results table.
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
        print(
            f"{n} & "
            f"{np.mean(m['acc']):.3f} $\\pm$ {np.std(m['acc']):.3f} & "
            f"{np.mean(m['prec']):.3f} $\\pm$ {np.std(m['prec']):.3f} & "
            f"{np.mean(m['rec']):.3f} $\\pm$ {np.std(m['rec']):.3f} & "
            f"{np.mean(m['f1']):.3f} $\\pm$ {np.std(m['f1']):.3f} \\\\"
        )
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    # Feature importances averaged over folds.
    all_feat_names = _statistical_feature_names()
    if rf_importances:
        _plot_feature_importance(
            rf_importances, all_feat_names,
            'Random Forest Feature Importances (avg over folds)',
            'rf_feature_importance.png',
        )
    if xgb_importances:
        _plot_feature_importance(
            xgb_importances, all_feat_names,
            'XGBoost Feature Importances (avg over folds)',
            'xgb_feature_importance.png',
        )

    # -------------------------------------------------------------------------
    # Retrain on all data and save final artefacts.
    # -------------------------------------------------------------------------
    print("\nTraining final models on all data...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    final_scaler = StandardScaler().fit(X_stat)
    X_stat_all_s = final_scaler.transform(X_stat)
    joblib.dump(final_scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))

    final_lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED).fit(X_stat_all_s, y)
    joblib.dump(final_lr, os.path.join(MODELS_DIR, 'lr_model.pkl'))

    final_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_SEED).fit(X_stat_all_s, y)
    joblib.dump(final_rf, os.path.join(MODELS_DIR, 'rf_model.pkl'))

    final_xgb = xgb.XGBClassifier(eval_metric='logloss', random_state=RANDOM_SEED).fit(X_stat_all_s, y)
    joblib.dump(final_xgb, os.path.join(MODELS_DIR, 'xgb_model.pkl'))

    # CNN retrained on all padded sequences.
    cnn_final = Simple1DCNN(len(FEATURE_NAMES), FIXED_LEN).to(DEVICE)
    cnn_optim = torch.optim.Adam(cnn_final.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    cnn_crit = nn.BCELoss()
    cnn_all_loader = DataLoader(
        TensorDataset(torch.tensor(X_cnn), torch.tensor(y, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    for _ in range(NUM_EPOCHS_FINAL):
        cnn_final.train()
        for xb, yb in cnn_all_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            cnn_optim.zero_grad()
            loss = cnn_crit(cnn_final(xb), yb)
            loss.backward()
            cnn_optim.step()
    torch.save(cnn_final.state_dict(), os.path.join(MODELS_DIR, 'cnn_final.pth'))

    # LSTM retrained on all raw sequences with a single global normalisation.
    raw_mean_all = np.mean(np.vstack(X_raw), axis=0)
    raw_std_all = np.std(np.vstack(X_raw), axis=0) + 1e-8
    X_raw_all_norm = [(seq - raw_mean_all) / raw_std_all for seq in X_raw]
    lstm_all_loader = DataLoader(
        LSTMSequenceDataset(X_raw_all_norm, y.tolist()),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=lstm_collate_fn,
    )
    lstm_final = AimDetectorLSTM(len(FEATURE_NAMES)).to(DEVICE)
    lstm_optim = torch.optim.Adam(lstm_final.parameters(), lr=LEARNING_RATE)
    lstm_crit = nn.BCELoss()
    for _ in range(NUM_EPOCHS_FINAL):
        lstm_final.train()
        for x, mask, yb in lstm_all_loader:
            x, mask, yb = x.to(DEVICE), mask.to(DEVICE), yb.to(DEVICE)
            lstm_optim.zero_grad()
            loss = lstm_crit(lstm_final(x, mask), yb)
            loss.backward()
            lstm_optim.step()
    torch.save(lstm_final.state_dict(), os.path.join(MODELS_DIR, 'lstm_final.pth'))
    np.savez(os.path.join(MODELS_DIR, 'lstm_norm.npz'), mean=raw_mean_all, std=raw_std_all)

    print(f"All models and scaler saved to '{MODELS_DIR}/' directory.")


if __name__ == "__main__":
    main()
