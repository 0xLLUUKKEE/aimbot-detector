# three_match_analysis.py
"""
Sliding‑window analysis on three held‑out matches:
  - toggling match
  - full cheat match
  - full clean match
Uses the final classical models + simple average ensemble.
Saves one figure per match.
"""

import os, joblib, numpy as np, matplotlib.pyplot as plt, torch

from utils import *

# ------------------------------
# CONFIG – list of (demo_path, cheater_cn, match_label)
# ------------------------------

MATCHES = [
    ("data/demo20260510_1926_local_ac_scaffold_10min_DM.json", 2, "toggling"),
    #("data/demo20260309_2217_local_ac_lainio_10min_DM.json", 3, "toggling"),
    ("demo20260322_2136_local_ac_scaffold_10min_DM.json", 0, "full_cheat"),
    ("data/demo20260309_2147_local_ac_desert3_10min_DM.json", 0, "full_clean"),   
]
CHEATER_CN = 1                              # client number of the player who toggles
WINDOW_MS = 30_000
STRIDE_MS = 5_000

# Paths to saved classical models & scaler
MODEL_DIR = "models"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
LR_PATH     = os.path.join(MODEL_DIR, "lr_model.pkl")
RF_PATH     = os.path.join(MODEL_DIR, "rf_model.pkl")
XGB_PATH    = os.path.join(MODEL_DIR, "xgb_model.pkl")

# Paths to saved deep models & normalisation
CNN_PATH    = os.path.join(MODEL_DIR, "cnn_final.pth")
LSTM_PATH   = os.path.join(MODEL_DIR, "lstm_final.pth")
LSTM_NORM   = os.path.join(MODEL_DIR, "lstm_norm.npz")

FIXED_LEN = 2000   # for CNN
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# (Optional) known toggle times (gametime in ms)
TOGGLE_ON  = None
TOGGLE_OFF = None

# -----------------------------------------------------------------------------
# Load classical models
# -----------------------------------------------------------------------------
scaler   = joblib.load(SCALER_PATH)
lr_model = joblib.load(LR_PATH)
rf_model = joblib.load(RF_PATH)
xgb_model = joblib.load(XGB_PATH)

# -----------------------------------------------------------------------------
# Load deep models 
# -----------------------------------------------------------------------------
# Import required classes (copy from model_comparison.py if not already in utils)
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

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
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers>1 else 0,
                            bidirectional=False)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, mask):
        lengths = mask.sum(dim=1).cpu()
        x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(x_packed)
        return self.sigmoid(self.fc(hidden[-1])).squeeze(-1)

cnn_model = Simple1DCNN(len(FEATURE_NAMES), FIXED_LEN).to(DEVICE)
cnn_model.load_state_dict(torch.load(CNN_PATH, map_location=DEVICE))
cnn_model.eval()

lstm_model = AimDetectorLSTM(len(FEATURE_NAMES)).to(DEVICE)
lstm_model.load_state_dict(torch.load(LSTM_PATH, map_location=DEVICE))
lstm_model.eval()

# Load LSTM normalisation statistics
lstm_norm = np.load(LSTM_NORM)
lstm_mean = lstm_norm['mean']
lstm_std  = lstm_norm['std']

print("All models loaded.")

# -----------------------------------------------------------------------------
# Helper: process one match for classical models
# -----------------------------------------------------------------------------
def process_match_classical(demo_path, player_cn):
    events = load_events(demo_path)
    players_events = build_player_timelines(events)
    if player_cn not in players_events:
        raise ValueError(f"Client {player_cn} not found in {demo_path}.")
    player_ev = players_events[player_cn]
    all_players = {cn: evs for cn, evs in players_events.items()}
    player_ev = add_derived_features(player_ev)
    player_ev = compute_aim_correction_features(player_ev, all_players)

    start = player_ev[0]['gametime']
    end   = player_ev[-1]['gametime']
    lr_p, rf_p, xgb_p, avg_p = [], [], [], []
    times_min = []
    cur = start
    while cur + WINDOW_MS <= end:
        cur_end = cur + WINDOW_MS
        window = [e for e in player_ev if cur <= e['gametime'] < cur_end]
        if len(window) < 10:
            cur += STRIDE_MS
            continue
        seq = np.array([[float(e.get(f, 0.0)) for f in FEATURE_NAMES] for e in window], dtype=np.float32)
        feat = extract_statistical_features(seq).reshape(1, -1)
        feat_scaled = scaler.transform(feat)
        lr_p.append(lr_model.predict_proba(feat_scaled)[0,1])
        rf_p.append(rf_model.predict_proba(feat_scaled)[0,1])
        xgb_p.append(xgb_model.predict_proba(feat_scaled)[0,1])
        avg_p.append((lr_p[-1]+rf_p[-1]+xgb_p[-1])/3)
        times_min.append((cur - start) / 60_000.0)
        cur += STRIDE_MS
    return times_min, lr_p, rf_p, xgb_p, avg_p, start

# -----------------------------------------------------------------------------
# Helper: process one match for deep models (returns probs for CNN and LSTM)
# -----------------------------------------------------------------------------
def process_match_deep(demo_path, player_cn):
    events = load_events(demo_path)
    players_events = build_player_timelines(events)
    if player_cn not in players_events:
        raise ValueError(f"Client {player_cn} not found in {demo_path}.")
    player_ev = players_events[player_cn]
    all_players = {cn: evs for cn, evs in players_events.items()}
    player_ev = add_derived_features(player_ev)
    player_ev = compute_aim_correction_features(player_ev, all_players)

    start = player_ev[0]['gametime']
    end   = player_ev[-1]['gametime']
    cnn_p, lstm_p = [], []
    times_min = []
    cur = start
    while cur + WINDOW_MS <= end:
        cur_end = cur + WINDOW_MS
        window = [e for e in player_ev if cur <= e['gametime'] < cur_end]
        if len(window) < 10:
            cur += STRIDE_MS
            continue
        raw_seq = np.array([[float(e.get(f, 0.0)) for f in FEATURE_NAMES] for e in window], dtype=np.float32)

        # CNN
        T = raw_seq.shape[0]
        if T <= FIXED_LEN:
            padded = np.zeros((FIXED_LEN, len(FEATURE_NAMES)), dtype=np.float32)
            padded[:T] = raw_seq
        else:
            padded = raw_seq[-FIXED_LEN:]
        cnn_input = torch.tensor(padded.T).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            cnn_prob = cnn_model(cnn_input).item()
        cnn_p.append(cnn_prob)

        # LSTM
        norm_seq = (raw_seq - lstm_mean) / lstm_std
        lstm_input = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        mask = torch.ones(1, T, dtype=torch.bool).to(DEVICE)
        with torch.no_grad():
            lstm_prob = lstm_model(lstm_input, mask).item()
        lstm_p.append(lstm_prob)

        times_min.append((cur - start) / 60_000.0)
        cur += STRIDE_MS

    # ----- diagnostic print (first few values) -----
    print(f"CNN first 5 probs: {cnn_p[:5]}")
    print(f"LSTM first 5 probs: {lstm_p[:5]}")
    return times_min, cnn_p, lstm_p, start

# -----------------------------------------------------------------------------
# Produce classical and deep plots for each match
# -----------------------------------------------------------------------------
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

for demo_path, cn, label in MATCHES:
    print(f"Processing {label} match: {demo_path} (player cn={cn})")

    # ----- Classical -----
    times, lr_p, rf_p, xgb_p, avg_p, start_time = process_match_classical(demo_path, cn)
    plt.figure(figsize=(14, 7))
    plt.plot(times, lr_p, label='Logistic Regression', alpha=0.8)
    plt.plot(times, rf_p, label='Random Forest', alpha=0.8)
    plt.plot(times, xgb_p, label='XGBoost', alpha=0.8)
    plt.plot(times, avg_p, label='Simple Average', linewidth=2.5, color='black')
    plt.axhline(0.5, color='red', linestyle='--', label='Decision boundary')
    if label == "toggling" and TOGGLE_ON is not None:
        plt.axvline((TOGGLE_ON - start_time) / 60_000.0, color='green', linestyle=':', label='Toggle ON')
    if label == "toggling" and TOGGLE_OFF is not None:
        plt.axvline((TOGGLE_OFF - start_time) / 60_000.0, color='orange', linestyle=':', label='Toggle OFF')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Cheating probability')
    plt.title(f'Classical models – {label.replace("_"," ")} match')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{label}_classical.png"))
    plt.show()

    # ----- Deep -----
    times, cnn_p, lstm_p, _ = process_match_deep(demo_path, cn)
    plt.figure(figsize=(14, 7))
    # Plot CNN with a dashed line and larger markers so it cannot be missed
    plt.plot(times, cnn_p, 'o--', label='1D‑CNN', color='cyan', markersize=4, alpha=0.9)
    plt.plot(times, lstm_p, label='LSTM', color='magenta', alpha=0.8)
    plt.axhline(0.5, color='red', linestyle='--', label='Decision boundary')
    if label == "toggling" and TOGGLE_ON is not None:
        plt.axvline((TOGGLE_ON - start_time) / 60_000.0, color='green', linestyle=':', label='Toggle ON')
    if label == "toggling" and TOGGLE_OFF is not None:
        plt.axvline((TOGGLE_OFF - start_time) / 60_000.0, color='orange', linestyle=':', label='Toggle OFF')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Cheating probability')
    plt.title(f'Deep models – {label.replace("_"," ")} match')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{label}_deep.png"))
    plt.show()

print("All plots saved in results/")