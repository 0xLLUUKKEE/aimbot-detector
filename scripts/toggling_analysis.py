# toggling_all_models.py
"""
Sliding‑window analysis of a toggling match using ALL trained models.
Compares Logistic Regression, Random Forest, XGBoost, and their
simple average ensemble. Produces a single overlaid probability plot.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from utils import *

#TOGGLING_DEMO_PATH = "demo20260423_2206_local_ac_elevation_10min_DM.json"   # your toggling match
# -----------------------------------------------------------------------------
# Configuration 
# -----------------------------------------------------------------------------
TOGGLING_DEMO_PATH = "data/demo20260309_2217_local_ac_lainio_10min_DM.json"   # toggling match file
#TOGGLING_DEMO_PATH = "demo20260322_2136_local_ac_scaffold_10min_DM.json"   # Full cheat match
CHEATER_CN = 1                              # client number of the player who toggles
WINDOW_MS = 30000                            # 30 seconds
STRIDE_MS = 5000                             # 5 seconds for denser curve

# Paths to saved models
MODEL_DIR = "models"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
LR_PATH     = os.path.join(MODEL_DIR, "lr_model.pkl")
RF_PATH     = os.path.join(MODEL_DIR, "rf_model.pkl")
XGB_PATH    = os.path.join(MODEL_DIR, "xgb_model.pkl")

# (Optional) known toggle times in gametime (ms) – set to None if unknown
TOGGLE_ON  = None   # e.g., 120000
TOGGLE_OFF = None   # e.g., 360000

# -----------------------------------------------------------------------------
# Load scaler and all models
# -----------------------------------------------------------------------------
print("Loading models...")
scaler = joblib.load(SCALER_PATH)

lr_model = joblib.load(LR_PATH)
print("  Loaded Logistic Regression")

rf_model = joblib.load(RF_PATH)
print("  Loaded Random Forest")

xgb_model = joblib.load(XGB_PATH)
print("  Loaded XGBoost")

# -----------------------------------------------------------------------------
# 1. Load demo and extract the cheater's timeline
# -----------------------------------------------------------------------------
print(f"Loading toggling match: {TOGGLING_DEMO_PATH}")
events = load_events(TOGGLING_DEMO_PATH)
players_events = build_player_timelines(events)

if CHEATER_CN not in players_events:
    raise ValueError(f"Client {CHEATER_CN} not found in demo. Available: {list(players_events.keys())}")

cheater_events = players_events[CHEATER_CN]

# Add derived features and aim‑correction features using all players' timelines
all_players = {cn: evs for cn, evs in players_events.items()}
cheater_events = add_derived_features(cheater_events)
cheater_events = compute_aim_correction_features(cheater_events, all_players)

# -----------------------------------------------------------------------------
# 2. Sliding windows and predictions for each model
# -----------------------------------------------------------------------------
start_time = cheater_events[0]['gametime']
end_time   = cheater_events[-1]['gametime']

window_starts = []
lr_probs = []
rf_probs = []
xgb_probs = []
avg_probs = []

cur = start_time
while cur + WINDOW_MS <= end_time:
    cur_end = cur + WINDOW_MS
    window_events = [ev for ev in cheater_events if cur <= ev['gametime'] < cur_end]

    if len(window_events) < 10:
        cur += STRIDE_MS
        continue

    # Build feature matrix for this window
    seq = np.array([[float(ev.get(feat, 0.0)) for feat in FEATURE_NAMES] for ev in window_events],
                   dtype=np.float32)

    # Extract statistical features
    feat_vec = extract_statistical_features(seq).reshape(1, -1)

    # Scale using the same scaler as training
    feat_scaled = scaler.transform(feat_vec)

    # Predict cheating probability for each model
    lr_prob = lr_model.predict_proba(feat_scaled)[0, 1]
    rf_prob = rf_model.predict_proba(feat_scaled)[0, 1]
    xgb_prob = xgb_model.predict_proba(feat_scaled)[0, 1]

    # Simple average of the three probabilities
    avg_prob = (lr_prob + rf_prob + xgb_prob) / 3.0

    lr_probs.append(lr_prob)
    rf_probs.append(rf_prob)
    xgb_probs.append(xgb_prob)
    avg_probs.append(avg_prob)

    window_starts.append(cur)
    cur += STRIDE_MS

# -----------------------------------------------------------------------------
# 3. Plot all curves together
# -----------------------------------------------------------------------------
if not window_starts:
    print("No windows could be created. Check window size and demo length.")
else:
    times_sec = [(t - start_time) / 1000.0 for t in window_starts]
    times_min = [(t - start_time) / 60000.0 for t in window_starts]
    plt.figure(figsize=(14, 7))
    plt.plot(times_min, lr_probs, label='Logistic Regression', alpha=0.8)
    plt.plot(times_min, rf_probs, label='Random Forest', alpha=0.8)
    plt.plot(times_min, xgb_probs, label='XGBoost', alpha=0.8)
    plt.plot(times_min, avg_probs, label='Simple Average', linewidth=2.5, color='black')

    plt.axhline(0.5, color='red', linestyle='--', label='Decision boundary (0.5)')

    # Optional: mark known toggle events
    if TOGGLE_ON is not None:
        plt.axvline((TOGGLE_ON - start_time) / 1000.0, color='green', linestyle=':', label='Toggle ON')
    if TOGGLE_OFF is not None:
        plt.axvline((TOGGLE_OFF - start_time) / 1000.0, color='orange', linestyle=':', label='Toggle OFF')

    plt.xlabel('Time (minutes from match start)')
    plt.ylabel('Cheating probability')
    plt.title('Model comparison on a toggling match')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('toggling_comparison.png')
    plt.show()
    print("Figure saved as toggling_comparison.png")