"""Sliding-window per-match analysis on three held-out matches.

For each of the toggling, full-cheat, and full-clean matches this script
runs the saved classical models (plus the simple-average ensemble) and the
saved deep models over fixed-width windows, and saves one figure per match
and model family.
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch

from model_comparison import AimDetectorLSTM, Simple1DCNN
from utils import (
    FEATURE_NAMES,
    add_derived_features,
    build_player_timelines,
    compute_aim_correction_features,
    extract_statistical_features,
    load_events,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# (demo_path, cheater_cn, match_label)
MATCHES = [
    ("data/demo20260510_1926_local_ac_scaffold_10min_DM.json", 2, "toggling"),
    ("data/demo20260322_2136_local_ac_scaffold_10min_DM.json", 0, "full_cheat"),
    ("data/demo20260309_2147_local_ac_desert3_10min_DM.json", 0, "full_clean"),
]

WINDOW_MS = 30_000
STRIDE_MS = 5_000
MIN_WINDOW_EVENTS = 10
DECISION_THRESHOLD = 0.5
FIXED_LEN = 2000  # Must match the CNN input length used during training.

MODEL_DIR = "models"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
LR_PATH = os.path.join(MODEL_DIR, "lr_model.pkl")
RF_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
XGB_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
CNN_PATH = os.path.join(MODEL_DIR, "cnn_final.pth")
LSTM_PATH = os.path.join(MODEL_DIR, "lstm_final.pth")
LSTM_NORM_PATH = os.path.join(MODEL_DIR, "lstm_norm.npz")

OUTPUT_DIR = "results"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Optional toggle markers (gametime in ms). Set when known for the toggling match.
TOGGLE_ON = None
TOGGLE_OFF = None


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def load_models():
    scaler = joblib.load(SCALER_PATH)
    lr_model = joblib.load(LR_PATH)
    rf_model = joblib.load(RF_PATH)
    xgb_model = joblib.load(XGB_PATH)

    cnn_model = Simple1DCNN(len(FEATURE_NAMES), FIXED_LEN).to(DEVICE)
    cnn_model.load_state_dict(torch.load(CNN_PATH, map_location=DEVICE))
    cnn_model.eval()

    lstm_model = AimDetectorLSTM(len(FEATURE_NAMES)).to(DEVICE)
    lstm_model.load_state_dict(torch.load(LSTM_PATH, map_location=DEVICE))
    lstm_model.eval()

    lstm_norm = np.load(LSTM_NORM_PATH)
    return {
        'scaler': scaler,
        'lr': lr_model,
        'rf': rf_model,
        'xgb': xgb_model,
        'cnn': cnn_model,
        'lstm': lstm_model,
        'lstm_mean': lstm_norm['mean'],
        'lstm_std': lstm_norm['std'],
    }


# -----------------------------------------------------------------------------
# Shared per-match preparation
# -----------------------------------------------------------------------------
def _prepare_player_events(demo_path, player_cn):
    events = load_events(demo_path)
    players_events = build_player_timelines(events)
    if player_cn not in players_events:
        raise ValueError(f"Client {player_cn} not found in {demo_path}.")
    player_ev = players_events[player_cn]
    player_ev = add_derived_features(player_ev)
    player_ev = compute_aim_correction_features(player_ev, players_events)
    return player_ev


def _iterate_windows(player_ev):
    """Yield (window_events, window_start_gametime) for each valid window."""
    start = player_ev[0]['gametime']
    end = player_ev[-1]['gametime']
    cur = start
    while cur + WINDOW_MS <= end:
        cur_end = cur + WINDOW_MS
        window = [e for e in player_ev if cur <= e['gametime'] < cur_end]
        if len(window) >= MIN_WINDOW_EVENTS:
            yield window, cur
        cur += STRIDE_MS


def _window_to_array(window):
    return np.array(
        [[float(e.get(f, 0.0)) for f in FEATURE_NAMES] for e in window],
        dtype=np.float32,
    )


# -----------------------------------------------------------------------------
# Classical sliding window
# -----------------------------------------------------------------------------
def process_match_classical(demo_path, player_cn, models):
    player_ev = _prepare_player_events(demo_path, player_cn)
    start = player_ev[0]['gametime']
    lr_p, rf_p, xgb_p, avg_p, times_min = [], [], [], [], []

    for window, cur in _iterate_windows(player_ev):
        seq = _window_to_array(window)
        feat = extract_statistical_features(seq).reshape(1, -1)
        feat_scaled = models['scaler'].transform(feat)
        lr_prob = models['lr'].predict_proba(feat_scaled)[0, 1]
        rf_prob = models['rf'].predict_proba(feat_scaled)[0, 1]
        xgb_prob = models['xgb'].predict_proba(feat_scaled)[0, 1]
        lr_p.append(lr_prob)
        rf_p.append(rf_prob)
        xgb_p.append(xgb_prob)
        avg_p.append((lr_prob + rf_prob + xgb_prob) / 3.0)
        times_min.append((cur - start) / 60_000.0)

    return times_min, lr_p, rf_p, xgb_p, avg_p, start


# -----------------------------------------------------------------------------
# Deep sliding window
# -----------------------------------------------------------------------------
def process_match_deep(demo_path, player_cn, models):
    player_ev = _prepare_player_events(demo_path, player_cn)
    start = player_ev[0]['gametime']
    cnn_p, lstm_p, times_min = [], [], []

    for window, cur in _iterate_windows(player_ev):
        raw_seq = _window_to_array(window)
        T = raw_seq.shape[0]

        # CNN: pad or right-crop to FIXED_LEN, channels-first.
        if T <= FIXED_LEN:
            padded = np.zeros((FIXED_LEN, len(FEATURE_NAMES)), dtype=np.float32)
            padded[:T] = raw_seq
        else:
            padded = raw_seq[-FIXED_LEN:]
        cnn_input = torch.tensor(padded.T).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            cnn_p.append(models['cnn'](cnn_input).item())

        # LSTM: variable length with global normalisation.
        norm_seq = (raw_seq - models['lstm_mean']) / models['lstm_std']
        lstm_input = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        mask = torch.ones(1, T, dtype=torch.bool).to(DEVICE)
        with torch.no_grad():
            lstm_p.append(models['lstm'](lstm_input, mask).item())

        times_min.append((cur - start) / 60_000.0)

    print(f"CNN first 5 probs: {cnn_p[:5]}")
    print(f"LSTM first 5 probs: {lstm_p[:5]}")
    return times_min, cnn_p, lstm_p, start


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def _annotate_toggles(label, start_time):
    if label != "toggling":
        return
    if TOGGLE_ON is not None:
        plt.axvline((TOGGLE_ON - start_time) / 60_000.0,
                    color='green', linestyle=':', label='Toggle ON')
    if TOGGLE_OFF is not None:
        plt.axvline((TOGGLE_OFF - start_time) / 60_000.0,
                    color='orange', linestyle=':', label='Toggle OFF')


def _save_plot(label, family, out_dir):
    plt.xlabel('Time (minutes)')
    plt.ylabel('Cheating probability')
    plt.title(f'{family} – {label.replace("_", " ")} match')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{label}_{family.lower().split()[0]}.png"))
    plt.show()


def plot_classical(times, lr_p, rf_p, xgb_p, avg_p, label, start_time, out_dir):
    plt.figure(figsize=(14, 7))
    plt.plot(times, lr_p, label='Logistic Regression', alpha=0.8)
    plt.plot(times, rf_p, label='Random Forest', alpha=0.8)
    plt.plot(times, xgb_p, label='XGBoost', alpha=0.8)
    plt.plot(times, avg_p, label='Simple Average', linewidth=2.5, color='black')
    plt.axhline(DECISION_THRESHOLD, color='red', linestyle='--', label='Decision boundary')
    _annotate_toggles(label, start_time)
    _save_plot(label, 'Classical models', out_dir)


def plot_deep(times, cnn_p, lstm_p, label, start_time, out_dir):
    plt.figure(figsize=(14, 7))
    # CNN with dashed line plus markers so it stays visible when the curve is flat.
    plt.plot(times, cnn_p, 'o--', label='1D-CNN', color='cyan', markersize=4, alpha=0.9)
    plt.plot(times, lstm_p, label='LSTM', color='magenta', alpha=0.8)
    plt.axhline(DECISION_THRESHOLD, color='red', linestyle='--', label='Decision boundary')
    _annotate_toggles(label, start_time)
    _save_plot(label, 'Deep models', out_dir)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main():
    models = load_models()
    print("All models loaded.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for demo_path, cn, label in MATCHES:
        print(f"Processing {label} match: {demo_path} (player cn={cn})")

        times, lr_p, rf_p, xgb_p, avg_p, start_time = process_match_classical(
            demo_path, cn, models)
        plot_classical(times, lr_p, rf_p, xgb_p, avg_p, label, start_time, OUTPUT_DIR)

        times, cnn_p, lstm_p, start_time = process_match_deep(demo_path, cn, models)
        plot_deep(times, cnn_p, lstm_p, label, start_time, OUTPUT_DIR)

    print(f"All plots saved in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
