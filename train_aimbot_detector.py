# -*- coding: utf-8 -*-
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import precision_score, recall_score, f1_score
import random
import matplotlib.pyplot as plt
import ast
# -----------------------------------------------------------------------------
# Configuration – adjust these paths and lists
# -----------------------------------------------------------------------------
DEMO_FILES = [
    # list of (json_path, cheat_dict) where cheat_dict maps cn -> bool (cheater or not)
    # Example:
    # ("match1.json", {0: True, 1: False, 2: False, 3: False, 4: False, 5: False}),
    # ("match2.json", {0: False, 1: False, 2: False, 3: False, 4: False, 5: False}),  # clean match
    # etc.
    ("demo20260306_2245_local_ac_desert3_10min_DM.json", {0: False, 1: False, 2: False, 3: False, 4: False, 5: False}),  # clean match
    ("demo20260306_2316_local_ac_lainio_10min_DM.json", {0: False, 1: False, 2: False, 3: False, 4: False, 5: False}),  # clean match


    ("demo20260322_2024_local_ac_scaffold_10min_DM.json", {0: True, 1: True, 2: True}),  # full cheating match match
    ("demo20260322_2042_local_ac_desert3_10min_DM.json", {0: True, 1: True, 2: True}),  # full cheating match match
    ("demo20260322_2052_local_ac_scaffold_10min_DM.json", {0: True, 1: True, 2: True}),  # full cheating match match
    ("demo20260322_2102_local_ac_desert_10min_DM.json", {0: True, 1: True, 2: True}),  # full cheating match match
    ("demo20260322_2114_local_ac_elevation_10min_DM.json", {0: True, 1: True, 2: True}),  # full cheating match match
    ("demo20260322_2126_local_ac_desert3_10min_DM.json", {0: True, 1: True, 2: True}),  # full cheating match match
    ("demo20260322_2136_local_ac_scaffold_10min_DM.json", {0: True, 1: True, 2: True}),  # full cheating match match
    ("demo20260322_2146_local_ac_desert_10min_DM.json", {0: True, 1: True, 2: True}),  # full cheating match match


    ("demo20260325_1513_local_ac_desert3_10min_DM.json", {0: True, 1: False, 2: False, 3: False}),  # mixed match
    ("demo20260325_1523_local_ac_scaffold_10min_DM.json", {0: False, 1: False, 2: False , 3: True}),  # mixed match
    ("demo20260325_1533_local_ac_desert_10min_DM.json", {0: False, 1: False, 2: True, 3: False}),  # mixed match
    ("demo20260325_1544_local_ac_lainio_10min_DM.json", {0: False, 1: False, 2: False, 3: True})  # mixed match
]

WINDOW_MS = 30000       # 30 seconds
STRIDE_MS = 15000       # 15 seconds overlap
FEATURE_NAMES = ['x', 'y', 'z', 'yaw', 'pitch', 'dx', 'dy', 'dz', 'dyaw', 'dpitch', 'shooting']

# Training hyperparameters
BATCH_SIZE = 32
HIDDEN_SIZE = 64
NUM_LAYERS = 1
LEARNING_RATE = 0.001
NUM_EPOCHS = 20

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def load_events(json_path):
    """Load file with Python dict literals (one per line) into a list of dicts."""
    events = []
    with open(json_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Use ast.literal_eval to safely parse Python literal dicts
            events.append(ast.literal_eval(line))
    return events

def build_player_timelines(events):
    """Group position events by client number, sort by gametime."""
    players = {}
    for ev in events:
        if ev.get('type') == 'position':
            cn = ev['cn']
            players.setdefault(cn, []).append(ev)
    for cn in players:
        players[cn].sort(key=lambda e: e['gametime'])
    return players

def add_derived_features(events):
    """Add dx, dy, dz, dyaw, dpitch to each event (in‑place)."""
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

        prev = ev
    return events

def create_windows(player_events, is_cheater, window_ms=WINDOW_MS, stride_ms=STRIDE_MS):
    """Return list of (feature_matrix, label) for windows."""
    if not player_events:
        return []
    windows = []
    start_time = player_events[0]['gametime']
    end_time = player_events[-1]['gametime']

    cur_start = start_time
    while cur_start + window_ms <= end_time:
        cur_end = cur_start + window_ms
        window_events = [ev for ev in player_events if cur_start <= ev['gametime'] < cur_end]
        if window_events:
            seq = []
            for ev in window_events:
                vec = [ev[name] for name in FEATURE_NAMES]
                seq.append(vec)
            windows.append((np.array(seq, dtype=np.float32), 1 if is_cheater else 0))
        cur_start += stride_ms
    return windows

# -----------------------------------------------------------------------------
# 1. Process all matches to collect windows and labels
# -----------------------------------------------------------------------------
print("Loading and processing demos...")
all_windows = []
all_labels = []
match_indices = []

for idx, (json_path, cheat_dict) in enumerate(DEMO_FILES):
    events = load_events(json_path)
    players = build_player_timelines(events)
    for cn, ev_list in players.items():
        ev_list = add_derived_features(ev_list)
        is_cheater = cheat_dict.get(cn, False)
        windows = create_windows(ev_list, is_cheater)
        for seq, lbl in windows:
            all_windows.append(seq)
            all_labels.append(lbl)
            match_indices.append(idx)

print(f"Total windows collected: {len(all_windows)}")

# -----------------------------------------------------------------------------
# 2. Normalisation
# -----------------------------------------------------------------------------
print("Normalising features...")
all_timesteps = np.vstack(all_windows)
mean = all_timesteps.mean(axis=0)
std = all_timesteps.std(axis=0) + 1e-8
norm_windows = [(seq - mean) / std for seq in all_windows]

# -----------------------------------------------------------------------------
# 3. Split by match (no leakage)
# -----------------------------------------------------------------------------
unique_matches = list(set(match_indices))
random.shuffle(unique_matches)
train_matches = unique_matches[:int(0.7 * len(unique_matches))]
val_matches   = unique_matches[int(0.7 * len(unique_matches)):int(0.85 * len(unique_matches))]
test_matches  = unique_matches[int(0.85 * len(unique_matches)):]

train_idx = [i for i, m in enumerate(match_indices) if m in train_matches]
val_idx   = [i for i, m in enumerate(match_indices) if m in val_matches]
test_idx  = [i for i, m in enumerate(match_indices) if m in test_matches]

X_train = [norm_windows[i] for i in train_idx]
y_train = [all_labels[i] for i in train_idx]
X_val   = [norm_windows[i] for i in val_idx]
y_val   = [all_labels[i] for i in val_idx]
X_test  = [norm_windows[i] for i in test_idx]
y_test  = [all_labels[i] for i in test_idx]

print(f"Train: {len(X_train)} windows, Val: {len(X_val)}, Test: {len(X_test)}")

# -----------------------------------------------------------------------------
# 4. PyTorch Dataset and DataLoader
# -----------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class CheatDataset(Dataset):
    def __init__(self, windows, labels):
        self.windows = windows
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        seq = torch.tensor(self.windows[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return seq, label

def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True)
    labels = torch.stack(labels)
    return sequences_padded, labels

train_dataset = CheatDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn)
val_dataset   = CheatDataset(X_val, y_val)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_fn)
test_dataset  = CheatDataset(X_test, y_test)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_fn)

# -----------------------------------------------------------------------------
# 5. Model definition
# -----------------------------------------------------------------------------
class AimDetector(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        out = self.sigmoid(out).squeeze()
        return out

model = AimDetector(input_size=len(FEATURE_NAMES)).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -----------------------------------------------------------------------------
# 6. Training loop
# -----------------------------------------------------------------------------
print("Starting training...")
for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_x.size(0)
        preds = (outputs > 0.5).float()
        train_correct += (preds == batch_y).sum().item()
        train_total += batch_y.size(0)
    train_loss /= train_total
    train_acc = train_correct / train_total

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_x.size(0)
            preds = (outputs > 0.5).float()
            val_correct += (preds == batch_y).sum().item()
            val_total += batch_y.size(0)
    val_loss /= val_total
    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - "
          f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
          f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

# -----------------------------------------------------------------------------
# 7. Test evaluation
# -----------------------------------------------------------------------------
model.eval()
test_correct = 0
test_total = 0
all_preds = []
all_true = []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        preds = (outputs > 0.5).float()
        test_correct += (preds == batch_y).sum().item()
        test_total += batch_y.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_true.extend(batch_y.cpu().numpy())

test_acc = test_correct / test_total
print(f"Test accuracy: {test_acc:.4f}")
print(f"Precision: {precision_score(all_true, all_preds):.4f}")
print(f"Recall:    {recall_score(all_true, all_preds):.4f}")
print(f"F1 score:  {f1_score(all_true, all_preds):.4f}")

# -----------------------------------------------------------------------------
# 8. (Optional) Plot predictions on a toggling match
# -----------------------------------------------------------------------------
def plot_toggling_match(json_path, cheater_cn, model, device, mean, std):
    events = load_events(json_path)
    players = build_player_timelines(events)
    if cheater_cn not in players:
        print(f"Cheater cn {cheater_cn} not found in demo.")
        return
    player_events = players[cheater_cn]
    player_events = add_derived_features(player_events)

    # Use small stride (1 second) to get dense probability plot
    windows = create_windows(player_events, False, window_ms=30000, stride_ms=1000)
    if not windows:
        print("No windows created.")
        return

    seqs = [seq for seq, _ in windows]
    norm_seqs = [(seq - mean) / std for seq in seqs]

    model.eval()
    probs = []
    with torch.no_grad():
        for seq in norm_seqs:
            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            prob = model(x).item()
            probs.append(prob)

    plt.figure(figsize=(10, 5))
    plt.plot(probs)
    plt.xlabel('Window index (approx 1s stride)')
    plt.ylabel('Cheating probability')
    plt.title('Model output on toggling match')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()

# Uncomment to plot a toggling match (adjust path and cheater cn)
# plot_toggling_match("path_to_toggling_match.json", cheater_cn=0,
#                     model=model, device=device, mean=mean, std=std)

print("Done.")
train_cheat = sum([all_labels[i] for i in train_idx])
val_cheat = sum([all_labels[i] for i in val_idx])
test_cheat = sum([all_labels[i] for i in test_idx])
print(f"Train: {len(train_idx)} windows, {train_cheat} cheat ({train_cheat/len(train_idx):.2%})")
print(f"Val:   {len(val_idx)} windows, {val_cheat} cheat ({val_cheat/len(val_idx):.2%})")
print(f"Test:  {len(test_idx)} windows, {test_cheat} cheat ({test_cheat/len(test_idx):.2%})")
