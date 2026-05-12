# utils.py
import ast
import math
import numpy as np
from collections import defaultdict

# Features without absolute positions (x, y, z) to avoid map bias.
FEATURE_NAMES = [
    'yaw', 'pitch',
    'dx', 'dy', 'dz',
    'dyaw', 'dpitch',
    'shooting',
    'min_angle_to_enemy',
    'aim_correction_delta'
]

def load_events(json_path):
    """Load file with Python dict literals (one per line) into a list of dicts."""
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

def extract_statistical_features(seq):
    """
    seq: np.array of shape (T, F) - variable length.
    F = len(FEATURE_NAMES) = 10.
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

    # Temporal features – indices reflect the new order (no x,y,z)
    # dyaw index = 5, dpitch index = 6
    if seq.shape[0] >= 2:
        dyaw_series = seq[1:, 5]   # shift by 1
        dyaw_lag = seq[:-1, 5]
        ac_dyaw = np.corrcoef(dyaw_lag, dyaw_series)[0, 1] if np.std(dyaw_lag) > 0 else 0.0

        dpitch_series = seq[1:, 6]
        dpitch_lag = seq[:-1, 6]
        ac_dpitch = np.corrcoef(dpitch_lag, dpitch_series)[0, 1] if np.std(dpitch_lag) > 0 else 0.0
    else:
        ac_dyaw = ac_dpitch = 0.0

    # shooting = index 7
    shooting = seq[:, 7]
    frac_shooting = np.mean(shooting)
    shooting_events = np.sum(np.diff(shooting) > 0.5)

    # dx,dy,dz = indices 2,3,4
    total_dist = np.sum(np.sqrt(seq[:, 2]**2 + seq[:, 3]**2 + seq[:, 4]**2))
    duration = len(seq) * 0.04   # approx 40ms per tick
    avg_speed = total_dist / duration if duration > 0 else 0.0

    # aim_correction_delta = index 9
    max_aim_corr = np.max(seq[:, 9])
    aim_on_target = np.mean(seq[:, 9] < 10.0)

    # Concatenate all features (7*10 = 70 statistics + 7 extra = 77 features)
    features = np.concatenate([
        mean, std, min_val, max_val, median, p5, p95,
        [ac_dyaw, ac_dpitch, frac_shooting, shooting_events, avg_speed, max_aim_corr, aim_on_target]
    ])
    return features

def build_full_sequence(player_events, all_players_events, is_cheater):
    if len(player_events) < 10: return None, None
    player_events = add_derived_features(player_events)
    player_events = compute_aim_correction_features(player_events, all_players_events)
    seq = []
    for ev in player_events:
        vec = [float(ev.get(feat, 0.0)) for feat in FEATURE_NAMES]
        seq.append(vec)
    return np.array(seq, dtype=np.float32), 1 if is_cheater else 0