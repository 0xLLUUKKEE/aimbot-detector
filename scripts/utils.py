# utils.py
import ast
import math
import numpy as np
from collections import defaultdict

FEATURE_NAMES = [
    'x', 'y', 'z', 'yaw', 'pitch',
    'dx', 'dy', 'dz', 'dyaw', 'dpitch', 'shooting',
    'min_angle_to_enemy', 'aim_correction_delta'
]

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