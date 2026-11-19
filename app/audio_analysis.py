"""
Adaptive Tone Engine — Audio Analysis Service
Analyzes guitar audio to detect attack, sustain, dynamics,
and generates tone recommendations that improve over time.
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple
import warnings


def analyze_audio(path: str, sr: int = 22050) -> Dict:
    """
    Analyze an audio file and return detailed guitar-playing metrics.

    Detects:
    - Onsets (attack/transient detection)
    - Sustain (note decay characteristics)
    - Dynamics (RMS, peak, crest factor, dynamic range)
    - Tempo and rhythmic feel
    - Spectral centroid (brightness)

    Args:
        path: filesystem path to an audio file
        sr: target sample rate for analysis

    Returns:
        dict containing onsets, sustain, dynamics, spectral features,
        and a raw recommendation object for the learning engine
    """
    try:
        y, orig_sr = librosa.load(path, sr=sr, mono=True)
    except Exception as e:
        raise ValueError(f"Could not load audio: {e}")

    if len(y) == 0:
        raise ValueError("Audio file is empty or unreadable")

    # Ensure non-silent
    if np.max(np.abs(y)) < 1e-6:
        raise ValueError("Audio is silent")

    # ---- Onset detection (attack) ----
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, wait=3, pre_avg=3, post_avg=3,
        pre_max=3, post_max=3
    )
    onset_times = librosa.frames_to_time(onsets, sr=sr).tolist()
    onset_strengths = onset_env[onsets].tolist() if len(onsets) > 0 else []

    # Average attack intensity
    avg_attack = float(np.mean(onset_strengths)) if onset_strengths else 0.0
    max_attack = float(np.max(onset_strengths)) if onset_strengths else 0.0

    # ---- Sustain analysis ----
    # Use RMS energy over time to model note decay
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    # Find sustained regions (above -30 dB relative to peak)
    threshold_db = -30
    sustained_mask = rms_db > threshold_db
    sustain_ratio = float(np.mean(sustained_mask)) if len(sustained_mask) > 0 else 0.0

    # Decay rate: slope of RMS in dB over time
    times = np.arange(len(rms_db))
    if len(rms_db) > 10:
        # Fit line to dB envelope to estimate decay
        coeffs = np.polyfit(times, rms_db, 1)
        decay_slope = float(coeffs[0])
    else:
        decay_slope = 0.0

    sustain_regions = []
    in_region = False
    start_frame = 0
    for i, is_sustained in enumerate(sustained_mask):
        if is_sustained and not in_region:
            in_region = True
            start_frame = i
        elif not is_sustained and in_region:
            in_region = False
            end_time = librosa.frames_to_time(i, sr=sr, hop_length=hop_length)
            start_time = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length)
            sustain_regions.append({
                "start": round(start_time, 3),
                "end": round(end_time, 3),
                "duration": round(end_time - start_time, 3)
            })
    if in_region:
        end_time = librosa.frames_to_time(len(sustained_mask), sr=sr, hop_length=hop_length)
        start_time = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length)
        sustain_regions.append({
            "start": round(start_time, 3),
            "end": round(end_time, 3),
            "duration": round(end_time - start_time, 3)
        })

    avg_sustain_duration = (
        float(np.mean([r["duration"] for r in sustain_regions]))
        if sustain_regions else 0.0
    )

    # ---- Dynamics ----
    rms_value = float(np.sqrt(np.mean(y**2)))
    peak_value = float(np.max(np.abs(y)))
    crest_factor = peak_value / rms_value if rms_value > 1e-10 else 0.0

    # Dynamic range: difference between 95th and 5th percentile of dB
    y_db = librosa.amplitude_to_db(np.abs(y) + 1e-10, ref=np.max)
    dynamic_range = float(np.percentile(y_db, 95) - np.percentile(y_db, 5))

    # ---- Spectral features ----
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    avg_brightness = float(np.mean(spectral_centroid))
    brightness_variance = float(np.var(spectral_centroid))

    # Roughness / noisiness via spectral flatness
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    avg_flatness = float(np.mean(flatness))

    # ---- Tempo ----
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    try:
        tempo_val = float(tempo)
    except Exception:
        tempo_val = 0.0

    # ---- Assemble result ----
    result = {
        "onsets": {
            "count": len(onset_times),
            "times": [round(t, 3) for t in onset_times],
            "avg_strength": round(avg_attack, 4),
            "max_strength": round(max_attack, 4),
        },
        "sustain": {
            "regions": sustain_regions,
            "region_count": len(sustain_regions),
            "avg_duration_sec": round(avg_sustain_duration, 3),
            "sustain_ratio": round(sustain_ratio, 4),
            "decay_slope_db_per_frame": round(decay_slope, 6),
        },
        "dynamics": {
            "rms": round(rms_value, 6),
            "peak": round(peak_value, 6),
            "crest_factor": round(crest_factor, 2),
            "dynamic_range_db": round(dynamic_range, 2),
        },
        "spectral": {
            "avg_brightness_hz": round(avg_brightness, 1),
            "brightness_variance": round(brightness_variance, 1),
            "spectral_flatness": round(avg_flatness, 4),
        },
        "tempo": {
            "bpm": round(tempo_val, 1) if tempo_val > 0 else None,
        },
        "duration_sec": round(len(y) / sr, 3),
    }

    return result


def compute_playing_profile(analysis: Dict) -> Dict:
    """
    Derive a high-level playing profile from raw analysis.
    Used by the learning engine to categorize a player's style.
    """
    dynamics = analysis["dynamics"]
    sustain = analysis["sustain"]
    spectral = analysis["spectral"]
    onsets = analysis["onsets"]

    # Classify attack style
    if onsets["avg_strength"] > 0.7:
        attack_style = "aggressive"
    elif onsets["avg_strength"] > 0.4:
        attack_style = "moderate"
    else:
        attack_style = "soft"

    # Classify sustain preference
    if sustain["avg_duration_sec"] > 1.5 and sustain["sustain_ratio"] > 0.5:
        sustain_style = "long_sustaining"
    elif sustain["avg_duration_sec"] > 0.5:
        sustain_style = "medium_sustaining"
    else:
        sustain_style = "staccato"

    # Classify dynamics style
    if dynamics["dynamic_range_db"] > 30:
        dynamics_style = "very_dynamic"
    elif dynamics["dynamic_range_db"] > 15:
        dynamics_style = "moderately_dynamic"
    else:
        dynamics_style = "compressed"

    # Brightness
    if spectral["avg_brightness_hz"] > 3000:
        brightness = "bright"
    elif spectral["avg_brightness_hz"] > 1500:
        brightness = "balanced"
    else:
        brightness = "dark"

    return {
        "attack_style": attack_style,
        "sustain_style": sustain_style,
        "dynamics_style": dynamics_style,
        "brightness": brightness,
        "crest_factor": dynamics["crest_factor"],
        "tempo_bpm": analysis["tempo"]["bpm"],
    }
