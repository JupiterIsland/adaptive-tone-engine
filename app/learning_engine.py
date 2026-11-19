"""
Adaptive Tone Engine — Recommendation & Learning Engine
Learns player preferences from analysis history and generates
tone recommendations that improve over time.
"""

import json
import sqlite3
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

from .audio_analysis import compute_playing_profile

DB_PATH = Path(__file__).parent.parent / "data" / "tone_engine.db"


def _get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    conn = _get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id TEXT NOT NULL,
            filename TEXT,
            analysis_json TEXT NOT NULL,
            profile_json TEXT NOT NULL,
            recommendation_json TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS player_profiles (
            player_id TEXT PRIMARY KEY,
            total_analyses INTEGER DEFAULT 0,
            aggregated_profile TEXT,
            preferred_tone_json TEXT,
            last_updated TEXT
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id TEXT NOT NULL,
            analysis_id INTEGER,
            recommendation_id TEXT,
            rating INTEGER,  -- 1-5 or -1/0/1
            comment TEXT,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_analyses_player ON analyses(player_id);
        CREATE INDEX IF NOT EXISTS idx_feedback_player ON feedback(player_id);
    """)
    conn.commit()
    conn.close()


def store_analysis(
    player_id: str,
    filename: Optional[str],
    analysis: Dict,
    recommendation: Optional[Dict] = None,
) -> int:
    """Store an analysis result and return the analysis_id."""
    profile = compute_playing_profile(analysis)
    conn = _get_db()
    cursor = conn.execute(
        """
        INSERT INTO analyses (player_id, filename, analysis_json, profile_json, recommendation_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            player_id,
            filename,
            json.dumps(analysis),
            json.dumps(profile),
            json.dumps(recommendation) if recommendation else None,
            datetime.utcnow().isoformat(),
        ),
    )
    analysis_id = cursor.lastrowid
    conn.commit()
    conn.close()
    _update_player_profile(player_id)
    return analysis_id


def _update_player_profile(player_id: str) -> None:
    """Recompute aggregated player profile from all analyses."""
    conn = _get_db()
    rows = conn.execute(
        "SELECT profile_json FROM analyses WHERE player_id = ? ORDER BY created_at DESC",
        (player_id,),
    ).fetchall()
    conn.close()

    if not rows:
        return

    profiles = [json.loads(row["profile_json"]) for row in rows]

    # Simple aggregation: mode for categorical, mean for numeric
    def mode(values: List[str]) -> str:
        from collections import Counter
        return Counter(values).most_common(1)[0][0]

    agg = {
        "attack_style": mode([p["attack_style"] for p in profiles]),
        "sustain_style": mode([p["sustain_style"] for p in profiles]),
        "dynamics_style": mode([p["dynamics_style"] for p in profiles]),
        "brightness": mode([p["brightness"] for p in profiles]),
        "avg_crest_factor": round(
            sum(p["crest_factor"] for p in profiles) / len(profiles), 2
        ),
        "recent_tempo_bpm": profiles[0]["tempo_bpm"],
        "total_recordings": len(profiles),
    }

    conn = _get_db()
    conn.execute(
        """
        INSERT INTO player_profiles (player_id, total_analyses, aggregated_profile, last_updated)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(player_id) DO UPDATE SET
            total_analyses = excluded.total_analyses,
            aggregated_profile = excluded.aggregated_profile,
            last_updated = excluded.last_updated
        """,
        (player_id, len(profiles), json.dumps(agg), datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_player_profile(player_id: str) -> Optional[Dict]:
    """Retrieve aggregated profile for a player."""
    conn = _get_db()
    row = conn.execute(
        "SELECT * FROM player_profiles WHERE player_id = ?", (player_id,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "player_id": row["player_id"],
        "total_analyses": row["total_analyses"],
        "profile": json.loads(row["aggregated_profile"]) if row["aggregated_profile"] else None,
        "preferred_tone": json.loads(row["preferred_tone_json"]) if row["preferred_tone_json"] else None,
        "last_updated": row["last_updated"],
    }


def generate_recommendation(analysis: Dict, player_id: str) -> Dict:
    """
    Generate a tone recommendation based on current analysis
    and the player's historical preferences.
    """
    profile = compute_playing_profile(analysis)
    player = get_player_profile(player_id)

    rec = {
        "gain_adjustment_db": 0.0,
        "tone_stack": {},
        "effects": {},
        "reasoning": [],
    }

    # ---- Attack-based recommendations ----
    if profile["attack_style"] == "aggressive":
        rec["tone_stack"]["treble"] = "-2 dB (smooth sharp attacks)"
        rec["tone_stack"]["presence"] = "-1 dB"
        rec["effects"]["compressor"] = "light (3:1 ratio, slow attack)"
        rec["reasoning"].append("Aggressive attack detected — smoothing treble and adding light compression")
    elif profile["attack_style"] == "soft":
        rec["tone_stack"]["treble"] = "+2 dB (add bite)"
        rec["effects"]["compressor"] = "medium (4:1 ratio, fast attack)"
        rec["reasoning"].append("Soft attack detected — boosting treble presence and compression to bring notes forward")
    else:
        rec["effects"]["compressor"] = "off or light"
        rec["reasoning"].append("Moderate attack — natural dynamics preserved")

    # ---- Sustain-based recommendations ----
    if profile["sustain_style"] == "staccato":
        rec["effects"]["reverb"] = "medium hall (2.5s decay)"
        rec["effects"]["delay"] = "slapback (80ms, 20% feedback)"
        rec["tone_stack"]["mids"] = "+1 dB"
        rec["reasoning"].append("Staccato playing — adding reverb/delay sustain and mid push")
    elif profile["sustain_style"] == "long_sustaining":
        rec["effects"]["reverb"] = "short plate (0.8s decay)"
        rec["effects"]["compressor"] = rec["effects"].get("compressor", "off") + " + limiter"
        rec["reasoning"].append("Long sustain detected — reducing reverb, adding limiting to control mud")

    # ---- Dynamics-based recommendations ----
    if profile["dynamics_style"] == "very_dynamic":
        rec["gain_adjustment_db"] = 0.0
        rec["effects"]["compressor"] = "off"
        rec["reasoning"].append("Wide dynamic range — compressor off to preserve expression")
    elif profile["dynamics_style"] == "compressed":
        rec["gain_adjustment_db"] = -3.0
        rec["effects"]["compressor"] = "off"
        rec["tone_stack"]["mids"] = rec["tone_stack"].get("mids", "0 dB") + " +1 dB"
        rec["reasoning"].append("Compressed input — reducing gain, adding midrange clarity")

    # ---- Brightness-based recommendations ----
    if profile["brightness"] == "bright":
        rec["tone_stack"]["treble"] = rec["tone_stack"].get("treble", "0 dB") + " -1 dB"
        rec["tone_stack"]["presence"] = "-2 dB"
        rec["reasoning"].append("Bright signal — rolling off highs")
    elif profile["brightness"] == "dark":
        rec["tone_stack"]["treble"] = rec["tone_stack"].get("treble", "0 dB") + " +2 dB"
        rec["tone_stack"]["presence"] = "+1 dB"
        rec["reasoning"].append("Dark signal — boosting highs and presence")

    # ---- Learn from history: if player has rejected similar recs, adjust ----
    if player and player.get("profile"):
        agg = player["profile"]
        # If historically the player prefers long sustain but we detect staccato,
        # we might be analyzing a specific passage rather than overall style.
        # Keep the recommendation but flag it.
        if agg.get("sustain_style") == "long_sustaining" and profile["sustain_style"] == "staccato":
            rec["reasoning"].append(
                "Note: player usually sustains notes; this passage is staccato. "
                "Recommendation applies to this passage only."
            )

    return rec


def record_feedback(
    player_id: str,
    analysis_id: int,
    rating: int,
    comment: Optional[str] = None,
) -> None:
    """Record player feedback on a recommendation."""
    conn = _get_db()
    conn.execute(
        """
        INSERT INTO feedback (player_id, analysis_id, rating, comment, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (player_id, analysis_id, rating, comment, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_player_history(player_id: str, limit: int = 20) -> List[Dict]:
    """Get recent analysis history for a player."""
    conn = _get_db()
    rows = conn.execute(
        """
        SELECT id, filename, analysis_json, recommendation_json, created_at
        FROM analyses WHERE player_id = ? ORDER BY created_at DESC LIMIT ?
        """,
        (player_id, limit),
    ).fetchall()
    conn.close()
    return [
        {
            "analysis_id": row["id"],
            "filename": row["filename"],
            "analysis": json.loads(row["analysis_json"]),
            "recommendation": json.loads(row["recommendation_json"]) if row["recommendation_json"] else None,
            "created_at": row["created_at"],
        }
        for row in rows
    ]
