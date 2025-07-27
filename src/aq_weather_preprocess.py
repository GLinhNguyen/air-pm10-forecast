# src/aq_weather_preprocess.py
"""
Preprocess Kraków air‑quality + weather all the way to decoded JSON.

Changes (2025‑07‑27 – v4)
=========================
* **Case granularity selectable** via ``case_scope`` parameter:
  * ``"station"`` – original behaviour (one station per case)
  * ``"day"``     – one calendar day per case with many stations inside
* ``preprocess_all`` now accepts ``case_scope`` and forwards it to
  :func:`build_cases`.
* Internally caches weather by day for fast lookup when building day‑level
  cases.
"""

from __future__ import annotations
import json
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ──────────────── Config ────────────────────────────────────────────────────
SENTINELS_PM10 = {-999, 9999, 99999}
MAX_GAP_HOURS  = 3          # interpolate gaps ≤ this length
KEEP_RAW_WX    = False      # True → keep TMP/WND strings too

# ─────────────── Weather decoders ───────────────────────────────────────────

def _to_float_safe(tok: str | None, scale: float = 1.0) -> float | None:
    if tok in (None, "", "99999", "+99999", "9999", "+9999", "99"):
        return None
    try:
        return float(tok) / scale
    except ValueError:
        return None


def _decode_tmp(tmp: str) -> Tuple[float | None, int | None]:
    parts = str(tmp).split(",")
    return _to_float_safe(parts[0], 10.0), int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None


def _decode_wnd(wnd: str) -> Tuple[float | None, float | None, int | None, int | None]:
    parts = str(wnd).split(",")
    if len(parts) < 5:
        return None, None, None, None
    return (
        _to_float_safe(parts[0]),                       # dir °
        _to_float_safe(parts[3], 10.0),                 # speed m s‑1
        int(parts[1]) if parts[1].isdigit() else None,  # dir QC
        int(parts[4]) if parts[4].isdigit() else None,  # spd QC
    )

# ─────────────── Station metadata ───────────────────────────────────────────

def load_station_meta(path: Path) -> pd.DataFrame:
    return (
        pd.read_excel(path)
          .rename(columns={
              "Station Code": "station_code",
              "WGS84 φ N": "latitude",
              "WGS84 λ E": "longitude"})
          [["station_code", "latitude", "longitude"]]
    )

# ─────────────── PM10 handlers ──────────────────────────────────────────────

def _read_single_pm10_excel(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path).rename(columns={"DateTime": "timestamp"})
    df = df.melt(id_vars="timestamp", var_name="station_code", value_name="pm10")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _impute_pm10(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.pivot(index="timestamp", columns="station_code", values="pm10")
    full = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="1H", tz="UTC"))
    full = full.interpolate(method="time", limit=MAX_GAP_HOURS, limit_direction="both")
    long = (full.reset_index()
                 .melt(id_vars="index", var_name="station_code", value_name="pm10")
                 .rename(columns={"index": "timestamp"})
                 .dropna(subset=["pm10"]))
    return long


def load_pm10_history(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob("*_PM10_1g.xlsx"))
    raw  = pd.concat([_read_single_pm10_excel(f) for f in files], ignore_index=True)
    clean = raw[~raw["pm10"].isin(SENTINELS_PM10)].dropna(subset=["pm10"])
    clean = clean.drop_duplicates(["station_code", "timestamp"])
    return _impute_pm10(clean).sort_values(["station_code", "timestamp"])

# ─────────────── Weather handlers ───────────────────────────────────────────

def _read_single_weather_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["STATION", "DATE", "TMP", "WND"])
    df = df.rename(columns={"STATION": "station_id", "DATE": "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["temp_c"], df["tmp_qc"]                 = zip(*df["TMP"].apply(_decode_tmp))
    df["wind_dir_deg"], df["wind_speed_ms"], \
    df["wnd_dir_qc"], df["wnd_spd_qc"]         = zip(*df["WND"].apply(_decode_wnd))
    if not KEEP_RAW_WX:
        df = df.drop(columns=["TMP", "WND"])
    return df.drop_duplicates(["station_id", "date"])


def load_weather_raw(folder: Path) -> pd.DataFrame:
    return (pd.concat([_read_single_weather_csv(f) for f in sorted(folder.glob("*.csv"))], ignore_index=True)
              .sort_values(["station_id", "date"]))

# ─────────────── Build JSON cases ───────────────────────────────────────────

def build_cases(pm10: pd.DataFrame,
                meta: pd.DataFrame,
                weather: pd.DataFrame | None = None,
                *,
                horizon_hours: int = 24,
                case_scope: str = "station") -> Dict[str, List[dict]]:
    """Build the JSON dataset.

    Parameters
    ----------
    pm10 : DataFrame – long‑format PM₁₀ data.
    meta : DataFrame – station coordinates.
    weather : DataFrame or None – decoded hourly weather.
    horizon_hours : int – forecast horizon (not yet used for features).
    case_scope : "station" | "day" – granularity of a case.
    """

    allowed = {"station", "day"}
    if case_scope not in allowed:
        raise ValueError(f"case_scope must be one of {allowed}")

    # Merge coordinates once
    pm10_geo = pm10.merge(meta, on="station_code", how="left")

    # Pre‑index weather by day for quick attach
    wx_by_day: Dict[pd.Timestamp, List[dict]] = {}
    if weather is not None and not weather.empty:
        weather = weather.copy()
        weather["day"] = weather["date"].dt.normalize()
        for d, g in weather.groupby("day"):
            wx_by_day[d] = [
                {"date": dt.isoformat(),
                 "temp_c": None if pd.isna(t) else round(t, 1),
                 "wind_dir_deg": wd,
                 "wind_speed_ms": ws}
                for dt, t, wd, ws in zip(g["date"],
                                         g["temp_c"],
                                         g["wind_dir_deg"],
                                         g["wind_speed_ms"])
            ]

    cases: List[dict] = []

    # ── Per‑station case (legacy) ──────────────────────────────────────────
    if case_scope == "station":
        for code, g in pm10_geo.groupby("station_code"):
            g = g.sort_values("timestamp")
            history = [{"timestamp": ts.isoformat(), "pm10": round(v, 4)}
                       for ts, v in zip(g["timestamp"], g["pm10"])]
            pred_start = (g["timestamp"].iloc[-1] + timedelta(hours=1)).isoformat()
            day = g["timestamp"].dt.normalize().iloc[-1]
            cases.append({
                "case_id": f"{code}_{day.date()}",
                "stations": [{
                    "station_code": code,
                    "longitude": g["longitude"].iat[0],
                    "latitude":  g["latitude"].iat[0],
                    "history":   history,
                }],
                "target": {
                    "longitude": g["longitude"].iat[0],
                    "latitude":  g["latitude"].iat[0],
                    "prediction_start_time": pred_start,
                },
                **({"weather": wx_by_day.get(day)} if wx_by_day else {}),
            })

    # ── Per‑day case (new) ────────────────────────────────────────────────
    else:
        pm10_geo = pm10_geo.copy()
        pm10_geo["day"] = pm10_geo["timestamp"].dt.normalize()
        for day, g_day in pm10_geo.groupby("day"):
            stations_payload: List[dict] = []
            for code, g_st in g_day.groupby("station_code"):
                g_st = g_st.sort_values("timestamp")
                history = [{"timestamp": ts.isoformat(), "pm10": round(v, 4)}
                           for ts, v in zip(g_st["timestamp"], g_st["pm10"])]
                stations_payload.append({
                    "station_code": code,
                    "longitude": g_st["longitude"].iat[0],
                    "latitude":  g_st["latitude"].iat[0],
                    "history":   history,
                })
            # Target = centre of mass of all stations that day
            centres = g_day[["longitude", "latitude"]].mean()
            pred_start = (day + timedelta(days=1)).isoformat()
            cases.append({
                "case_id": f"day_{day.date()}",
                "stations": stations_payload,
                "target": {
                    "longitude": float(centres["longitude"]),
                    "latitude":  float(centres["latitude"]),
                    "prediction_start_time": pred_start,
                },
                **({"weather": wx_by_day.get(day)} if wx_by_day else {}),
            })

    return {"cases": cases}

# ─────────────── IO helpers ─────────────────────────────────────────────────

def save_cases_to_json(cases: Dict, path: Path, *, indent: int = 2) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=indent, ensure_ascii=False)
    print(f"✅  Saved {len(cases['cases'])} cases → {path}")


def preprocess_all(aq_dir: Path, wx_dir: Path, 
                   *, case_scope: str = "station") -> Dict:
    """End‑to‑end helper: load, clean, build, save."""
    meta   = load_station_meta(aq_dir / "Stations.xlsx")
    pm10   = load_pm10_history(aq_dir)
    wx     = load_weather_raw(wx_dir)
    cases  = build_cases(pm10, meta, wx, case_scope=case_scope)
    return cases

# ─────────────── Flatten back for EDA ───────────────────────────────────────

def flatten_cases_to_df(json_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pm_rows, wx_rows = [], []
    for case in data["cases"]:
        for st in case["stations"]:
            lon, lat = st["longitude"], st["latitude"]
            for h in st["history"]:
                pm_rows.append({
                    "case_id":      case["case_id"],
                    "station_code": st["station_code"],
                    "timestamp":    pd.to_datetime(h["timestamp"], utc=True),
                    "pm10":         h["pm10"],
                    "longitude":    lon,
                    "latitude":     lat,
                })
        for w in case.get("weather", []):
            wx_rows.append({
                "case_id": case["case_id"],
                "date":    pd.to_datetime(w["date"], utc=True),
                "temp_c":  w.get("temp_c"),
                "wind_dir_deg":  w.get("wind_dir_deg"),
                "wind_speed_ms": w.get("wind_speed_ms"),
            })

    return (pd.DataFrame(pm_rows).sort_values(["station_code", "timestamp"]),
            pd.DataFrame(wx_rows).sort_values("date"))
