# src/utils/cmapss.py
"""
Utilities for loading and inspecting NASA C-MAPSS turbofan datasets.

Supported layouts (auto-detected):

1) Flat:
   data/raw/train_FD001.txt
   data/raw/test_FD001.txt
   data/raw/RUL_FD001.txt

2) NASA nested:
   data/raw/Turbofan_Engine_Degradation_Simulation_Data_Set/CMAPSSData/train_FD001.txt
   data/raw/Turbofan_Engine_Degradation_Simulation_Data_Set/CMAPSSData/test_FD001.txt
   data/raw/Turbofan_Engine_Degradation_Simulation_Data_Set/CMAPSSData/RUL_FD001.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

# ---- Canonical column names for C-MAPSS (26 columns when all are present) ----
COLS: List[str] = (
    ["engine_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"s{i}" for i in range(1, 22)]
)

ALLOWED_FD: Tuple[str, ...] = ("FD001", "FD002", "FD003", "FD004")

# ---- Dataset metadata (from the NASA README / literature) ----
FD_METADATA: Dict[str, Dict[str, object]] = {
    # Single condition, ONE fault mode (HPC degradation)
    "FD001": {"conditions": 1, "fault_modes": 1, "description": "HPC degradation"},
    # SIX conditions, ONE fault mode (HPC degradation)
    "FD002": {"conditions": 6, "fault_modes": 1, "description": "HPC degradation"},
    # ONE condition, TWO fault modes (HPC + Fan)
    "FD003": {"conditions": 1, "fault_modes": 2, "description": "HPC + Fan degradation"},
    # SIX conditions, TWO fault modes (HPC + Fan)
    "FD004": {"conditions": 6, "fault_modes": 2, "description": "HPC + Fan degradation"},
}


@dataclass
class FDData:
    """Container for one FD variant."""
    train: pd.DataFrame
    test: pd.DataFrame
    rul_test: pd.Series


# ------------------------- Path helpers ------------------------- #
def _candidate_paths(base_dir: Path, fd: str) -> List[Tuple[Path, Path, Path]]:
    """Return candidate (train, test, rul) triplets to try in order."""
    # Flat layout
    flat = (
        base_dir / f"train_{fd}.txt",
        base_dir / f"test_{fd}.txt",
        base_dir / f"RUL_{fd}.txt",
    )
    # NASA nested layout
    nested_root = base_dir / "Turbofan_Engine_Degradation_Simulation_Data_Set" / "CMAPSSData"
    nested = (
        nested_root / f"train_{fd}.txt",
        nested_root / f"test_{fd}.txt",
        nested_root / f"RUL_{fd}.txt",
    )
    return [flat, nested]


def _resolve_paths(base_dir: Path, fd: str) -> Tuple[Path, Path, Path]:
    """Find the first existing triplet of files; raise if none found."""
    for train_p, test_p, rul_p in _candidate_paths(base_dir, fd):
        if train_p.exists() and test_p.exists() and rul_p.exists():
            return train_p, test_p, rul_p
    tried = []
    for t, te, r in _candidate_paths(base_dir, fd):
        tried.extend([str(t), str(te), str(r)])
    raise FileNotFoundError(
        f"Could not find files for {fd}. Looked for:\n- " + "\n- ".join(tried)
    )


# ------------------------- Readers & validators ------------------------- #
def _read_space_delimited(path: Path, expected_cols: List[str] = COLS) -> pd.DataFrame:
    """
    Read a space-delimited file with variable whitespace & no header.
    Some mirrors include a trailing empty column; we trim to expected size.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] > len(expected_cols):
        df = df.iloc[:, : len(expected_cols)]
    df.columns = expected_cols[: df.shape[1]]
    # Normalize dtypes
    if "engine_id" in df.columns:
        df["engine_id"] = df["engine_id"].astype(int)
    if "cycle" in df.columns:
        df["cycle"] = df["cycle"].astype(int)
    return df


def _validate_columns(df: pd.DataFrame, min_cols: Iterable[str] = ("engine_id", "cycle")) -> None:
    missing = [c for c in min_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# ------------------------- Public API ------------------------- #
def load_fd_raw(fd: str = "FD001", base_dir: str = "data/raw", verbose: bool = True) -> FDData:
    """
    Load one C-MAPSS FD variant (FD001–FD004) from base_dir.
    Returns FDData(train, test, rul_test).

    Parameters
    ----------
    fd : str
        One of {"FD001","FD002","FD003","FD004"}.
    base_dir : str
        Directory that contains either the flat files or the NASA nested structure.
    verbose : bool
        If True, print a short summary (meta + shapes).

    Raises
    ------
    FileNotFoundError
        If the files cannot be located in either flat or nested structure.
    ValueError
        If fd is invalid or columns are missing.
    """
    if fd not in ALLOWED_FD:
        raise ValueError(f"`fd` must be one of {ALLOWED_FD}, got {fd!r}")

    base = Path(base_dir)
    train_path, test_path, rul_path = _resolve_paths(base, fd)

    train = _read_space_delimited(train_path)
    test = _read_space_delimited(test_path)
    rul_test = pd.read_csv(rul_path, sep=r"\s+", header=None, engine="python")[0].rename("RUL")

    _validate_columns(train)
    _validate_columns(test)

    # Sort for sanity
    train = train.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    test = test.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

    if verbose:
        describe_fd(fd, train=train, test=test, rul_test=rul_test)

    return FDData(train=train, test=test, rul_test=rul_test)


def describe_fd(fd: str, train: Optional[pd.DataFrame] = None,
                test: Optional[pd.DataFrame] = None,
                rul_test: Optional[pd.Series] = None) -> None:
    """
    Print dataset metadata and (optionally) shapes + engine counts.
    """
    meta = FD_METADATA.get(fd, {})
    cond = meta.get("conditions", "?")
    faults = meta.get("fault_modes", "?")
    desc = meta.get("description", "n/a")

    print(f"Dataset {fd}: {cond} condition(s), {faults} fault mode(s) — {desc}")

    if train is not None:
        n_eng_train = train["engine_id"].nunique() if "engine_id" in train.columns else "?"
        print(f"  Train shape: {tuple(train.shape)}  | engines: {n_eng_train}")
    if test is not None:
        n_eng_test = test["engine_id"].nunique() if "engine_id" in test.columns else "?"
        print(f"  Test  shape: {tuple(test.shape)}  | engines: {n_eng_test}")
    if rul_test is not None:
        print(f"  RUL_test length: {len(rul_test)}")


def add_unit_cycle_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure rows are sorted by (engine_id, cycle) and that cycle is int.
    (Mostly redundant if using load_fd_raw, but safe to call.)
    """
    out = df.copy()
    out["engine_id"] = out["engine_id"].astype(int)
    out["cycle"] = out["cycle"].astype(int)
    out = out.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    return out


def make_train_rul_labels(train: pd.DataFrame, column_name: str = "RUL") -> pd.DataFrame:
    """
    For training sequences (which run to failure), compute:
    RUL = max_cycle_for_engine - cycle
    """
    _validate_columns(train)
    out = train.copy()
    max_cycle = out.groupby("engine_id")["cycle"].transform("max")
    out[column_name] = (max_cycle - out["cycle"]).astype(int)
    return out


def sequence_length_stats(df: pd.DataFrame) -> pd.Series:
    """
    Convenience: per-engine maximum cycle summary (count, mean, std, min, quartiles, max).
    """
    _validate_columns(df)
    return df.groupby("engine_id")["cycle"].max().describe()


def select_sensors(df: pd.DataFrame, keep: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Keep only a subset of sensor columns, plus engine_id/cycle/op_settings.
    If keep is None, returns df unchanged.
    """
    if keep is None:
        return df
    base_cols = ["engine_id", "cycle"] + [c for c in df.columns if c.startswith("op_setting_")]
    sensor_cols = [c for c in df.columns if c in keep]
    cols = base_cols + sensor_cols
    cols = [c for c in cols if c in df.columns]  # guard
    return df[cols].copy()
