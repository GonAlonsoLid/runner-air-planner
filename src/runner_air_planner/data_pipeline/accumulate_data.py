"""Utilities to accumulate historical data from multiple data collection runs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_accumulated_dataset(accumulated_path: Path) -> pd.DataFrame:
    """Load previously accumulated dataset.
    
    Args:
        accumulated_path: Path to accumulated CSV file
        
    Returns:
        DataFrame with accumulated data, or empty DataFrame if file doesn't exist
    """
    if not accumulated_path.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(accumulated_path)
    if "measurement_time" in df.columns:
        df["measurement_time"] = pd.to_datetime(df["measurement_time"])
    if "weather_observed_at" in df.columns:
        df["weather_observed_at"] = pd.to_datetime(df["weather_observed_at"])
    
    return df


def accumulate_ml_dataset(
    new_data: pd.DataFrame,
    accumulated_path: Path,
    *,
    max_days: int = 30,
    deduplicate: bool = True,
) -> pd.DataFrame:
    """Accumulate new ML dataset with existing historical data.
    
    Args:
        new_data: New dataset to add
        accumulated_path: Path to accumulated CSV file
        max_days: Maximum days of history to keep (default 30)
        deduplicate: Remove duplicate records based on station_code + measurement_time
        
    Returns:
        Combined DataFrame with accumulated data
    """
    # Load existing data
    accumulated = load_accumulated_dataset(accumulated_path)
    
    if accumulated.empty:
        combined = new_data.copy()
    else:
        # Combine datasets
        combined = pd.concat([accumulated, new_data], ignore_index=True)
        
        # Remove duplicates if requested
        if deduplicate and "station_code" in combined.columns and "measurement_time" in combined.columns:
            # Keep the most recent version of each station+time combination
            combined = combined.sort_values("measurement_time", ascending=False)
            combined = combined.drop_duplicates(
                subset=["station_code", "measurement_time"],
                keep="first"
            )
        
        # Filter by max_days if specified
        if max_days > 0 and "measurement_time" in combined.columns:
            latest_time = combined["measurement_time"].max()
            cutoff_time = latest_time - pd.Timedelta(days=max_days)
            combined = combined[combined["measurement_time"] >= cutoff_time]
    
    # Sort by station and time
    if "station_code" in combined.columns and "measurement_time" in combined.columns:
        combined = combined.sort_values(["station_code", "measurement_time"]).reset_index(drop=True)
    
    return combined


def save_accumulated_dataset(
    data: pd.DataFrame,
    accumulated_path: Path,
) -> Path:
    """Save accumulated dataset to CSV.
    
    Args:
        data: DataFrame to save
        accumulated_path: Path to save the file
        
    Returns:
        Path where file was saved
    """
    accumulated_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert datetime columns to strings for CSV
    data_to_save = data.copy()
    for col in data_to_save.columns:
        if pd.api.types.is_datetime64_any_dtype(data_to_save[col]):
            data_to_save[col] = data_to_save[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    data_to_save.to_csv(accumulated_path, index=False)
    return accumulated_path


__all__ = [
    "load_accumulated_dataset",
    "accumulate_ml_dataset",
    "save_accumulated_dataset",
]

