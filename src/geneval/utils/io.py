from typing import Any, Dict
import pandas as pd
import numpy as np
import os
import json

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    return pd.read_csv(file_path)

def save_data(data: pd.DataFrame, file_path: str) -> None:
    """Save data to a CSV file."""
    data.to_csv(file_path, index=False)

def load_json(file_path: str) -> Dict[str, Any]:
    """Load data from a JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data: Dict[str, Any], file_path: str) -> None:
    """Save data to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)