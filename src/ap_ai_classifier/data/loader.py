from pathlib import Path
import pandas as pd
from ap_ai_classifier.config import RAW_CSV_PATH

REQUIRED_COLUMNS = [
    "DATE", "REF", "DETAIL", "NET", "TC", "VAT",
    "SUPPLIER", "NOMINAL", "DEPARTMENT"
]

def load_raw_df(csv_path: Path = RAW_CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    return df
