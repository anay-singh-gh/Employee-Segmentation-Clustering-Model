"""Sections 1-3: load, explore, clean and check feature correlation."""
import pandas as pd

from . import config


def load_data(path=config.DATA_PATH) -> pd.DataFrame:
    """Section 1: load the IBM HR dataset and print a quick structural summary."""
    df = pd.read_csv(path)
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nAttrition Value Counts:")
    print(df["Attrition"].value_counts())
    print("\nData Types:")
    print(df.dtypes.value_counts())
    print("\nMissing values:", int(df.isna().sum().sum()))
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Section 2: drop constant/identifier columns and encode binary columns as 0/1.

    Attrition is encoded but is never part of the clustering features.
    """
    df = df.drop(columns=config.DROP_COLS)
    for col in config.BINARY_ENCODE_COLS:
        df[col] = df[col].map(config.BINARY_MAP)
    print("Remaining columns:", df.shape[1])
    return df


def correlated_pairs(df: pd.DataFrame, threshold=config.CORR_THRESHOLD):
    """Section 3: list highly correlated numeric pairs (|r| > threshold)."""
    cols = config.CANDIDATE_NUMERIC
    corr = df[cols].corr().abs()
    pairs = [(a, b, round(corr.loc[a, b], 2))
             for i, a in enumerate(cols)
             for b in cols[i + 1:] if corr.loc[a, b] > threshold]
    pairs = sorted(pairs, key=lambda x: -x[2])
    print(f"Highly correlated pairs (|r| > {threshold}):")
    for a, b, r in pairs:
        print(f"  {a:<22} {b:<22} r = {r}")
    return pairs
