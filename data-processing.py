import numpy as np
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

# all the city csvs in the GSV Cities dataset
CITY_FILES = [
    "Bangkok.csv",
    "Barcelona.csv",
    "Boston.csv",
    "Brussels.csv",
    "BuenosAires.csv",
    "Chicago.csv",
    "Lisbon.csv",
    "London.csv",
    "LosAngeles.csv",
    "Madrid.csv",
    "Medellin.csv",
    "Melbourne.csv",
    "MexicoCity.csv",
    "Miami.csv",
    "Minneapolis.csv",
    "OSL.csv",
    "Osaka.csv",
    "PRG.csv",
    "PRS.csv",
    "Phoenix.csv",
    "Rome.csv",
    "TRT.csv",
    "WashingtonDC.csv",
]

HANDLE = "amaralibey/gsv-cities"


def load_all_cities() -> pd.DataFrame:
    """Load all city DataFrames and combine with a 'city' column."""
    frames = []
    for filename in CITY_FILES:
        city_name = filename.replace(".csv", "")
        file_path = f"Dataframes/{filename}"
        df = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            HANDLE,
            file_path,
        )
        df["city"] = city_name
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined.set_index("place_id")


def print_overview(all_df: pd.DataFrame) -> None:
    """Print overall dataset stats and per-city summary."""
    print("=" * 70)
    print("  GSV CITIES — OVERALL OVERVIEW")
    print("=" * 70)

    print("\n--- Totals ---")
    print(f"  Total records:  {len(all_df):,}")
    print(f"  Cities:         {all_df['city'].nunique()}")
    print(f"  Columns:        {list(all_df.columns)}")

    if "year" in all_df.columns:
        print(f"\n  Year range:     {all_df['year'].min()} – {all_df['year'].max()}")

    print("\n--- Missing values ---")
    missing = all_df.isna().sum()
    if missing.any():
        print(missing[missing > 0].to_string())
    else:
        print("  None")

    if "year" in all_df.columns:
        print("\n--- Records per year (all cities) ---")
        print(all_df["year"].value_counts().sort_index().to_string())

    # Per-city summary table
    print("\n" + "=" * 70)
    print("  PER-CITY SUMMARY")
    print("=" * 70)

    def row_stats(g):
        d = {"records": len(g)}
        if "year" in g.columns:
            d["year_min"] = g["year"].min()
            d["year_max"] = g["year"].max()
        if "lat" in g.columns:
            d["lat_min"] = g["lat"].min()
            d["lat_max"] = g["lat"].max()
        if "lon" in g.columns:
            d["lon_min"] = g["lon"].min()
            d["lon_max"] = g["lon"].max()
        return pd.Series(d)

    city_summary = all_df.groupby("city", sort=False).apply(row_stats)
    city_summary["records"] = city_summary["records"].astype(int)
    if "year_min" in city_summary.columns:
        city_summary["year_min"] = city_summary["year_min"].astype(int)
        city_summary["year_max"] = city_summary["year_max"].astype(int)
    print(city_summary.to_string())

    print("=" * 70)


if __name__ == "__main__":
    all_df = load_all_cities()
    print("First 5 records (combined):")
    print(all_df.head())
    print()
    print_overview(all_df)