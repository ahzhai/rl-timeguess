#!/usr/bin/env bash
# Download GSV-Cities dataset from Kaggle: https://www.kaggle.com/datasets/amaralibey/gsv-cities/data
# Requires: pip install kaggle, and ~/.kaggle/kaggle.json (from Kaggle Account → Create New Token)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

mkdir -p "$DATA_DIR"
cd "$PROJECT_ROOT"

if ! command -v kaggle &> /dev/null; then
  echo "Kaggle CLI not found. Install with: pip install kaggle"
  exit 1
fi

echo "Downloading amaralibey/gsv-cities..."
kaggle datasets download -d amaralibey/gsv-cities -p "$DATA_DIR"

echo "Unzipping..."
unzip -o "$DATA_DIR/gsv-cities.zip" -d "$DATA_DIR/gsv-cities"

echo "Done. Data is in $DATA_DIR/gsv-cities/"
