import pandas as pd
from pathlib import Path

# CHANGE split name here: train / val / test
SPLIT = "val"   # 👈 change to "train" or "test" when needed

SPLIT_DIR = Path(f"data/raw/word_level/{SPLIT}")
CSV_FILE = SPLIT_DIR / f"{SPLIT}idation_labels.csv" #{SPLIT}ing for testing and training, {SPLIT}idation for valid
OUTPUT_FILE = SPLIT_DIR / "labels.txt"

# Read CSV
df = pd.read_csv(CSV_FILE)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        image_name = str(row["IMAGE"]).strip()
        label = str(row["MEDICINE_NAME"]).strip()

        if image_name and label:
            f.write(f"{image_name} {label}\n")

print(f"✅ labels.txt created for {SPLIT}")
