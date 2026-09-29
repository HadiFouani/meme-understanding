from pathlib import Path
from datasets import load_dataset

# C:\...\Foundational Models for Meme Understanding\meme-understanding
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# C:\...\Foundational Models for Meme Understanding
WORKSPACE_ROOT = PROJECT_ROOT.parent

DATASET_DIR = (
    WORKSPACE_ROOT
    / "MemeLens"
    / "data"
    / "hf_raw"
    / "Hateful_en__MIMIC_Islamophpbia"
)

print("Dataset directory:")
print(DATASET_DIR)

print("\nExists:")
print(DATASET_DIR.exists())

print("\nParquet files:")
for file in DATASET_DIR.glob("*.parquet"):
    print(file.name)

ds = load_dataset(
    "parquet",
    data_files={
        "train": str(DATASET_DIR / "train-*.parquet"),
        "validation": str(DATASET_DIR / "validation-*.parquet"),
        "test": str(DATASET_DIR / "test-*.parquet"),
    },
)

print("\nDATASET:")
print(ds)

sample = ds["test"][60]

print("\nAVAILABLE FIELDS:")
print(sample.keys())

print("\nTEXT:")
print(sample["text"])

print("\nLABEL:")
print(sample["label"])

print("\nIMAGE:")
print(sample["image"])

sample["image"].show()