from pathlib import Path
import os
from dotenv import load_dotenv
import torch

DEBUG_MODE: bool = True

BASE_DIR: Path = Path(__file__).parent
RELIABLE_SOURCES_PATH: Path = BASE_DIR / "data" / "reliable-sources.pkl"
SAMPLE_INPUT_PATH: Path = BASE_DIR / "data" / "sample-input.txt"
SAMPLE_SOURCE_TEXTS_PATH: Path = BASE_DIR / "data" / "sample-source-texts.txt"
MBFC_PATH = BASE_DIR / "data" / "mbfc.csv"

load_dotenv(BASE_DIR / ".env")
SERPAPI_KEY: str = os.getenv("SERPAPI_KEY", "")

DEMO_MODE: bool = False # if True, the program will use hardcoded text instead of loading it from URLs

EXAMPLE_URL = "https://example.com"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

from enum import Enum
from pydantic import BaseModel, HttpUrl

class Relation(Enum):
    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"

def print_list(lst: list[str]) -> None:
    if not lst:
        print("NO ITEMS FOUND")
        return
    for i, v in enumerate(lst, start=1):
        print(f"{i}. {v}")
    print()