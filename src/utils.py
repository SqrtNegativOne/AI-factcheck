from pathlib import Path
import os
from dotenv import load_dotenv
from huggingface_hub import login
import torch

# Common imports to claim_extraction_models and other_models:
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR: Path = Path(__file__).parent
RELIABLE_SOURCES_PATH: Path = BASE_DIR / "data" / "reliable-sources.pkl"
SAMPLE_INPUT_PATH: Path = BASE_DIR / "data" / "sample-input.txt"
SAMPLE_SOURCE_TEXTS_PATH: Path = BASE_DIR / "data" / "sample-source-texts.txt"
MBFC_PATH = BASE_DIR / "data" / "mbfc.csv"

load_dotenv(BASE_DIR / ".env")
SERPAPI_KEY: str = os.getenv("SERPAPI_KEY", "")

HF_KEY = os.getenv("HF_KEY")
if HF_KEY is None:
    raise ValueError("HF_KEY not found in environment variables")
# Log in to Hugging Face Hub
login(token=HF_KEY)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

from enum import Enum

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

if __name__ == "__main__":
    pass