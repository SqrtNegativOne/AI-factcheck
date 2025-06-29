from pathlib import Path
import os
from dotenv import load_dotenv
import torch

DEBUG_MODE: bool = True

BASE_DIR: Path = Path(__file__).parent
RELIABLE_SOURCES_PATH: Path = BASE_DIR / "data" / "reliable-sources.pkl"
SAMPLE_INPUT_PATH: Path = BASE_DIR / "data" / "sample-input.txt"
SAMPLE_SOURCE_TEXTS_PATH: Path = BASE_DIR / "data" / "sample-source-texts.txt"
CLAIM_EXTRACTION_TEMPLATE_PATH: Path = BASE_DIR / "data" / "claim-extraction-template.txt"
MBFC_PATH = BASE_DIR / "data" / "mbfc.csv"

load_dotenv(BASE_DIR / ".env")
SERPAPI_KEY: str = os.getenv("SERPAPI_KEY", "")

DEMO_MODE: bool = False # if True, the program will use hardcoded text instead of loading it from URLs

INPUT_URL = "https://edition.cnn.com/2023/10/29/sport/nfl-week-8-how-to-watch-spt-intl/index.html" #"https://www.theguardian.com/science/brain-flapping/2014/nov/25/climate-change-is-an-obvious-myth-how-much-more-evidence-do-you-need" # must not be same as EXAMPLE_URL
EXAMPLE_URL = "https://example.com"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

from enum import Enum
from pydantic import BaseModel, HttpUrl

class Relation(Enum):
    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"

class Contradiction(BaseModel):
    claim: str
    source_url: HttpUrl
    source_claim: str

def print_list(lst: list[str]) -> None:
    if not lst:
        print("NO ITEMS FOUND")
        return
    for i, v in enumerate(lst, start=1):
        print(f"{i}. {v}")
    print()