# Get paths
from pathlib import Path
BASE_DIR: Path = Path(__file__).parent
RELIABLE_SOURCES_PATH: Path = BASE_DIR / "data" / "reliable-sources.pkl"
SAMPLE_INPUT_PATH: Path = BASE_DIR / "data" / "sample-input.txt"
SAMPLE_SOURCE_TEXTS_PATH: Path = BASE_DIR / "data" / "sample-source-texts.txt"
MBFC_PATH: Path = BASE_DIR / "data" / "mbfc.csv"

# Load environment variables
import os
from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")
SERPAPI_KEY: str = os.getenv("SERPAPI_KEY", "")
HF_KEY: str = os.getenv("HF_KEY", "")

# Log in to Hugging Face Hub
from huggingface_hub import login
if HF_KEY == "":
    raise ValueError("HF_KEY not found in environment variables")
login(token=HF_KEY)

# Used by NLI models and main.py, therefore stored here
from enum import Enum
class Relation(Enum):
    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


ASCII_ART = r"""

           _____   ______      _____ _______ _____ _    _ ______ _____ _  __
     /\   |_   _| |  ____/\   / ____|__   __/ ____| |  | |  ____/ ____| |/ /
    /  \    | |   | |__ /  \ | |       | | | |    | |__| | |__ | |    | ' / 
   / /\ \   | |   |  __/ /\ \| |       | | | |    |  __  |  __|| |    |  <  
  / ____ \ _| |_  | | / ____ \ |____   | | | |____| |  | | |___| |____| . \ 
 /_/    \_\_____| |_|/_/    \_\_____|  |_|  \_____|_|  |_|______\_____|_|\_\
                                                                            
                                                                            
"""

# This file is used to store configuration and utility functions that are shared across the project.
# Not supposed to be run directly, but can be imported by other modules.
if __name__ == "__main__":
    pass