"""
Run the fact-check pipeline end-to-end with fully stubbed components.
No API keys, no GPU, no network required.

Usage:
    cd src && python run_dummy.py
"""

import sys
import types
import logging
from pydantic import HttpUrl

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# ── Dummy implementations ────────────────────────────────────────────────────

class DummyClaimExtractor:
    """Returns a fixed set of claims regardless of input text."""
    def extract_claims(self, text: str) -> list[str]:
        return [
            "The Apollo 11 moon landing occurred in 1969.",
            "Neil Armstrong was the first person to walk on the Moon.",
            "Vaccines do not cause autism.",
            "Global average temperatures have risen since the industrial revolution.",
        ]


class DummyNLIModel:
    """Returns NEUTRAL for all pairs (no contradiction or entailment detected)."""
    def recognise_textual_entailment(self, premise: str, hypothesis: str):
        from utils import Relation
        return Relation.NEUTRAL

    def recognise_textual_entailment_batch(self, premises: list[str], hypothesis: str):
        from utils import Relation
        return [Relation.NEUTRAL] * len(premises)


class DummySourcesFinder:
    """Returns an empty list — skips all network/NLI steps."""
    def find_sources(self, claim: str) -> list[HttpUrl]:
        return []


class DummyTextLoader:
    """Returns hardcoded article text without fetching any URL."""
    def load_text(self, url: str) -> str:
        return (
            "NASA confirmed that the Apollo 11 mission landed on the Moon on July 20, 1969. "
            "Neil Armstrong became the first human to walk on the lunar surface. "
            "Multiple scientific studies have found no link between vaccines and autism. "
            "Climate scientists report that global temperatures have increased by approximately "
            "1.1°C above pre-industrial levels."
        )


# ── Fake config module ───────────────────────────────────────────────────────

cfg = types.ModuleType("config")

cfg.INPUT_URL = HttpUrl("https://example.com/dummy-article")

# Non-empty string: skips URL loading and source-bias lookup in main()
cfg.HARDCODED_TEXT = DummyTextLoader().load_text("")

from langchain_text_splitters import RecursiveCharacterTextSplitter
cfg.TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

cfg.TEXT_LOADER       = DummyTextLoader()
cfg.CLAIM_EXTRACTOR   = DummyClaimExtractor()
cfg.DECONTEXTUALISE   = False
cfg.CHECK_FALSIFIABILITY = False
cfg.SEARCH_API        = DummySourcesFinder()
cfg.NLI_MODEL         = DummyNLIModel()

# Embedding model / vectorstore are never reached when SEARCH_API returns []
cfg.EMBEDDING_MODEL   = None
cfg.VECTORSTORE       = None

cfg.EXTRACT_CLAIMS_FROM_SOURCE_URLS = True
cfg.EMBED_SOURCE_CLAIMS             = True

sys.modules["config"] = cfg

# ── Run ──────────────────────────────────────────────────────────────────────

import main
main.main()
