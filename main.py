import os
from dotenv import load_dotenv
from pathlib import Path

from enum import Enum
import pickle
from pydantic import BaseModel, Field, HttpUrl
from abc import ABC, abstractmethod

from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.output_parsers import PydanticOutputParser

import serpapi
import re

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

import requests
from bs4 import BeautifulSoup

import random
import newspaper


DEBUG_MODE: bool = True
DEMO_MODE: bool = False # if True, the program will use hardcoded text instead of loading it from URLs

BASE_DIR: Path = Path(__file__).parent
RELIABLE_SOURCES_PATH: Path = BASE_DIR / "data" / "reliable-sources.pkl"
SAMPLE_INPUT_PATH: Path = BASE_DIR / "data" / "sample-input.txt"
SAMPLE_SOURCE_TEXTS_PATH: Path = BASE_DIR / "data" / "sample-source-texts.txt"
CLAIM_EXTRACTION_TEMPLATE_PATH: Path = BASE_DIR / "data" / "claim-extraction-template.txt"

load_dotenv(BASE_DIR / ".env")
SERPAPI_KEY: str = os.getenv("SERPAPI_KEY", "")

INPUT_URL = "https://edition.cnn.com/2023/10/29/sport/nfl-week-8-how-to-watch-spt-intl/index.html" #"https://www.theguardian.com/science/brain-flapping/2014/nov/25/climate-change-is-an-obvious-myth-how-much-more-evidence-do-you-need" # must not be same as EXAMPLE_URL
EXAMPLE_URL = "https://example.com"


class Relation(Enum):
    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"

class Contradiction(BaseModel):
    claim: str
    source_url: HttpUrl
    source_claim: str

class Claims(BaseModel):
    claims: list[str] = Field(description="List of claims extracted from the text.")

class ClaimExtractor(ABC):
    @abstractmethod
    def extract_claims(self, text: str) -> list[str]:
        pass

class SourcesFinder(ABC):
    def __init__(self) -> None:

        if not RELIABLE_SOURCES_PATH.exists():
            raise FileNotFoundError(f"Reliable sources file not found at {RELIABLE_SOURCES_PATH}. Please provide a valid file.")
        
        with open(RELIABLE_SOURCES_PATH, "rb") as f:
            self.reliable_sources_regex: re.Pattern[str] = pickle.load(f)

    @abstractmethod
    def find_sources(self, claim: str) -> list[str]:
        pass

class NLIModel(ABC):
    @abstractmethod
    def recognise_textual_entailment(self, premise: str, hypothesis: str) -> Relation:
        pass


class OllamaClaimExtractor(ClaimExtractor):
    def __init__(self, model_name: str) -> None:
        with open(CLAIM_EXTRACTION_TEMPLATE_PATH, "r", encoding="utf-8") as f:
            CLAIM_EXTRACTION_TEMPLATE: str = f.read()
        prompt = PromptTemplate.from_template(CLAIM_EXTRACTION_TEMPLATE)

        llm = OllamaLLM(model=model_name, temperature=0.0)
        self.parser = PydanticOutputParser(pydantic_object=Claims)
        self.chain = prompt | llm | self.parser

    def extract_claims(self, text: str) -> list[str]:

        if text.strip() == "":
            if DEBUG_MODE:
                print("=> No text provided for claim extraction. Returning empty list.")
            return []

        claims: list[str] = self.chain.invoke({"text": text, "format_instructions": self.parser.get_format_instructions()}).claims

        # Cleaning up claims
        dont_start_with = ['-', '- ', 'The ', 'speaker ', 'text ', 'post ', 'article ', 'blog post ', 'blog article ', 'blog post article ']
        for substring in dont_start_with:
            if all(claim.startswith(substring) for claim in claims):
                claims = [claim[len(substring):] for claim in claims]
        
        if all(claim.endswith(".") for claim in claims):
            claims = [claim[:-1] for claim in claims]
        
        claims = [claim.strip() for claim in claims if claim.strip()]

        if DEBUG_MODE:
            print(f"\n=> Cleaned claims from the text:\n")
            print_list(claims)
        
        return claims
    
class SerpApiSourcesFinder(SourcesFinder):
    def find_sources(self, claim: str) -> list[str]:
        params = {
            "engine": "google_light",
            "q": claim,
            "api_key": SERPAPI_KEY,
        }
        search = serpapi.search(params)

        urls = []
        for result in search.get("organic_results", []):
            urls.append(result.get("link"))

        filtered_urls = [url for url in urls if self.reliable_sources_regex.match(url)]

        if DEBUG_MODE:
            if not filtered_urls:
                print(f"\n=> No reliable sources found for this claim: {claim}\nHere are some unreliable ones instead.")
                print_list(urls)
            else:
                print(f"\n=> Found {len(filtered_urls)} reliable sources for claim: {claim}")
                print_list(filtered_urls)
        
        return filtered_urls

class DemoSourcesFinder(SourcesFinder):
    def find_sources(self, claim: str) -> list[str]:
        return [EXAMPLE_URL] # Always the same

class SearXNGSourcesFinder(SourcesFinder):
    def find_sources(self, claim: str) -> list[str]:
        raise NotImplementedError("SearXNGSourcesFinder is not implemented yet.")

class HuggingFaceNLIModel(NLIModel):
    def __init__(self, model_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def recognise_textual_entailment(self, premise: str, hypothesis: str) -> Relation:
        inputs = self.tokenizer.encode_plus(premise, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1)
        return (Relation.CONTRADICTION, Relation.NEUTRAL, Relation.ENTAILMENT)[probs.argmax().item()]


def load_text(url: str) -> str:
    # Yeah this is not going to work. Kinda expected
    # soup = BeautifulSoup(requests.get(url).text, "html.parser")
    # paragraphs = soup.find_all("p")
    # text = "\n".join([p.get_text() for p in paragraphs])
    # if DEBUG_MODE:
    #     print(f"\nLoaded text from {url}.\nFirst 500 characters:")
    #     print(text[:500])  # Print first 500 characters for debugging
    #     print()
    # return text.strip()

    # choices = ["Global warming is a hoax, the earth is flat, and I want to kill myself.",
    #            "The earth is obviously not flat bruh, it's a cube",
    #            "Of course jet fuel can melt steel beams; they inject all sorts of contrail-chemicals in there",
    #            "They used to say that the earth was flat, and now they say it's round?? What's next, they'll say it's a dinosaur?",
    #            "9/11 is fake because jet fuel can't melt steel beams",
    #            "If global warming is true, why does snow exist? take that LIBERALS",
    #            "They say an evil dictator is taking over the world who says the earth is flat and 9/11 was an inside job. Personally I have no interest in politics, so I don't really have an opinion on this."]
    # return random.choice(choices)

    if DEMO_MODE:
        if url == INPUT_URL:
            return SAMPLE_INPUT_PATH.read_text(encoding="utf-8")
        elif url == EXAMPLE_URL:
            return SAMPLE_SOURCE_TEXTS_PATH.read_text(encoding="utf-8") # TODO: INCORRECT. fix later
        else:
            raise ValueError(f"Unknown URL: {url}. Disable demo mode to use real URLs.")
    
    try:
        article = newspaper.article(url)
    except Exception as e:
        print(f"Error loading article from {url}: {e}\njust gonna pretend that didn't happen and return absolutely nothing")
        return ""
        
    text = article.text

    if DEBUG_MODE:
        print(f"\n=> Loaded text from {url}:")
        print('-' * 50)
        print(text[:500])  # Print first 500 characters for debugging
        print('-' * 50)

    return text

def print_list(lst: list[str]) -> None:
    if not lst:
        print("NO ITEMS FOUND")
        return
    for i, v in enumerate(lst, start=1):
        print(f"{i}. {v}")
    print()

def findings(claims: list[str], sources_urls: set[str], contradictions: list[Contradiction], unverified_veracities: list[int], source_claim_count: int) -> None:
    print("\nClaims made by the article:")
    print_list(claims)

    if not sources_urls:
        print("The article does not contain any claims that can be verified with reliable sources.")
        if DEBUG_MODE: print("or the claim extractor and source finder isn't/aren't working properly idk")
        return
    print(f"Sources used:")
    print_list(list(sources_urls))
    print(f"Source claims extracted: {source_claim_count}")

    if not contradictions:
        print("\nNo contradictions were found.")
    else:
        print("\nContradictions found:")
        for contradiction in contradictions:
            print(f"- Claim: {contradiction.claim}\n  Source URL: {contradiction.source_url}\n  Source Claim: {contradiction.source_claim}")
    
    if not unverified_veracities:
        print("\nAll claims were supported by at least one reliable source.")
    else:
        print("\nClaims with no entailments found:")
        for claim_index in unverified_veracities:
            print(f"- {claims[claim_index]}")

def main():
    # Load models
    claim_extractor: ClaimExtractor = OllamaClaimExtractor(model_name='qwen:7b-chat')
    if DEMO_MODE:
        sources_finder: SourcesFinder = DemoSourcesFinder()
    else:
        sources_finder: SourcesFinder = SerpApiSourcesFinder()
    claim_checker: NLIModel = HuggingFaceNLIModel(model_name='roberta-large-mnli')
    print("\n=> Models loaded successfully. Or maybe not idk let's see")

    contradictions: list[Contradiction] = []
    sources_urls = set()
    source_claim_count = 0
    source_url_claims: dict[str, list[str]] = {} # Cache claims for each source URL to avoid re-extraction

    # Main algorithm
    claims: list[str] = claim_extractor.extract_claims(load_text(INPUT_URL))
    unverified_veracities: list[int] = []

    for claim_index, claim in enumerate(claims):
        source_urls: list[str] = sources_finder.find_sources(claim)
        sources_urls.update(source_urls)
        
        for source_url in source_urls:

            # Check cache or just cache
            if source_url in source_url_claims:
                source_claims = source_url_claims[source_url]
            else:
                source_claims: list[str] = claim_extractor.extract_claims(load_text(source_url))
                source_url_claims[source_url] = source_claims
            
            verified_veracity = False

            for source_claim in source_claims:
                source_claim_count += 1

                relation = claim_checker.recognise_textual_entailment(source_claim, claim)
                if relation == relation.ENTAILMENT:
                    verified_veracity = True
                    if DEBUG_MODE:
                        print(f"\n=> Claim '{claim}' is supported by source '{source_url}': {source_claim}")
                    break
                elif relation == Relation.CONTRADICTION:
                    contradictions.append(Contradiction(claim=claim, source_url=HttpUrl(source_url), source_claim=source_claim))
                    verified_veracity = True
                    if DEBUG_MODE:
                        print(f"\n=> Claim '{claim}' is contradicted by source '{source_url}': {source_claim}")
                    break
            
            if verified_veracity:
                break

        else: # nobreak
            unverified_veracities.append(claim_index)


    findings(claims, sources_urls, contradictions, unverified_veracities, source_claim_count)


if __name__ == "__main__":
    main()