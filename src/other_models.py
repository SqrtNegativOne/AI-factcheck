from abc import ABC, abstractmethod

from utils import Relation

from pydantic import HttpUrl

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class TextFromURLLoader(ABC):
    @staticmethod
    def show_text(text: str) -> None:
        # Use print instead of logging only
        print(f"\n=> First 500 characters of the loaded text:")
        print('-' * 50)
        print(text[:500] + '...')
        print('-' * 50)

    @abstractmethod
    def load_text(self, url: str) -> str:
        pass

class RandomTextLoader(TextFromURLLoader):
    def __init__(self) -> None:
        import random
        self.random = random
        
    def load_text(self, url: str) -> str:
        choices = ["Global warming is a hoax, the earth is flat, and I want to kill myself.",
            "The earth is obviously not flat bruh, it's a cube",
            "Of course jet fuel can melt steel beams; they inject all sorts of contrail-chemicals in there",
            "They used to say that the earth was flat, and now they say it's round?? What's next, they'll say it's a dinosaur?",
            "9/11 is fake because jet fuel can't melt steel beams",
            "If global warming is true, why does snow exist? take that LIBERALS",
            "They say an evil dictator is taking over the world who says the earth is flat and 9/11 was an inside job. Personally I have no interest in politics, so I don't really have an opinion on this."]
        return self.random.choice(choices)
    
class BeautifulSoupTextLoader(TextFromURLLoader):
    def __init__(self) -> None:
        super().__init__()
        try:
            from bs4 import BeautifulSoup
            import requests
        except ImportError:
            raise ImportError("Please install the beautifulsoup4 and requests packages")
        self.BeautifulSoup = BeautifulSoup
        self.requests = requests
        
    def load_text(self, url: str) -> str:
        response = self.requests.get(url)
        soup = self.BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        self.show_text(text)
        return text

class Newspaper4kTextLoader(TextFromURLLoader):
    def __init__(self) -> None:
        super().__init__()
        import newspaper
        self.newspaper = newspaper
        
        # When newspaper4k uses cloudscraper it gives an incomprehensible newspaper.exceptions.ArticleException.
        # import cloudscraper # Apparently used by newspaper4k?
        # self.cloudscraper = cloudscraper # ...but not really used by the code yet.

    def load_text(self, url: str) -> str:
        article = self.newspaper.Article(url)
        article.download()
        article.parse()
        text = article.text.strip()
        self.show_text(text)
        return text

class TrafilaturaTextLoader(TextFromURLLoader):
    def __init__(self) -> None:
        super().__init__()
        from trafilatura import fetch_url, extract
        self.fetch_url = fetch_url
        self.extract = extract

    def load_text(self, url: str) -> str:
        downloaded = self.fetch_url(url)
        if downloaded is None:
            raise ValueError(f"Failed to download content from {url}.")
        text = self.extract(downloaded, output_format="markdown", with_metadata=False, include_comments=False, include_tables=False, include_images=False)
        if text is None:
            raise ValueError(f"Failed to extract text from {url} after downloading its contents.")
        self.show_text(text)
        return text



class SourcesFinder(ABC):
    def __init__(self) -> None:
        import re
        import pickle
        from utils import RELIABLE_SOURCES_PATH

        if not RELIABLE_SOURCES_PATH.exists():
            raise FileNotFoundError(f"Reliable sources file not found at {RELIABLE_SOURCES_PATH}. Please provide a valid file.")
        
        with open(RELIABLE_SOURCES_PATH, "rb") as f:
            self.reliable_sources_regex: re.Pattern[str] = pickle.load(f)

    @abstractmethod
    def find_sources(self, claim: str) -> list[HttpUrl]:
        pass
    
class SerpApiSourcesFinder(SourcesFinder):
    def __init__(self) -> None:
        super().__init__()

        import serpapi
        self.serpapi = serpapi

        from utils import SERPAPI_KEY
        if not SERPAPI_KEY:
            raise ValueError("SERPAPI_KEY is not set. Please provide a valid API key.")
        self.serpapi_key = SERPAPI_KEY

    def find_sources(self, claim: str) -> list[HttpUrl]:

        params = {
            "engine": "google_light",
            "q": claim,
            "api_key": self.serpapi_key,
        }
        search = self.serpapi.search(params)

        urls: list[HttpUrl] = []
        for result in search.get("organic_results", []):
            url_str: str = result.get("link")
            if self.reliable_sources_regex.match(url_str):
                urls.append(HttpUrl(url_str))
        
        return urls

class DemoSourcesFinder(SourcesFinder):
    def find_sources(self, claim: str) -> list[HttpUrl]:
        return []

class SearXNGSourcesFinder(SourcesFinder):
    def find_sources(self, claim: str) -> list[HttpUrl]:
        raise NotImplementedError("SearXNGSourcesFinder is not implemented yet.")

class SerperAPISourcesFinder(SourcesFinder):
    def find_sources(self, claim: str) -> list[HttpUrl]:
        raise NotImplementedError("SerperAPISourcesFinder is not implemented yet.")



class NLIModel(ABC):
    @abstractmethod
    def recognise_textual_entailment(self, premise: str, hypothesis: str) -> Relation:
        pass

class HuggingFaceNLIModel(NLIModel):
    def __init__(self, model_name: str) -> None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def recognise_textual_entailment(self, premise: str, hypothesis: str) -> Relation:
        import torch
        import torch.nn.functional as F
        inputs = self.tokenizer.encode_plus(premise, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1)
        return [Relation.CONTRADICTION, Relation.NEUTRAL, Relation.ENTAILMENT][int(probs.argmax().item())]


if __name__ == "__main__":
    logging.error("This module is not intended to be run directly. Please import it in your main application.")