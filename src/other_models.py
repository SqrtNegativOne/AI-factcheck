from abc import ABC, abstractmethod
import pickle

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

from utils import *



def debug_text(text: str) -> None:
    print(f"\n=> Loaded text:")
    print('-' * 50)
    print(text[:500])  # Print first 500 characters for debugging
    print('-' * 50)

class TextFromURLLoader(ABC):
    @abstractmethod
    def load_text(self, url: str) -> str:
        pass

class NewspaperTextLoader(TextFromURLLoader):
    def __init__(self) -> None:
        try:
            import newspaper
        except ImportError:
            raise ImportError("Please install the newspaper4k package")
        self.newspaper = newspaper

    def load_text(self, url: str) -> str:
        article = self.newspaper.Article(url)
        article.download()
        article.parse()
        text = article.text.strip()
        debug_text(text)
        return text

class BeautifulSoupTextLoader(TextFromURLLoader):
    def __init__(self) -> None:
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
        debug_text(text)
        return text

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



class SourcesFinder(ABC):
    def __init__(self) -> None:
        import re

        if not RELIABLE_SOURCES_PATH.exists():
            raise FileNotFoundError(f"Reliable sources file not found at {RELIABLE_SOURCES_PATH}. Please provide a valid file.")
        
        with open(RELIABLE_SOURCES_PATH, "rb") as f:
            self.reliable_sources_regex: re.Pattern[str] = pickle.load(f)

    @abstractmethod
    def find_sources(self, claim: str) -> list[str]:
        pass
    
class SerpApiSourcesFinder(SourcesFinder):
    def __init__(self) -> None:
        super().__init__()

        import serpapi
        self.serpapi = serpapi

    def find_sources(self, claim: str) -> list[str]:
        params = {
            "engine": "google_light",
            "q": claim,
            "api_key": SERPAPI_KEY,
        }
        search = self.serpapi.search(params)

        urls = []
        for result in search.get("organic_results", []):
            urls.append(result.get("link"))

        filtered_urls = [url for url in urls if self.reliable_sources_regex.match(url)]

        if not filtered_urls:
            print(f"\n=> No reliable sources found for this claim: {claim}\nHere are some unreliable ones instead.")
            print_list(urls)
        else:
            print(f"\n=> Found {len(filtered_urls)} reliable sources for claim: {claim}")
            print_list(filtered_urls)
        
        return filtered_urls

class DemoSourcesFinder(SourcesFinder):
    def find_sources(self, claim: str) -> list[str]:
        return []

class SearXNGSourcesFinder(SourcesFinder):
    def find_sources(self, claim: str) -> list[str]:
        raise NotImplementedError("SearXNGSourcesFinder is not implemented yet.")



class NLIModel(ABC):
    @abstractmethod
    def recognise_textual_entailment(self, premise: str, hypothesis: str) -> Relation:
        pass

class HuggingFaceNLIModel(NLIModel):
    def __init__(self, model_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def recognise_textual_entailment(self, premise: str, hypothesis: str) -> Relation:
        inputs = self.tokenizer.encode_plus(premise, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1)
        return [Relation.CONTRADICTION, Relation.NEUTRAL, Relation.ENTAILMENT][int(probs.argmax().item())]