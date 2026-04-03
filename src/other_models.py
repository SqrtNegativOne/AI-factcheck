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

class NewsPleaseTextLoader(TextFromURLLoader):
    """news-please — structured news article extractor with broad site support.
    Extracts maintext, title, authors, and publish date from news URLs.
    Install: pip install news-please"""
    def __init__(self, include_title: bool = True) -> None:
        super().__init__()
        try:
            from newsplease import NewsPlease
        except ImportError:
            raise ImportError("Please install the news-please package: pip install news-please")
        self.NewsPlease = NewsPlease
        self.include_title = include_title

    def load_text(self, url: str) -> str:
        article = self.NewsPlease.from_url(url)
        if article is None:
            raise ValueError(f"news-please failed to fetch article from {url}.")
        text = article.maintext
        if not text:
            raise ValueError(f"news-please extracted no text from {url}.")
        if self.include_title and article.title:
            text = f"{article.title}\n\n{text}"
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
    """Serper.dev — Google-quality results. ~2,500 free searches/month, then $0.30/1K.
    Requires SERPER_KEY in .env."""
    def __init__(self) -> None:
        super().__init__()
        import requests
        self.requests = requests
        from utils import SERPER_KEY
        if not SERPER_KEY:
            raise ValueError("SERPER_KEY is not set. Get one at https://serper.dev")
        self.api_key = SERPER_KEY

    def find_sources(self, claim: str) -> list[HttpUrl]:
        response = self.requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
            json={"q": claim, "num": 10},
            timeout=10,
        )
        response.raise_for_status()
        urls: list[HttpUrl] = []
        for result in response.json().get("organic", []):
            url_str: str = result.get("link", "")
            if url_str and self.reliable_sources_regex.match(url_str):
                urls.append(HttpUrl(url_str))
        return urls


class DuckDuckGoSourcesFinder(SourcesFinder):
    """Free DuckDuckGo search — no API key required.
    May rate-limit under heavy use; good for development and low-volume production."""
    def __init__(self, max_results: int = 10) -> None:
        super().__init__()
        from duckduckgo_search import DDGS
        self.ddgs = DDGS()
        self.max_results = max_results

    def find_sources(self, claim: str) -> list[HttpUrl]:
        urls: list[HttpUrl] = []
        try:
            results = self.ddgs.text(claim, max_results=self.max_results)
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed for claim '{claim[:60]}...': {e}")
            return []
        for result in results:
            url_str: str = result.get("href", "")
            if url_str and self.reliable_sources_regex.match(url_str):
                urls.append(HttpUrl(url_str))
        return urls


class TavilySourcesFinder(SourcesFinder):
    """Tavily — LLM-optimised search that returns pre-extracted content.
    1,000 free searches/month. Requires TAVILY_KEY in .env."""
    def __init__(self, max_results: int = 10) -> None:
        super().__init__()
        from tavily import TavilyClient
        from utils import TAVILY_KEY
        if not TAVILY_KEY:
            raise ValueError("TAVILY_KEY is not set. Get one at https://tavily.com")
        self.client = TavilyClient(api_key=TAVILY_KEY)
        self.max_results = max_results

    def find_sources(self, claim: str) -> list[HttpUrl]:
        response = self.client.search(claim, max_results=self.max_results)
        urls: list[HttpUrl] = []
        for result in response.get("results", []):
            url_str: str = result.get("url", "")
            if url_str and self.reliable_sources_regex.match(url_str):
                urls.append(HttpUrl(url_str))
        return urls



class NLIModel(ABC):
    @abstractmethod
    def recognise_textual_entailment(self, premise: str, hypothesis: str) -> Relation:
        pass

    def recognise_textual_entailment_batch(self, premises: list[str], hypothesis: str) -> list[Relation]:
        """Batch version — defaults to sequential calls. Override for GPU efficiency."""
        return [self.recognise_textual_entailment(p, hypothesis) for p in premises]

class HuggingFaceNLIModel(NLIModel):
    """NLI model loaded from HuggingFace.

    Good choices (all non-gated):
    - 'cross-encoder/nli-deberta-v3-base'      ~86M params, beats roberta-large on MNLI
    - 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'  304M, SOTA on ANLI
    - 'roberta-large-mnli'                      355M, solid baseline
    """
    def __init__(self, model_name: str) -> None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Build label map from model config so any model works regardless of label order
        id2label: dict = self.model.config.id2label
        self._label_map: list[Relation] = []
        for i in range(len(id2label)):
            raw = id2label[i].lower()
            if "entail" in raw:
                self._label_map.append(Relation.ENTAILMENT)
            elif "contradict" in raw:
                self._label_map.append(Relation.CONTRADICTION)
            else:
                self._label_map.append(Relation.NEUTRAL)
        logger.info(f"NLI label map for {model_name}: {[r.value for r in self._label_map]}")

    def recognise_textual_entailment(self, premise: str, hypothesis: str) -> Relation:
        import torch
        import torch.nn.functional as F
        device = next(self.model.parameters()).device
        inputs = self.tokenizer.encode_plus(premise, hypothesis, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1)
        return self._label_map[int(probs.argmax().item())]

    def recognise_textual_entailment_batch(self, premises: list[str], hypothesis: str) -> list[Relation]:
        import torch
        import torch.nn.functional as F
        if not premises:
            return []
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            premises,
            [hypothesis] * len(premises),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1)
        return [self._label_map[int(p.argmax().item())] for p in probs]


if __name__ == "__main__":
    logging.error("This module is not intended to be run directly. Please import it in your main application.")