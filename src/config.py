"""
Swapstation for AI Fact-Check
Swap out models and configurations and algorithm styles here.

main.py imports everything from this file, so only keep constants. Other stuff goes in utils.py.
"""



from pydantic import HttpUrl
#"https://www.theguardian.com/science/brain-flapping/2014/nov/25/climate-change-is-an-obvious-myth-how-much-more-evidence-do-you-need"
INPUT_URL: HttpUrl = HttpUrl("https://edition.cnn.com/2023/10/29/sport/nfl-week-8-how-to-watch-spt-intl/index.html")
# To use hardcoded text instead of loading it from URLs, set HARDCODED_TEXT to a non-empty string.
HARDCODED_TEXT = "" #"""What I'm suggesting is we have a sort of an eco-evangelical hysteria going on and it leads me to almost wonder if we are becoming a nation of environmental hypochondriacs that are willing to use the power of the state to impose enormous restrictions on the rights and the comforts of, and incomes of individuals who serve essentially a paranoia, a phobia, that has very little fact evidence in fact. Now these are observations that are popular to make because right now it's almost taken as an article of faith that this crisis is real. Let me say I take it as an article of faith if the Lord God almighty made the heavens and the Earth, and He made them to His satisfaction and it is quite pretentious of we little weaklings here on earth to think that, that we are going to destroy God's creation."""



from other_models import TextFromURLLoader, TrafilaturaTextLoader
TEXT_LOADER: TextFromURLLoader = TrafilaturaTextLoader()

from langchain.text_splitter import RecursiveCharacterTextSplitter
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)



# ── Claim extraction ────────────────────────────────────────────────────────────
# Uncomment ONE of the following:

# API-based (fast, high quality — needs API key in .env):
from claim_extraction_models import ClaimExtractor, AnthropicClaimExtractor
CLAIM_EXTRACTOR: ClaimExtractor = AnthropicClaimExtractor(model_name="claude-haiku-4-5-20251001")

# from claim_extraction_models import ClaimExtractor, OpenAIClaimExtractor
# CLAIM_EXTRACTOR: ClaimExtractor = OpenAIClaimExtractor(model_name="gpt-4o-mini")

# Local via Ollama (free, no key needed — requires `ollama serve` + model pulled):
# from claim_extraction_models import ClaimExtractor, OllamaClaimExtractor
# CLAIM_EXTRACTOR: ClaimExtractor = OllamaClaimExtractor(model_name="qwen2.5:7b")

# Local HuggingFace — gated, needs HF_KEY and Google ToS acceptance:
# from claim_extraction_models import ClaimExtractor, Gemma_APS_Claim_Extractor
# CLAIM_EXTRACTOR: ClaimExtractor = Gemma_APS_Claim_Extractor()



DECONTEXTUALISE = False
if DECONTEXTUALISE:
    from claim_extraction_models import Decontextualiser, NonDecontextualiser
    CLAIM_DECONTEXTUALISER: Decontextualiser = NonDecontextualiser()

CHECK_FALSIFIABILITY = False
if CHECK_FALSIFIABILITY:
    from claim_extraction_models import FalsifiabilityChecker, NonFalsifiabilityChecker
    CLAIM_FALSIFIABILITY_CHECKER: FalsifiabilityChecker = NonFalsifiabilityChecker()



# ── Search / source finding ─────────────────────────────────────────────────────
# Uncomment ONE of the following:

# Free, no key needed (may rate-limit under heavy use):
from other_models import SourcesFinder, DuckDuckGoSourcesFinder
SEARCH_API: SourcesFinder = DuckDuckGoSourcesFinder(max_results=10)

# Google-quality, 2,500 free/month then $0.30/1K (needs SERPER_KEY in .env):
# from other_models import SourcesFinder, SerperAPISourcesFinder
# SEARCH_API: SourcesFinder = SerperAPISourcesFinder()

# LLM-optimised, returns pre-extracted content, 1,000 free/month (needs TAVILY_KEY in .env):
# from other_models import SourcesFinder, TavilySourcesFinder
# SEARCH_API: SourcesFinder = TavilySourcesFinder()

# Original SerpAPI (100 free/month, needs SERPAPI_KEY in .env):
# from other_models import SourcesFinder, SerpApiSourcesFinder
# SEARCH_API: SourcesFinder = SerpApiSourcesFinder()



# ── NLI model ───────────────────────────────────────────────────────────────────
# Uncomment ONE of the following (all non-gated, no HF_KEY required):

from other_models import NLIModel, HuggingFaceNLIModel
# Best size/accuracy tradeoff — ~86M params, beats roberta-large-mnli on MNLI:
NLI_MODEL: NLIModel = HuggingFaceNLIModel(model_name='cross-encoder/nli-deberta-v3-base')

# SOTA on adversarial NLI (ANLI) — 304M params, trained on MNLI+FEVER+ANLI+Ling+WANLI:
# NLI_MODEL: NLIModel = HuggingFaceNLIModel(model_name='MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli')

# Original baseline — 355M params, solid but larger and weaker than DeBERTa:
# NLI_MODEL: NLIModel = HuggingFaceNLIModel(model_name='roberta-large-mnli')

EXTRACT_CLAIMS_FROM_SOURCE_URLS: bool = True # If False, will use an LLM for verification instead of an NLI model.
EMBED_SOURCE_CLAIMS: bool = True # If False, it will run the NLI model on each pair of source claim and input proposition, which is slower (O(n²)) but should be more accurate.



from langchain_huggingface import HuggingFaceEmbeddings
EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
from langchain_community.vectorstores import FAISS
VECTORSTORE = FAISS



if __name__ == "__main__":
    print("This is a configuration file. Import it in main.py to use the constants defined here.")
