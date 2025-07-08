"""
Swapstation for AI Fact-Check
Swap out models and configurations and algorithm styles here.

main.py imports everything from this file, so only keep constants. Other stuff goes in utils.py.
"""


from pydantic import HttpUrl
#"https://www.theguardian.com/science/brain-flapping/2014/nov/25/climate-change-is-an-obvious-myth-how-much-more-evidence-do-you-need"
INPUT_URL: HttpUrl = HttpUrl("https://edition.cnn.com/2023/10/29/sport/nfl-week-8-how-to-watch-spt-intl/index.html")
# To use hardcoded text instead of loading it from URLs, set HARDCODED_TEXT to a non-empty string.
HARDCODED_TEXT = """What I’m suggesting is we have a sort of an eco-evangelical hysteria going on and it leads me to almost wonder if we are becoming a nation of environmental hypochondriacs that are willing to use the power of the state to impose enormous restrictions on the rights and the comforts of, and incomes of individuals who serve essentially a paranoia, a phobia, that has very little fact evidence in fact. Now these are observations that are popular to make because right now it's almost taken as an article of faith that this crisis is real. Let me say I take it as an article of faith if the Lord God almighty made the heavens and the Earth, and He made them to His satisfaction and it is quite pretentious of we little weaklings here on earth to think that, that we are going to destroy God’s creation."""


from other_models import TextFromURLLoader, NewspaperTextLoader
TEXT_LOADER: TextFromURLLoader = NewspaperTextLoader()

from langchain.text_splitter import RecursiveCharacterTextSplitter
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


from claim_extraction_models import ClaimExtractor, Gemma_APS_Claim_Extractor
CLAIM_EXTRACTOR: ClaimExtractor = Gemma_APS_Claim_Extractor()
DECONTEXTUALISE = False
if DECONTEXTUALISE:
    from claim_extraction_models import Decontextualiser, NonDecontextualiser
    CLAIM_DECONTEXTUALISER: Decontextualiser = NonDecontextualiser()
CHECK_FALSIFIABILITY = False
if CHECK_FALSIFIABILITY:
    from claim_extraction_models import FalsifiabilityChecker, NonFalsifiabilityChecker
    CLAIM_FALSIFIABILITY_CHECKER: FalsifiabilityChecker = NonFalsifiabilityChecker()


from other_models import SourcesFinder, SerpApiSourcesFinder
SEARCH_API: SourcesFinder = SerpApiSourcesFinder()


from other_models import NLIModel, HuggingFaceNLIModel
NLI_MODEL: NLIModel = HuggingFaceNLIModel(model_name='roberta-large-mnli')

EXTRACT_CLAIMS_FROM_SOURCE_URLS: bool = True # If False, will use an LLM for verification instead of an NLI model.
EMBED_SOURCE_CLAIMS: bool = True # If False, it will run the NLI model on each pair of source claim and input proposition, which is slower (O(n²)) but should be more accurate.


from langchain_huggingface import HuggingFaceEmbeddings
EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
from langchain_community.vectorstores import FAISS
VECTORSTORE = FAISS


if __name__ == "__main__":
    pass