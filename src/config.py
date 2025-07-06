INPUT_URL = "https://edition.cnn.com/2023/10/29/sport/nfl-week-8-how-to-watch-spt-intl/index.html" #"https://www.theguardian.com/science/brain-flapping/2014/nov/25/climate-change-is-an-obvious-myth-how-much-more-evidence-do-you-need" # must not be same as EXAMPLE_URL
# To use hardcoded text instead of loading it from URLs, set HARDCODED_TEXT to a non-empty string.
HARDCODED_TEXT = """"""


from other_models import NewspaperTextLoader
TEXT_LOADER = NewspaperTextLoader()

from langchain.text_splitter import RecursiveCharacterTextSplitter
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


from claim_extraction_models import OllamaClaimExtractor, NonDecontextualiser, NonFalsifiabilityChecker
CLAIM_EXTRACTOR = OllamaClaimExtractor(model_name='qwen:7b-chat')
DECONTEXTUALISE = False
if DECONTEXTUALISE:
    CLAIM_DECONTEXTUALISER = NonDecontextualiser()
CHECK_FALSIFIABILITY = False
if CHECK_FALSIFIABILITY:
    CLAIM_FALSIFIABILITY_CHECKER = NonFalsifiabilityChecker()


from other_models import SerpApiSourcesFinder
SEARCH_API = SerpApiSourcesFinder()


from other_models import NLIModel, HuggingFaceNLIModel
NLI_MODEL: NLIModel = HuggingFaceNLIModel(model_name='roberta-large-mnli')

EXTRACT_CLAIMS_FROM_SOURCE_URLS: bool = True # If False, will use an LLM for verification instead of an NLI model.
EMBED_SOURCE_CLAIMS: bool = True # If False, it will run the NLI model on each pair of source claim and input proposition, which is slower (O(n²)) but should be more accurate.


from langchain_huggingface import HuggingFaceEmbeddings
EMBEDDING_MODEL: HuggingFaceEmbeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
from langchain_community.vectorstores import FAISS
VECTORSTORE = FAISS


if __name__ == "__main__":
    pass