import claim_extraction_models
import other_models
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

INPUT_URL = "https://edition.cnn.com/2023/10/29/sport/nfl-week-8-how-to-watch-spt-intl/index.html" #"https://www.theguardian.com/science/brain-flapping/2014/nov/25/climate-change-is-an-obvious-myth-how-much-more-evidence-do-you-need" # must not be same as EXAMPLE_URL

TEXT_LOADER = other_models.NewspaperTextLoader()
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

CLAIM_EXTRACTOR = claim_extraction_models.OllamaClaimExtractor(model_name='qwen:7b-chat')
CLAIM_DECONTEXTUALISER = claim_extraction_models.NonDecontextualiser()
CLAIM_FALSIFIABILITY_CHECKER = claim_extraction_models.NonFalsifiabilityChecker()

SEARCH_API = other_models.SerpApiSourcesFinder()

NLI_MODEL: other_models.NLIModel = other_models.HuggingFaceNLIModel(model_name='roberta-large-mnli')

EXTRACT_CLAIMS_FROM_SOURCE_URLS: bool = True # If False, will use an LLM for verification instead of an NLI model.
EMBED_SOURCE_CLAIMS: bool = True # If False, it will run the NLI model on each pair of source claim and input proposition, which is slower (O(n²)) but should be more accurate.

EMBEDDING_MODEL: HuggingFaceEmbeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
VECTORSTORE = FAISS