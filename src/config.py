import claim_extraction_models
import other_models
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

TEXT_LOADER = other_models.NewspaperTextLoader()
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

CLAIM_EXTRACTOR = claim_extraction_models.OllamaClaimExtractor(model_name='qwen:7b-chat')
CLAIM_DECONTEXTUALISER = claim_extraction_models.NonDecontextualiser()
CLAIM_FALSIFIABILITY_CHECKER = claim_extraction_models.NonFalsifiabilityChecker()

SEARCH_API = other_models.SerpApiSourcesFinder()

NLI_MODEL: other_models.NLIModel = other_models.HuggingFaceNLIModel(model_name='roberta-large-mnli')

EXTRACT_CLAIMS_FROM_SOURCE_URLS: bool = True # If False, will use an LLM for verification instead of an NLI model.
EMBED_SOURCE_CLAIMS: bool = True # If False, it will run the NLI model on each pair of source claim and input proposition, which is slower (O(n²)) but should be more accurate.

EMBEDDINGS: HuggingFaceEmbeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
VECTORSTORE = FAISS