import claim_extraction_models
import other_models
from langchain.text_splitter import RecursiveCharacterTextSplitter

TEXT_LOADER = other_models.NewspaperTextLoader()
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

CLAIM_EXTRACTOR = claim_extraction_models.OllamaClaimExtractor(model_name='qwen:7b-chat')
CLAIM_DECONTEXTUALISER = claim_extraction_models.NonDecontextualiser()
CLAIM_FALSIFIABILITY_CHECKER = claim_extraction_models.NonFalsifiabilityChecker()

SOURCES_FINDER = other_models.DemoSourcesFinder() if other_models.DEMO_MODE else other_models.SerpApiSourcesFinder()

NLI_MODEL: other_models.NLIModel = other_models.HuggingFaceNLIModel(model_name='roberta-large-mnli')