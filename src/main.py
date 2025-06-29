from pydantic import HttpUrl
import pandas as pd
import tldextract

from utils import *
import claim_extraction_models
import other_models

from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_domain(url):
    extracted = tldextract.extract(url)
    return f"{extracted.domain}.{extracted.suffix}".lower()

def lookup_source_bias(news_url):
    domain = extract_domain(news_url)
    bias_df = pd.read_csv(MBFC_PATH, encoding='utf-8')
    match = bias_df[bias_df['url'].str.contains(domain, na=False, case=False)]
    if match.empty:
        print(f"\n=> No bias information found for domain: {domain}")
    row = match.iloc[0]
    results = {
        'domain': domain,
        'bias': row.get('bias_rating', 'N/A'),
        'factual_reporting': row.get('factual_reporting_rating', 'N/A'),
        'type': 'N/A',
        'notes': f"Matched from: {row.get('site_name', '')}"
    }
    for k, v in results.items():
        print(f"{k.capitalize()}: {v}")

def generate_atomic_claims_from_url(url: str) -> list[str]:
    text_loader = other_models.NewspaperTextLoader()
    full_text: str = text_loader.load_text(url)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_text(full_text)

    claim_extractor: claim_extraction_models.ClaimExtractor = claim_extraction_models.OllamaClaimExtractor(model_name='qwen:7b-chat')
    claims: list[str] = []
    for chunk in text_chunks:
        if DEBUG_MODE:
            print(f"\n=> Processing text chunk:\n{chunk}\n")
        claims.extend(claim_extractor.extract_claims(chunk))
    
    claim_decontextualiser = claim_extraction_models.NonDecontextualiser()
    context_before, context_after = 5, 1
    for i, claim in enumerate(claims):
        before = "".join(claims[max(0, i - context_before):i])
        after = "".join(claims[i + 1:i + context_after + 1]) if i + 1 < len(claims) else ""
        claims[i] = claim_decontextualiser.decontextualise(before, claim, after)
    
    claim_falsifiability_checker = claim_extraction_models.NonFalsifiabilityChecker()
    claims = [claim for claim in claims if claim_falsifiability_checker.is_falsifiable(claim)]

    return claims

def verify_atomic_claim(atomic_claim: str, text_loader: other_models.TextFromURLLoader) -> Relation:
    # https://github.com/mbzuai-nlp/fire/blob/main/eval/fire/verify_atomic_claim.py

    if DEMO_MODE:
        sources_finder: other_models.SourcesFinder = other_models.DemoSourcesFinder()
    else:
        sources_finder: other_models.SourcesFinder = other_models.SerpApiSourcesFinder()
    claim_checker: other_models.NLIModel = other_models.HuggingFaceNLIModel(model_name='roberta-large-mnli')

    source_urls: list[str] = sources_finder.find_sources(atomic_claim)
    sources_urls.update(source_urls)
    
    for source_url in source_urls:

        # Check cache or just cache
        if source_url in source_url_claims:
            source_claims = source_url_claims[source_url]
        else:
            source_claims: list[str] = claim_extractor.extract_claims(text_loader.load_text(source_url))
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

    return Relation.NEUTRAL  # Placeholder for actual implementation

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
    print("AI Fact-Check v0.1")
    if not DEMO_MODE:
        print(f"Input URL: {INPUT_URL}")
        lookup_source_bias(INPUT_URL)

    # Main algorithm
    contradictions: list[Contradiction] = []
    sources_urls = set()
    source_claim_count = 0
    source_url_claims: dict[str, list[str]] = {} # Cache claims for each source URL to avoid re-extraction
    unverified_veracities: list[int] = []

    claims = generate_atomic_claims_from_url(INPUT_URL)

    for claim_index, claim in enumerate(claims):
        status = verify_atomic_claim(claim, text_loader)

    findings(claims, sources_urls, contradictions, unverified_veracities, source_claim_count)


if __name__ == "__main__":
    main()