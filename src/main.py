from pydantic import HttpUrl, BaseModel, Field
import pandas as pd
import tldextract

from utils import *
from config import *

from langchain.docstore.document import Document

from typing import Any

import logging

logging.basicConfig(level=logging.INFO)

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

class Proposition(BaseModel):
    claim: str = Field(..., description="The text of the proposition")
    source: HttpUrl = Field(..., description="The source URL of the proposition")
    chunk: str = Field(..., description="The text chunk from which the proposition was extracted") # should be fine memory-wise ∵ python stores strings as pointers to the same objects

def generate_atomic_claims_from_url(url: str) -> list[Proposition]:
    full_text: str = TEXT_LOADER.load_text(url)
    text_chunks = TEXT_SPLITTER.split_text(full_text)

    proposition_objects: list[Proposition] = []

    # Get raw claims
    for i, chunk in enumerate(text_chunks):
        if DEBUG_MODE:
            logging.debug(f"\n=> Processing text chunk {i}:\n{chunk}\n")
        for claim in CLAIM_EXTRACTOR.extract_claims(chunk):
            if DEBUG_MODE:
                logging.debug(f"Extracted claim: {claim}")
            proposition_objects.append(Proposition(claim=claim, source=HttpUrl(url), chunk=chunk))

    # Decontextualise
    for i, claim in enumerate(proposition_objects):
        before = "".join(c.claim for c in proposition_objects[max(0, i-5):i])
        after  = "".join(c.claim for c in proposition_objects[i+1:i+2])
        new_claim = CLAIM_DECONTEXTUALISER.decontextualise(before, claim.claim, after)

        if DEBUG_MODE:
            logging.debug(f"\n=> Decontextualised claim {i}:\n{claim.claim}\nTo:\n{new_claim}\n")

        proposition_objects[i] = Proposition(
            claim=new_claim,
            source=claim.source,
            chunk=claim.chunk
        )

    # Check falsifiability
    proposition_objects = [c for c in proposition_objects if CLAIM_FALSIFIABILITY_CHECKER.is_falsifiable(c.claim)]
    return proposition_objects

class PairRelation(BaseModel):
    proposition_to_check_object: Proposition = Field(..., description="Claim made by the input article")
    source_proposition_object: Proposition = Field(..., description="Claim extracted from the source")
    relation: Relation = Field(..., description="Relation between the two claims (entailment, contradiction, or neutral)")

def verify_atomic_claim(
        proposition_to_check_object: Proposition,
        source_url_to_proposition_objects_cache: dict[str, list[Proposition]]
) -> tuple[PairRelation, set[str]]: # pair relation, updated source_url_claims_cache
    # https://github.com/mbzuai-nlp/fire/blob/main/eval/fire/verify_atomic_claim.py

    source_claim_count = 0

    search_results: list[str] = SEARCH_API.find_sources(proposition_to_check_object.claim)

    for source_url in search_results:

        # Check cache or just cache
        if source_url in source_url_to_proposition_objects_cache:
            source_claims = source_url_to_proposition_objects_cache[source_url]
        else:
            source_claims: list[Proposition] = generate_atomic_claims_from_url(source_url)
            source_url_to_proposition_objects_cache[source_url] = source_claims

        source_claim_count += len(source_claims)
        docs = [Document(page_content=claim.claim, metadata={"source": claim.source}) for claim in source_claims]
        propositions_vectorstore = VECTORSTORE.from_documents(
            docs,
            EMBEDDINGS,
            collection_name=source_url,
            persist_directory=f"faiss/{extract_domain(source_url)}"
        )

        if DEBUG_MODE:
            logging.debug(f"\n=> Found {len(source_claims)} claims in source {source_url}:\n")
            for claim in source_claims:
                logging.debug(f"- {claim.claim}")

        check_against = propositions_vectorstore.similarity_search_with_score(proposition_to_check_object.claim, k=5)

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

    claims: list[Proposition] = generate_atomic_claims_from_url(INPUT_URL)

    for claim_index, claim in enumerate(claims):
        status = verify_atomic_claim(claim, source_url_claims)

    findings(claims, sources_urls, contradictions, unverified_veracities, source_claim_count)


if __name__ == "__main__":
    main()