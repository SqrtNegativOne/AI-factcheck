from pydantic import HttpUrl, BaseModel, Field
import pandas as pd
import tldextract
import concurrent.futures
import threading
import os

from utils import MBFC_PATH, Relation, ASCII_ART
from config import *

from langchain_core.documents import Document

import logging
logger = logging.getLogger(__name__)

# Serialize GPU model calls across threads
_GPU_LOCK = threading.Lock()

# Per-domain locks to prevent duplicate vectorstore builds
_domain_locks: dict[str, threading.Lock] = {}
_domain_locks_mutex = threading.Lock()

def _get_domain_lock(domain: str) -> threading.Lock:
    with _domain_locks_mutex:
        if domain not in _domain_locks:
            _domain_locks[domain] = threading.Lock()
        return _domain_locks[domain]


def extract_domain(url: HttpUrl) -> str:
    extracted = tldextract.extract(str(url))
    return f"{extracted.domain}.{extracted.suffix}".lower()

def lookup_source_bias(news_url: HttpUrl) -> None:
    domain: str = extract_domain(news_url)
    bias_df: pd.DataFrame = pd.read_csv(MBFC_PATH, encoding='utf-8')
    match: pd.DataFrame = bias_df[bias_df['url'].str.contains(domain, na=False, case=False)]
    if match.empty:
        logger.warning(f"\n=> No bias information found for domain: {domain}")
        return
    row: pd.Series = match.iloc[0]
    results: dict[str, str] = {
        'Domain': domain,
        'Bias': row.get('bias_rating', 'N/A'),
        'Factual Reporting': row.get('factual_reporting_rating', 'N/A'),
    }
    for k, v in results.items():
        print(f"{k}: {v}")

class Proposition(BaseModel):
    claim: str = Field(..., description="The text of the proposition")
    url: HttpUrl = Field(..., description="The source URL of the proposition")
    chunk: str = Field(..., description="The text chunk from which the proposition was extracted") # should be fine memory-wise ∵ python stores strings as pointers to the same objects

def generate_atomic_claims_from_url(url: HttpUrl) -> list[Proposition]:
    if HARDCODED_TEXT:
        logger.info(f"\n=> Using hardcoded text for URL: {str(url)}\n")
        full_text: str = HARDCODED_TEXT
    else:
        full_text: str = TEXT_LOADER.load_text(str(url))  # I/O — no GPU lock needed
    text_chunks: list[str] = TEXT_SPLITTER.split_text(full_text)

    proposition_objects: list[Proposition] = []

    # Get claims — GPU inference serialized per chunk so other threads can interleave I/O
    for i, chunk in enumerate(text_chunks):
        logger.info(f"\n=> Processing text chunk {i}:\n{chunk}\n")
        with _GPU_LOCK:
            claims = CLAIM_EXTRACTOR.extract_claims(chunk)
        for claim in claims:
            logger.info(f"Extracted claim: {claim}")
            proposition_objects.append(Proposition(claim=claim, url=url, chunk=chunk))

    # Decontextualise
    if DECONTEXTUALISE:
        for i, claim in enumerate(proposition_objects):
            before: str = "".join(c.claim for c in proposition_objects[max(0, i-5):i])
            after:  str = "".join(c.claim for c in proposition_objects[i+1:i+2])
            with _GPU_LOCK:
                new_claim: str = CLAIM_DECONTEXTUALISER.decontextualise(before, claim.claim, after)

            logger.info(f"\n=> Decontextualised claim {i}:\n{claim.claim}\nTo:\n{new_claim}\n")

            proposition_objects[i] = Proposition(
                claim=new_claim,
                url=claim.url,
                chunk=claim.chunk
            )

    # Check falsifiability
    if not CHECK_FALSIFIABILITY:
        return proposition_objects

    proposition_objects = [c for c in proposition_objects if CLAIM_FALSIFIABILITY_CHECKER.is_falsifiable(c.claim)]
    return proposition_objects

class PairRelation(BaseModel):
    proposition_to_check_object: Proposition = Field(..., description="Claim made by the input article")
    source_proposition_object: Proposition = Field(..., description="Claim extracted from the source")
    relation: Relation = Field(..., description="Relation between the two claims (entailment, contradiction, or neutral)")


def _build_vectorstore_for_url(source_url: HttpUrl, proposition_url: HttpUrl):
    """Fetch, extract claims from, and cache a source URL. Returns (source_url, vectorstore|None)."""
    if str(source_url) == str(proposition_url):
        logger.info(f"\n=> Skipping source {source_url} as it is the same as the proposition URL.")
        return source_url, None

    domain = extract_domain(source_url)
    file_path = os.path.join("faiss", domain, "index.faiss")

    # Per-domain lock prevents two threads from building the same index simultaneously
    domain_lock = _get_domain_lock(domain)
    with domain_lock:
        if os.path.isfile(file_path):
            with _GPU_LOCK:
                vs = VECTORSTORE.load_local(
                    f"faiss/{domain}",
                    EMBEDDING_MODEL,
                    allow_dangerous_deserialization=True,
                )
            return source_url, vs

        # generate_atomic_claims_from_url handles its own GPU locking per chunk
        try:
            source_claims: list[Proposition] = generate_atomic_claims_from_url(source_url)
        except Exception as e:
            logger.error(f"Failed to process source URL {source_url}: {e}")
            return source_url, None

        if not source_claims:
            return source_url, None

        logger.info(f"\n=> Found {len(source_claims)} claims in source {source_url}:\n")
        for claim in source_claims:
            logger.info(f"- {claim.claim}")

        docs: list[Document] = [
            Document(
                page_content=claim.claim,
                metadata={"source": str(claim.url), "chunk": claim.chunk}
            ) for claim in source_claims
        ]

        with _GPU_LOCK:
            vs = VECTORSTORE.from_documents(docs, EMBEDDING_MODEL)

        os.makedirs(f"faiss/{domain}", exist_ok=True)
        vs.save_local(f"faiss/{domain}")  # persist so future runs skip re-processing

        return source_url, vs


def verify_atomic_claim(
        proposition_to_check_object: Proposition
) -> tuple[PairRelation | None, set[HttpUrl]]: # pair relation, new sources used
    # https://github.com/mbzuai-nlp/fire/blob/main/eval/fire/verify_atomic_claim.py

    search_results: list[HttpUrl] = SEARCH_API.find_sources(proposition_to_check_object.claim)

    pair_relation: PairRelation | None = None

    # Build/load vectorstores for all source URLs in parallel (I/O-bound; GPU serialized via _GPU_LOCK)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(_build_vectorstore_for_url, source_url, proposition_to_check_object.url)
            for source_url in search_results
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                source_url, propositions_vectorstore = future.result()
            except Exception as e:
                logger.error(f"Error processing a source URL: {e}")
                continue

            if propositions_vectorstore is None:
                continue

            with _GPU_LOCK:
                check_against = propositions_vectorstore.similarity_search_with_score(
                    proposition_to_check_object.claim, k=5
                )

            logger.info(f"\n=> Found {len(check_against)} similar claims in source {source_url}:\n")
            for claim, score in check_against:
                logger.info(f"- {claim.page_content} (score: {score})")

            if not check_against:
                continue

            # Batch NLI: one GPU forward pass for all candidate pairs instead of N sequential calls
            premises = [c.page_content for c, _ in check_against]
            with _GPU_LOCK:
                relations = NLI_MODEL.recognise_textual_entailment_batch(
                    premises, proposition_to_check_object.claim
                )

            for i, (source_claim, score) in enumerate(check_against):
                if relations[i] == Relation.NEUTRAL:
                    continue

                pair_relation = PairRelation(
                    proposition_to_check_object=proposition_to_check_object,
                    source_proposition_object=Proposition(
                        claim=source_claim.page_content,
                        url=source_url,
                        chunk=source_claim.metadata["chunk"]
                    ),
                    relation=relations[i]
                )
                break

            if pair_relation is not None:
                # Cancel any not-yet-started futures — result found
                for f in futures:
                    f.cancel()
                break

    return pair_relation, set(search_results)

def findings(claims: list[Proposition], sources_urls: set[str], contradictions: list[PairRelation], unverified_veracities: list[int]) -> None:
    logger.info("\nClaims made by the article:")
    logger.info(f"Claims made by the article:\n{[c.claim for c in claims]}")

    if not sources_urls:
        logger.warning("The article does not contain any claims that can be verified with reliable sources.")
        return
    logger.info(f"Sources used:")
    logger.info(f"{list(sources_urls)}")

    if not contradictions:
        logger.warning("\nNo contradictions were found.")
    else:
        logger.info("\nContradictions found:")
        for pair_relation in contradictions:
            logger.info(f"Original claim: {pair_relation.proposition_to_check_object.claim}")
            logger.info(f"From chunk: {pair_relation.proposition_to_check_object.chunk}")
            logger.info(f"Source claim: {pair_relation.source_proposition_object.claim}")
            logger.info(f"From URL: {pair_relation.source_proposition_object.url}")
            logger.info(f"From chunk: {pair_relation.source_proposition_object.chunk}")

    if not unverified_veracities:
        logger.info("\nAll claims were supported by at least one reliable source.")
    else:
        logger.info("\nClaims with no entailments found:")
        for claim_index in unverified_veracities:
            logger.info(f"- {claims[claim_index]}")

def main():
    print(ASCII_ART)
    if not HARDCODED_TEXT:
        print(f"Input URL: {INPUT_URL}")
        lookup_source_bias(INPUT_URL)

    # Main algorithm
    contradictions: list[PairRelation] = []
    sources_urls = set()
    unverified_veracities: list[int] = []

    claims: list[Proposition] = generate_atomic_claims_from_url(INPUT_URL)

    # Verify all claims in parallel — SerpAPI searches and URL fetching overlap across claims
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_index = {
            executor.submit(verify_atomic_claim, claim): claim_index
            for claim_index, claim in enumerate(claims)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            claim_index = future_to_index[future]
            try:
                pair_relation, new_sources_urls = future.result()
            except Exception as e:
                logger.error(f"Error verifying claim {claim_index}: {e}")
                unverified_veracities.append(claim_index)
                continue
            sources_urls.update(new_sources_urls)

            if pair_relation is None:
                unverified_veracities.append(claim_index)
                continue
            if pair_relation.relation == Relation.CONTRADICTION:
                contradictions.append(pair_relation)

    findings(claims, sources_urls, contradictions, unverified_veracities)


if __name__ == "__main__":
    logger.info("Initialisation complete.")
    main()
