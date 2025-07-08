from pydantic import HttpUrl, BaseModel, Field
import pandas as pd
import tldextract

from utils import MBFC_PATH, Relation, ASCII_ART
from config import *

import os

from langchain.docstore.document import Document

import logging
logger = logging.getLogger(__name__)

def extract_domain(url: HttpUrl) -> str:
    extracted = tldextract.extract(str(url))
    return f"{extracted.domain}.{extracted.suffix}".lower()

def lookup_source_bias(news_url: HttpUrl) -> None:
    domain: str = extract_domain(news_url)
    bias_df: pd.DataFrame = pd.read_csv(MBFC_PATH, encoding='utf-8')
    match: pd.DataFrame = bias_df[bias_df['url'].str.contains(domain, na=False, case=False)]
    if match.empty:
        logger.warning(f"\n=> No bias information found for domain: {domain}")
    row: pd.Series = match.iloc[0]
    results: dict[str, str] = {
        'Domain': domain,
        'Bias': row.get('bias_rating', 'N/A'),
        'Factual Reporting': row.get('factual_reporting_rating', 'N/A'),
        #'Notes': f"Matched from: {row.get('site_name', '')}"
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
        full_text: str = TEXT_LOADER.load_text(str(url))
    text_chunks: list[str] = TEXT_SPLITTER.split_text(full_text)

    proposition_objects: list[Proposition] = []

    # Get claims
    for i, chunk in enumerate(text_chunks):
        logger.info(f"\n=> Processing text chunk {i}:\n{chunk}\n")
        for claim in CLAIM_EXTRACTOR.extract_claims(chunk):
            logger.info(f"Extracted claim: {claim}")
            proposition_objects.append(Proposition(claim=claim, url=url, chunk=chunk))

    # Decontextualise
    if DECONTEXTUALISE:
        for i, claim in enumerate(proposition_objects):
            before: str = "".join(c.claim for c in proposition_objects[max(0, i-5):i])
            after:  str = "".join(c.claim for c in proposition_objects[i+1:i+2])
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

def verify_atomic_claim(
        proposition_to_check_object: Proposition
) -> tuple[PairRelation | None, set[HttpUrl]]: # pair relation, new sources used
    # https://github.com/mbzuai-nlp/fire/blob/main/eval/fire/verify_atomic_claim.py

    search_results: list[HttpUrl] = SEARCH_API.find_sources(proposition_to_check_object.claim)

    pair_relation: PairRelation | None = None

    for source_url in search_results:

        if str(source_url) == str(proposition_to_check_object.url):
            logger.info(f"\n=> Skipping source {source_url} as it is the same as the proposition URL {proposition_to_check_object.url}.")
            continue

        file_path = os.path.join("faiss", extract_domain(source_url), "index.faiss")

        if os.path.isfile(file_path):
            # Get the propositions_vectorstore from the persisted directory
            propositions_vectorstore = VECTORSTORE.load_local(
                f"faiss/{extract_domain(source_url)}",
                EMBEDDING_MODEL
            )
        else:
            source_claims: list[Proposition] = generate_atomic_claims_from_url(source_url)

            docs: list[Document] = [
                Document(
                    page_content=claim.claim,
                    metadata={
                        "source": claim.url,
                        "chunk": claim.chunk
                    }
                ) for claim in source_claims
            ]

            propositions_vectorstore = VECTORSTORE.from_documents(
                docs,
                EMBEDDING_MODEL,
                collection_name=source_url,
                persist_directory=f"faiss/{extract_domain(source_url)}"
            )

            logger.info(f"\n=> Found {len(source_claims)} claims in source {source_url}:\n")
            for claim in source_claims:
                logger.info(f"- {claim.claim}")

        check_against = propositions_vectorstore.similarity_search_with_score(proposition_to_check_object.claim, k=5)

        logger.info(f"\n=> Found {len(check_against)} claims in source {source_url} that are similar to the proposition:\n")
        for claim, score in check_against:
            logger.info(f"- {claim.page_content} (score: {score})")

        if not check_against:
            continue

        for source_claim, score in check_against:
            relation: Relation = NLI_MODEL.recognise_textual_entailment(source_claim.page_content, proposition_to_check_object.claim)

            if relation == Relation.NEUTRAL:
                continue

            pair_relation = PairRelation(
                proposition_to_check_object=proposition_to_check_object,
                source_proposition_object=Proposition(
                    claim=source_claim.page_content,
                    url=source_url,
                    chunk=source_claim.metadata["chunk"]
                ),
                relation=relation
            )
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

    for claim_index, claim in enumerate(claims):
        pair_relation, new_sources_urls = verify_atomic_claim(claim)
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