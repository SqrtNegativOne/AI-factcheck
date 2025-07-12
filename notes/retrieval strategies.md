# Hybrid Retrieval (Dense + Sparse)
Dense retrieval alone misses keyword-level matches. Sparse (BM25) misses paraphrases. Combining both gives complementary strengths.

**Implementation**:

* Use a **dense retriever** (e.g., `Contriever`, `bge-large-en`, `GTR`, etc.) for semantic similarity.
* Combine with **BM25** or TF-IDF scores (sparse signals).
* Score fusion: e.g., linear combination or reciprocal rank fusion (RRF).

**Libraries**:

* **ColBERTv2** (contextualized late interaction dense retriever).
* **Jina**, **Haystack**, or **LlamaIndex** support hybrid pipelines.

# Claim-Aware Re-ranking (Two-stage)
Initial retrieval gets top-*k*, but may include irrelevant items. Re-ranking can deeply model interaction between query and evidence.

**Implementation**:

* First: retrieve *k* candidates using vector similarity (or hybrid).
* Then: re-rank with cross-encoder model like `cross-encoder/ms-marco-MiniLM-L-6-v2`, `deberta-v3`, `MonoT5`, or `ClaimBuster`.

**Scoring**:

* Use **pointwise or pairwise scoring**, considering claim–evidence entailment or textual similarity.

# Claim-to-Evidence Embedding Models (Supervised Dense)

Pretrained models like `bge`, `GTR`, `E5`, `OpenAI ada-002` embed claims and evidence in a shared space tuned for retrieval.

**Best**:

* `intfloat/e5-large` or `bge-large-en-v1.5`
* `facebook/contriever` (unsupervised but strong for general similarity)

**Avoid**: vanilla sentence transformers trained on NLI or STS unless fine-tuned.

# Query Expansion / Reformulation
Claims are often too short or implicit. Expanding them makes retrieval stronger.

**Approaches**:

* Add predicted entities, topics, or questions via:

  * Named Entity Recognition (NER)
  * Prompted LLM expansions (e.g., “What evidence would support or refute this?”)
* Use T5/BART to generate alternative phrasings.

# Entity-Aware Filtering or Indexing
Claims involve specific named entities. Use this to reduce the retrieval search space.

**Implementation**:

* Pre-index evidence by entities (NER tags or Wikidata IDs).
* At query time, filter to documents sharing entities with the claim.
* Combine with dense retrieval over the reduced set.

# Multi-hop or Contextual Retrieval
Claims may require more than one sentence (or a chain) to verify.

**Strategy**:

* Retrieve evidence chunks or passages (instead of single sentences).
* Use retrieval-augmented generation (RAG) or passage ranking techniques.
* Encode windowed contexts (e.g., sentence ±1 in index).

# Use of Natural Language Inference (NLI) Models Post-Retrieval
Retrieval gets candidates, but NLI tells if they support/refute the claim.

**Implementation**:

* After retrieval, run each sentence through an NLI model:

  * `roberta-large-mnli`
  * `deberta-v3-mnli`
* Classify each sentence as entail, contradict, or neutral.