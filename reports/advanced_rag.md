# Step 5: Advanced RAG Pipeline With Query Rewriting and Reranking

In this step, we build on the naive RAG framework implemented in Step 3 and add two advanced RAG features: query rewriting using Multi-Query Expansion (MQE) and cross-encoder reranking.

## Feature Selection Rationale
1. Query rewriting using Multi-Query Expansion (MQE). In our evaluation in Step 4, experiments revealed a trade-off pattern between recall and precision: increasing top-k improved coverage but introduced noise, while smaller top-k missed relevant evidence. MQE mitigates this by generating multiple paraphrases of the input question. Each paraphrase is used for retrieval, broadening coverage without permanently bloating the context window. This reduces the risk of missing critical evidence due to lexical mismatch.
2. Cross-Encoder Reranking. Step 4 also showed that simple concatenation of top passages outperformed diversification methods like MMR, highlighting the importance of strict relevance. To further improve precision, a cross-encoder model (trained on MS MARCO relevance judgments) is introduced. Unlike bi-encoder embeddings, cross-encoders jointly encode the query and passage, producing a more fine-grained relevance score. This ensures that only the most relevant passages are selected for final generation, limiting context dilution.

## Implementation Details
As with Step 3, the advanced pipeline begins with text corpus preparation. The RAG Mini Wikipedia dataset is chunked with 600-token maximum length. Embeddings are generated using the all-MiniLM-L6-v2 model, producing 384-dimensional normalized vectors. These are indexed in a FAISS IndexFlatIP structure for cosine similarity search.

For each user query, the system first invokes MQE using Flan-T5. Multiple paraphrases are produced via sampling (temperature=0.8, top_p=0.9, multiple return sequences). Rule-based cleaning and semantic deduplication filter near-identical rewrites, ensuring diverse queries. Each rewrite triggers FAISS retrieval of candidate passages, which are merged and deduplicated to form a broad candidate set.

Next, cross-encoder reranking is applied using the cross-encoder/ms-marco-MiniLM-L-6-v2 model. To allow for the 512-token limit of this model, both questions and passages are truncated (e.g., 32 tokens for the question, 480 for the passage). The cross-encoder assigns fine-grained relevance scores, and the top-N passages are retained (final_top_k=5).

Then, the pipeline compose ground context, with inline citation tags indicating the chunk ID and rerank score. This context, along with the question, is passed using a persona-style prompt to the Flan-T5 generator. The generator is instructed to provide concise, context-faithful answers, and fallback with “I don’t know” if insufficient evidence is retrieved.
